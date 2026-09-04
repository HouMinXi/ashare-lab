"""Tests for sync_gpu_data in scripts/ashare-data-update.sh."""

import subprocess
import textwrap
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "ashare-data-update.sh")

# Preamble: source the script with testing guard, override externals.
# wake_gpu is stubbed by default: the real one sends a magic packet and
# polls for 200s.  Tests that care about the wake override the stub.
_PREAMBLE = textwrap.dedent(f"""\
    export ASHARE_DATA_UPDATE_TESTING=1
    export HOME="${{TMPDIR:-/tmp}}"
    source {SCRIPT}
    wake_gpu() {{ return 0; }}
""")


def _run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", _PREAMBLE + snippet],
        capture_output=True, text=True, timeout=10,
    )


class TestSyncFunctionExists:
    def test_declared(self):
        r = _run("declare -f sync_gpu_data >/dev/null")
        assert r.returncode == 0, r.stderr


class TestSyncCallOrder:
    def test_tar_scp_ssh_in_order(self, tmp_path):
        log = tmp_path / "call.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            tar() {{ echo "tar" >> {log}; command tar "$@"; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f tar scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        calls = log.read_text().strip().splitlines()
        assert calls == ["tar", "scp", "ssh"]


class TestSyncHappyPath:
    def test_creates_and_cleans_tarball(self, tmp_path):
        """Verify tar creates the archive and trap cleans it on return."""
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() {{ return 0; }}
            ssh() {{ return 0; }}
            export -f scp ssh
            sync_gpu_data
            [ ! -f "$HOME/.cache/ashare-sync/cn_data_sync.tar.gz" ] && [ ! -f /tmp/cn_data_sync.tar.gz ] && echo CLEANED
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        assert "CLEANED" in r.stdout


class TestSyncScratchLocation:
    def test_tar_uses_disk_cache_not_tmp(self, tmp_path):
        """Verify tar command targets ~/.cache/ashare-sync and not /tmp."""
        log = tmp_path / "tar_target.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            tar() {{
                echo "$2" > {log}
                command tar "$@"
            }}
            scp() {{ return 0; }}
            ssh() {{ return 0; }}
            export -f tar scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        target = log.read_text().strip()
        assert target.endswith(".cache/ashare-sync/cn_data_sync.tar.gz")
        assert not target.startswith("/tmp/cn_data_sync.tar.gz")


class TestSyncTarFailure:
    def test_returns_nonzero(self):
        r = _run(textwrap.dedent("""\
            tar() { return 1; }
            export -f tar
            sync_gpu_data
        """))
        assert r.returncode != 0


class TestSyncScpFailure:
    def test_returns_nonzero(self, tmp_path):
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() {{ return 1; }}
            ssh() {{ echo "BUG: ssh should not be called"; return 0; }}
            export -f scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0
        assert "ssh should not be called" not in r.stdout


class TestSyncSshFailure:
    def test_returns_nonzero(self, tmp_path):
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() {{ return 0; }}
            ssh() {{ return 1; }}
            export -f scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0


class TestSyncPlacement:
    def test_function_outside_guard(self):
        """sync_gpu_data must be defined before the testing guard."""
        src = Path(SCRIPT).read_text()
        func_pos = src.index("sync_gpu_data()")
        guard_pos = src.index('ASHARE_DATA_UPDATE_TESTING')
        assert func_pos < guard_pos, (
            "sync_gpu_data must be defined outside the testing guard"
        )


class TestSyncWakesGpu:
    """The GPU sleeps between runs; without a wake, scp hits 'No route to
    host', the sync is skipped, and the 18:00 pipeline records a stale run
    (2026-09-02 incident).  These tests pin the wake to the sync path."""

    def test_wake_called_before_scp(self, tmp_path):
        log = tmp_path / "order.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            wake_gpu() {{ echo "wake" >> {log}; return 0; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f wake_gpu scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        calls = log.read_text().strip().splitlines()
        assert calls[0] == "wake", f"wake must run first, got {calls}"
        assert "scp" in calls and calls.index("wake") < calls.index("scp")

    def test_wake_failure_aborts_sync(self, tmp_path):
        """A sleeping GPU must fail the sync loudly, not scp into the void."""
        log = tmp_path / "calls.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            wake_gpu() {{ return 1; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f wake_gpu scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0, "sync must fail when the GPU will not wake"
        assert not log.exists(), "no transfer may be attempted after wake failure"


class TestSharedWakeLib:
    """One wake implementation, sourced by both scripts (no copy-paste)."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"

    def test_lib_defines_wake_gpu(self):
        assert "wake_gpu()" in self.LIB.read_text()

    def test_both_scripts_source_the_lib(self):
        for script in (Path(SCRIPT), self.PIPELINE):
            assert "lib/gpu-wake.sh" in script.read_text(), (
                f"{script.name} must source the shared wake lib"
            )

    def test_pipeline_has_no_inline_wake(self):
        """The pipeline's own WoL block was removed, not left duplicated."""
        src = self.PIPELINE.read_text()
        assert 'wol "$GPU_MAC"' not in src, (
            "inline wol call still present -- wake logic is duplicated"
        )


class TestWakeLibBehavior:
    """Direct tests of the shared lib, independent of the sync path."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"

    def _run_lib(self, snippet: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f"source {self.LIB}\n" + snippet],
            capture_output=True, text=True, timeout=10,
        )

    def test_missing_wol_binary_fails(self):
        """A missing wol used to print to stderr and carry on: the ping poll
        still succeeded against an already-awake host, so a box that never
        got a magic packet looked healthy.  Found by running the real path
        on a machine without wol installed."""
        r = self._run_lib(textwrap.dedent("""\
            PATH=/nonexistent
            command() { return 1; }
            ping() { return 0; }
            ssh() { return 0; }
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode != 0, (
            "wake must fail when wol is absent, not proceed to the ping poll"
        )
        assert "wol not installed" in r.stdout

    def test_wake_succeeds_when_host_answers(self):
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"

    def test_wol_send_failure_is_reported(self):
        """An unchecked wol turns 'the packet never went out' into the
        generic unreachable error 200 seconds later."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 1; }
            ping() { echo "BUG: ping ran after wol failed"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode != 0
        assert "wol failed" in r.stdout, r.stdout
        assert "BUG:" not in r.stdout, "must not poll after a failed send"

    def test_ping_poll_respects_the_budget(self):
        """The poll used to run a fixed 40 iterations (200s) whatever the
        caller asked for, so a small budget could not bound the stall.
        Measured before the fix: budget=5 still ran all 40 iterations."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { return 1; }
            ssh() { return 1; }
            iters=0
            sleep() { iters=$((iters+1)); command sleep 0.05; }
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 1
            echo "RC=$? ITERS=$iters"
        """))
        # The trailing echo sets the script's own status, so read wake_gpu's
        # return code from the output rather than from returncode.
        assert "RC=1" in r.stdout, r.stdout
        iters = int(r.stdout.split("ITERS=")[1].split()[0])
        assert iters < 40, (
            f"ping poll ran {iters} iterations on a 1s budget; "
            "the budget does not bound the poll"
        )


class TestSyncFailureAlerts:
    """Sync failure used to exit 0 silently; the only visible symptom was a
    stale pipeline run 15 minutes later, pointing at the wrong layer."""

    def test_alert_invoked_on_sync_failure(self):
        src = Path(SCRIPT).read_text()
        idx = src.index("WARNING: GPU data sync failed")
        tail = src[idx:idx + 400]
        assert "alert.py" in tail, (
            "GPU sync failure must raise an alert, not just echo a warning"
        )
        assert "gpu_data_sync" in tail, (
            "alert must name the failing stage so it is not confused "
            "with a data_update or pipeline failure"
        )

    def test_alert_gets_the_sync_log_not_the_fetch_log(self):
        """alert.py reports the tail of the log it is handed.  Passing
        STDERR_LOG (fetch-today's output) means the alert names a failed
        sync without saying which step broke."""
        src = Path(SCRIPT).read_text()
        idx = src.index("gpu_data_sync")
        call = src[idx:idx + 200]
        assert "SYNC_LOG" in call, "alert must carry the sync's own log"
        assert "STDERR_LOG" not in call, (
            "STDERR_LOG holds fetch-today output, not sync diagnostics"
        )

    def test_sync_failure_survives_the_tee_pipe(self):
        """The sync output is piped through tee; without pipefail the
        pipeline's status would be tee's (always 0) and the alert would
        never fire."""
        src = Path(SCRIPT).read_text()
        assert "set -uo pipefail" in src, (
            "piping sync output through tee requires pipefail, or the "
            "failure branch becomes unreachable"
        )
