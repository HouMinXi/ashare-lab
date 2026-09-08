"""Tests for sync_gpu_data in scripts/ashare-data-update.sh."""

import re
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


class TestGpuCodeSyncShipsModelMeta:
    """predict.py imports model_meta; GPU sync must ship the module.

    gpu-win latest.pt is a regular-file copy. Without model_meta.py on
    that host the staleness gate always falls back to mtime.
    """

    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"
    DEPLOY = Path(SCRIPT).resolve().parent / "deploy.sh"

    def test_pipeline_sync_copies_model_meta(self):
        src = self.PIPELINE.read_text()
        fn = src[src.index("sync_code_to_gpu()"): src.index("try_gpu_inference()")]
        assert "model_meta.py" in fn, (
            "sync_code_to_gpu must copy model_meta.py with predict.py"
        )
        assert "predict.py" in fn

    def test_deploy_sync_copies_model_meta(self):
        src = self.DEPLOY.read_text()
        staging = src[src.index("Staging files for GPU deploy"): src.index("Section 8")]
        assert "model_meta.py" in staging
        assert "predict.py" in staging


class TestGpuLastConsumer:
    """After a DEMAND_START reboot, llama is not running. Restore must
    follow the last durable consumer, not the ollama default."""

    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"
    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    SWITCH = Path(SCRIPT).resolve().parent / "gpu-switch.bat"

    def _run_lib(self, snippet: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f"source {self.LIB}\n" + snippet],
            capture_output=True, text=True, timeout=10,
        )

    def test_live_llama_wins_over_stale_file(self):
        r = self._run_lib(
            'resolve_gpu_restore_target "llama-server.exe 1234" "ollama"; echo RC=$?'
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-2:] == ["llama-server", "RC=0"] or (
            r.stdout.strip().splitlines()[-1] == "llama-server"
        )

    def test_reboot_uses_last_llama_when_process_gone(self):
        r = self._run_lib('resolve_gpu_restore_target "" "llama-server"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "llama-server"

    def test_reboot_uses_last_ollama(self):
        r = self._run_lib('resolve_gpu_restore_target "" "ollama\\r"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "ollama"

    def test_unknown_or_empty_last_stays_ollama(self):
        r = self._run_lib('resolve_gpu_restore_target "" "training"')
        assert r.stdout.strip().splitlines()[-1] == "ollama"
        r2 = self._run_lib('resolve_gpu_restore_target "" ""')
        assert r2.stdout.strip().splitlines()[-1] == "ollama"

    def test_pipeline_calls_resolver_not_hardcoded_default(self):
        src = self.PIPELINE.read_text()
        fn = src[src.index("try_gpu_inference()"): src.index("gpu-switch.bat training")]
        assert "resolve_gpu_restore_target" in fn
        assert "gpu_restore_target=\"ollama\"" not in fn

    def test_switch_writes_durable_consumer_for_serve_targets(self):
        src = self.SWITCH.read_text()
        assert "gpu-last-consumer.txt" in src
        ollama = src[src.index("\n:ollama\n"): src.index("\n:llama-server\n")]
        llama = src[src.index("\n:llama-server\n"): src.index("\n:training\n")]
        training = src[src.index("\n:training\n"): src.index("\n:training-3080\n")]
        train3080 = src[src.index("\n:training-3080\n"):]
        assert "gpu-last-consumer.txt" in ollama
        assert "gpu-last-consumer.txt" in llama
        assert "gpu-last-consumer.txt" not in training
        assert "gpu-last-consumer.txt" not in train3080


class TestRetrainSkipsServeNowLive:
    """A serve-now live model (w115) must be retrained on its own window,
    never by --force over the walk-forward set, which would write w11.pt
    and displace it."""

    RETRAIN = Path(SCRIPT).resolve().parent / "ashare-retrain.sh"

    def test_serve_now_step_is_derived_not_hardcoded(self):
        """get_all_windows() stops at the last window whose test period has
        started, so the freshest trainable step is always its length. A
        literal step number silently trains a stale window once the data
        grows past the next window boundary."""
        src = self.RETRAIN.read_text()
        assert "SERVE_NOW_STEP=11" not in src, (
            "step is hardcoded; derive it from len(get_all_windows())"
        )
        assert "get_all_windows" in src

    def test_serve_now_guard_before_train(self):
        src = self.RETRAIN.read_text()
        train_at = src.index("py -m ashare_lab.research.train $TRAIN_ARGS")
        comment_at = src.index("Serve-now live models")
        fn_at = src.index("read_expected_live_model()")
        assert fn_at < comment_at < train_at
        assert "-ge 100" in src
        assert "--serve-now $SERVE_NOW_STEP --force" in src
        fn_end = src.index("\n}", fn_at) + 2
        fn = src[fn_at:fn_end]
        assert "awk" in fn
        assert "python3 -c" not in fn
        assert "2>/dev/null" not in fn

    def test_deploy_gate_uses_train_date_for_serve_now(self):
        """w12 is numerically below w115, so the numeric gate would deploy a
        staler model without complaint. Serve-now deploys compare train_date,
        and a serve-now run that cannot read both dates shelves rather than
        falling through to a gate that cannot judge it."""
        src = self.RETRAIN.read_text()
        assert "NEW_TRAIN_DATE" in src
        assert "LIVE_TRAIN_DATE" in src
        gate_at = src.index('if [ "$SERVE_NOW_LIVE" -eq 1 ]')
        numeric_at = src.index('elif [ -n "$NEW_MODEL" ]')
        assert gate_at < numeric_at
        closed = src.index('if [ -z "$NEW_TRAIN_DATE" ] || [ -z "$LIVE_TRAIN_DATE" ]')
        assert gate_at < closed < numeric_at

    def test_alerts_do_not_interpolate_values_into_python(self):
        """Alert values read off gpu-win are data, not code. Embedding
        them in the inline python source lets a quote in meta.json break
        the alert, so the shelved-model warning is lost. Shell log lines
        are unaffected; only the python -c blocks are scanned."""
        src = self.RETRAIN.read_text()
        blocks = re.findall(r"python3 -c (.)\n(.*?)\n\1", src, re.S)
        assert blocks, "no inline python blocks found"
        for _quote, body in blocks:
            assert "$" not in body, (
                f"shell value interpolated into inline python:\n{body}"
            )
        assert src.count("os.environ[") >= 4


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
        """The poll used to run a fixed 40 iterations whatever the caller
        asked for, so the budget could not bound the stall.  Comparing two
        budgets is what makes this a budget test: a fixed-count poll gives
        the same iteration count for both, a bounded one does not.

        Iterations are counted rather than seconds so the assertion does
        not depend on wall-clock timing; sleep is stubbed to keep the test
        fast, which means SECONDS advances by real time, not by the stub."""
        def run_with_budget(budget: str) -> tuple[str, int]:
            r = self._run_lib(textwrap.dedent(f"""\
                wol() {{ return 0; }}
                ping() {{ return 1; }}
                ssh() {{ return 1; }}
                iters=0
                sleep() {{ iters=$((iters+1)); command sleep 0.05; }}
                wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser {budget}
                echo "RC=$? ITERS=$iters"
            """))
            # The trailing echo sets the script's own status, so read
            # wake_gpu's return code from the output, not from returncode.
            iters = int(r.stdout.split("ITERS=")[1].split()[0])
            return r.stdout, iters

        small_out, small_iters = run_with_budget("1")
        large_out, large_iters = run_with_budget("3")

        assert "RC=1" in small_out, small_out
        assert "RC=1" in large_out, large_out
        assert large_iters > small_iters, (
            f"budget=1 ran {small_iters} iterations and budget=3 ran "
            f"{large_iters}; the poll length does not follow the budget"
        )


    def test_non_numeric_budget_rejected(self):
        """A non-numeric budget makes the deadline comparison error out and
        evaluate false, so the ping poll never runs and the wake reports
        unreachable without having waited at all.  (An EMPTY budget is not
        this case: ${4:-300} substitutes the default, which is correct.)"""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { echo "BUG: polled with a bad budget"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser "abc"
            echo "RC=$?"
        """))
        assert "RC=1" in r.stdout, r.stdout
        assert "budget must be a positive integer" in r.stdout
        assert "BUG:" not in r.stdout

    def test_zero_budget_rejected(self):
        """Zero passes a bare ^[0-9]+$ but makes the deadline false on the
        first pass, so no poll runs and the caller is told the host is
        unreachable -- a misleading message for a bad argument."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { echo "BUG: polled with a zero budget"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 0
            echo "RC=$?"
        """))
        assert "RC=1" in r.stdout, r.stdout
        assert "budget must be a positive integer" in r.stdout, r.stdout
        assert "not reachable" not in r.stdout, (
            "a bad budget must not be reported as an unreachable host"
        )
        assert "BUG:" not in r.stdout

    def test_empty_budget_uses_the_default(self):
        """Documents the boundary the guard must NOT reject."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser ""
            echo "RC=$?"
        """))
        assert "RC=0" in r.stdout, r.stdout


class TestGpuIdentityIsSharedOnce:
    """GPU_HOST/MAC/USER live in the lib so a hardware swap is one edit."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"

    def test_lib_defines_the_identity(self):
        src = self.LIB.read_text()
        for var in ("GPU_HOST=", "GPU_MAC=", "GPU_USER="):
            assert var in src, f"{var} must live in the shared lib"

    def test_callers_do_not_redefine_it(self):
        for script in (Path(SCRIPT), self.PIPELINE):
            src = script.read_text()
            for var in ("GPU_HOST=", "GPU_MAC=", "GPU_USER="):
                assert var not in src, (
                    f"{script.name} redefines {var}; the hardware address "
                    "would then have to be changed in lockstep across files"
                )

    def test_identity_is_defined_before_first_use(self):
        """The lib is sourced partway down the pipeline; if any GPU_* use
        preceded the source line the variable would be empty under set -u."""
        src = self.PIPELINE.read_text().splitlines()
        source_line = next(
            i for i, ln in enumerate(src) if "lib/gpu-wake.sh" in ln and "source" in ln
        )
        uses = [
            i for i, ln in enumerate(src)
            if ("$GPU_HOST" in ln or "${GPU_HOST}" in ln
                or "$GPU_MAC" in ln or "${GPU_USER}" in ln)
        ]
        assert uses, "expected the pipeline to use the shared identity"
        assert min(uses) > source_line, (
            f"GPU_* used at line {min(uses) + 1} before the lib is sourced "
            f"at line {source_line + 1}"
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

    def test_sync_failure_detected_without_relying_on_pipefail(self):
        """The sync output is piped through tee, and tee always exits 0.
        Reading the pipeline status would make the alert depend on pipefail
        staying set -- a future refactor into a subshell or `bash -c` would
        silently disarm it.  PIPESTATUS[0] is explicit about which command's
        status matters."""
        src = Path(SCRIPT).read_text()
        assert "PIPESTATUS[0]" in src, (
            "sync failure must be read from PIPESTATUS, not the pipeline "
            "status that tee overwrites"
        )
        assert "if ! sync_gpu_data 2>&1 | tee" not in src, (
            "pipeline-status form reintroduces the pipefail dependency"
        )

    def test_pipestatus_form_detects_failure(self):
        """Behavioural check that the chosen form works even with pipefail
        off -- the point of using PIPESTATUS in the first place."""
        r = subprocess.run(
            ["bash", "-c", textwrap.dedent("""\
                set +o pipefail
                f() { return 1; }
                f 2>&1 | tee /dev/null
                rc=${PIPESTATUS[0]}
                [ "$rc" -ne 0 ] && echo DETECTED || echo MASKED
            """)],
            capture_output=True, text=True, timeout=10,
        )
        assert "DETECTED" in r.stdout, r.stdout
