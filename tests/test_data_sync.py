"""Tests for sync_gpu_data in scripts/ashare-data-update.sh."""

import subprocess
import textwrap
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "ashare-data-update.sh")

# Preamble: source the script with testing guard, override externals
_PREAMBLE = textwrap.dedent(f"""\
    export ASHARE_DATA_UPDATE_TESTING=1
    export HOME="${{TMPDIR:-/tmp}}"
    source {SCRIPT}
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
            [ ! -f /tmp/cn_data_sync.tar.gz ] && echo CLEANED
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        assert "CLEANED" in r.stdout


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
