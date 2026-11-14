"""The deadman must fail loudly, and it must fail on its own machine.

It was rewritten in a backup directory and ran there for three weeks while the
repository kept an older copy, so these tests exist as much to pin the behaviour
as to prove the file in the tree is the one that runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/deadman-check.sh")
PROBE = Path("scripts/ashare-probe-remote.sh")


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        # A closed port answers instantly, so the reachability branch is
        # exercised without waiting on a real host.
        "ASHARE_TARGET_HOST": "127.0.0.1",
        "ASHARE_TARGET_USER": "nobody",
        "ASHARE_DEADMAN_STATE": str(tmp_path / "state"),
        "ASHARE_HERMES_BIN": "/bin/false",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


def test_an_unreachable_host_alerts_locally_and_exits_1(tmp_path):
    result = _run(tmp_path, "--dry-run")

    assert result.returncode == 1, result.stdout + result.stderr
    log = (tmp_path / "state" / "deadman.log").read_text()
    assert "FAIL ssh unreachable" in log
    # The alert is rendered on this machine; routing it through the host being
    # checked is what made the previous version silent during an outage.
    assert "DRY-RUN alert" in result.stdout


def test_the_state_directory_stays_out_of_tmp(tmp_path):
    _run(tmp_path, "--dry-run")

    assert (tmp_path / "state" / "deadman.log").exists()
    assert "/tmp/" not in SCRIPT.read_text()


def test_an_unknown_flag_is_rejected(tmp_path):
    result = _run(tmp_path, "--nonsense")

    assert result.returncode == 2
    assert "usage" in result.stderr.lower()


def test_the_probe_ships_beside_the_watchdog():
    """The watchdog uploads a sibling probe; a missing one degrades silently."""
    assert PROBE.exists()
    text = PROBE.read_text()
    assert "pipeline_runs" in text
    assert "ic_history.tsv" in text


def test_both_scripts_are_executable():
    for path in (SCRIPT, PROBE):
        assert os.access(path, os.X_OK), f"{path} is not executable"
