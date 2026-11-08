"""The IC freshness check must allow for the t+5 scoring lag.

A score can only be judged once five sessions of forward return exist, so the
newest entry in ic_history.tsv is five trading days old even when everything
works -- ten calendar days across two weekends. A five-day bound turned a
healthy weekend into an alert.
"""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

SCRIPT = Path("scripts/deadman-check.sh")


def _run_with_fake_probe(tmp_path: Path, ic_last: str) -> subprocess.CompletedProcess:
    """Run the watchdog against a stub ssh that returns a crafted probe line."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/bin/bash\n"
        'echo "repo=ok"\n'
        f'echo "last_success={_days_ago(1)}"\n'
        'echo "failed_units=0"\n'
        f'echo "ic_last={ic_last}"\n'
        'echo "recent_bad=0"\n'
        'echo "latest_fresh=1"\n'
    )
    ssh.chmod(0o755)
    scp = bin_dir / "scp"
    scp.write_text("#!/bin/bash\nexit 0\n")
    scp.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ASHARE_TARGET_HOST": "127.0.0.1",
        "ASHARE_TARGET_USER": "nobody",
        "ASHARE_DEADMAN_STATE": str(tmp_path / "state"),
        "ASHARE_HERMES_BIN": "/bin/false",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _days_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def test_an_ic_series_five_sessions_behind_is_healthy(tmp_path):
    """Nine calendar days: the normal state on a Sunday."""
    result = _run_with_fake_probe(tmp_path, _days_ago(9))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "IC history" not in result.stdout


def test_a_genuinely_frozen_ic_series_alerts(tmp_path):
    result = _run_with_fake_probe(tmp_path, _days_ago(21))

    assert result.returncode == 1
    assert "IC history last updated" in result.stdout
