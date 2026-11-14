"""The deadman must not read an unanswered probe as a healthy host."""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deadman-check.sh"


def _run(tmp_path: Path, payload: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text('#!/bin/bash\ncat "$FAKE_PROBE"\n')
    ssh.chmod(0o755)
    scp = bin_dir / "scp"
    scp.write_text("#!/bin/bash\nexit 0\n")
    scp.chmod(0o755)

    probe = tmp_path / "probe.txt"
    probe.write_text(payload)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_PROBE": str(probe),
        "ASHARE_TARGET_HOST": "127.0.0.1",
        "ASHARE_TARGET_USER": "nobody",
        "ASHARE_DEADMAN_STATE": str(tmp_path / "state"),
        "ASHARE_HERMES_BIN": "/bin/false",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def _healthy(**overrides) -> str:
    fields = {
        "repo": "ok",
        "last_success": _days_ago(1),
        "failed_units": "0",
        "ic_last": _days_ago(7),
        "recent_bad": "0",
        "latest_fresh": "1",
    }
    fields.update(overrides)
    return "".join(f"{k}={v}\n" for k, v in fields.items() if v is not None)


def test_a_complete_probe_reports_healthy(tmp_path):
    result = _run(tmp_path, _healthy())

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_missing_freshness_field_is_not_health(tmp_path):
    """A locked database makes sqlite print nothing; that is not 'fresh'."""
    result = _run(tmp_path, _healthy(latest_fresh=None))

    assert result.returncode == 1
    assert "freshness unknown" in result.stdout


def test_a_missing_unit_count_is_not_health(tmp_path):
    result = _run(tmp_path, _healthy(failed_units=None))

    assert result.returncode == 1
    assert "unit state unknown" in result.stdout


def test_an_unparseable_date_is_not_fresh(tmp_path):
    """`date -d` failing fell back to now, so corruption looked like freshness."""
    result = _run(tmp_path, _healthy(ic_last="not-a-date"))

    assert result.returncode == 1
    assert "date unreadable" in result.stdout


def test_a_stale_but_parseable_date_still_alerts(tmp_path):
    result = _run(tmp_path, _healthy(ic_last=_days_ago(21)))

    assert result.returncode == 1
    assert "IC history last updated" in result.stdout
