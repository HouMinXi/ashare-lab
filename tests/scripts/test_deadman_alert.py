"""The alert has to leave the machine, and a failed send has to stay retryable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deadman-check.sh"


def _setup(tmp_path: Path, hermes_ok: bool):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/bash\nexit 255\n')  # unreachable host
    (bin_dir / "ssh").chmod(0o755)

    marker = tmp_path / "sent.txt"
    hermes = tmp_path / "hermes"
    hermes.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> "{marker}"\n'
        f"exit {0 if hermes_ok else 1}\n"
    )
    hermes.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ASHARE_TARGET_HOST": "192.0.2.1",
        "ASHARE_TARGET_USER": "nobody",
        "ASHARE_DEADMAN_STATE": str(tmp_path / "state"),
        "ASHARE_HERMES_BIN": str(hermes),
    }
    return env, marker


def _run(tmp_path: Path, hermes_ok: bool):
    env, marker = _setup(tmp_path, hermes_ok)
    result = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=120
    )
    return result, marker, tmp_path / "state" / "last-alert.stamp"


def test_a_real_alert_is_delivered_through_the_configured_outlet(tmp_path):
    """Without --dry-run the message must actually be handed to the outlet.

    Re-pointing this at the monitored host is the failure the script was
    rewritten to remove; a test that only reads the rendered dry-run text would
    not notice if it came back.
    """
    result, marker, stamp = _run(tmp_path, hermes_ok=True)

    assert result.returncode == 1
    assert marker.exists(), "the outlet was never invoked"
    assert "send" in marker.read_text()
    assert "host unreachable" in marker.read_text()
    assert stamp.exists(), "a delivered alert must be stamped so it is not repeated"


def test_a_failed_delivery_leaves_no_stamp_so_it_retries(tmp_path):
    result, marker, stamp = _run(tmp_path, hermes_ok=False)

    assert marker.exists(), "the outlet was never invoked"
    assert not stamp.exists(), "a failed delivery must not suppress the next attempt"
    assert "could not deliver" in result.stderr.lower()
