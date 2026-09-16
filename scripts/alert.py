#!/usr/bin/env python3
"""Independent pipeline failure alert -- stdlib only, no ashare_lab imports."""

import json
import logging
import os
import subprocess
import sys
from datetime import date, timedelta

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_STAMP = os.path.expanduser("~/.cache/ashare-alert-last.stamp")

# ponytail: weekday proxy for trading days -- no exchange_calendars
# available outside venv. Add XSHG holiday list if precision matters.
ALERT_INTERVAL_TRADING_DAYS = 3

_HERMES_BIN_CANDIDATES = (
    os.path.expanduser("~/code/hermes-agent/venv/bin/hermes"),
    os.path.expanduser("~/code/hermes-agent/hermes"),
)


def _send_qqbot(text: str) -> bool:
    """Send one text to QQBot home channel via hermes CLI."""
    hermes = next((p for p in _HERMES_BIN_CANDIDATES if os.path.isfile(p)), "")
    if not hermes:
        logger.warning("hermes CLI missing")
        return False
    try:
        r = subprocess.run(
            [hermes, "send", "-t", "qqbot", "--json"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            logger.warning("QQ send rc=%s", r.returncode)
            return False
        data = json.loads(r.stdout.decode("utf-8", errors="replace"))
        return bool(data.get("success"))
    except Exception as exc:
        logger.warning("QQ send failed: %s", exc)
        return False


def _weekdays_between(d1: date, d2: date) -> int:
    """Count weekdays (Mon-Fri) between d1 and d2, exclusive of d1."""
    if d2 <= d1:
        return 0
    count = 0
    cur = d1 + timedelta(days=1)
    while cur <= d2:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)
    return count


def _should_alert(stamp_path: str = DEFAULT_STAMP) -> bool:
    """Check frequency cap: first failure always alerts, then every 3 weekdays."""
    if not os.path.isfile(stamp_path):
        return True
    try:
        with open(stamp_path) as f:
            last = date.fromisoformat(f.read().strip())
    except (ValueError, OSError):
        return True
    return _weekdays_between(last, date.today()) >= ALERT_INTERVAL_TRADING_DAYS


def _write_stamp(stamp_path: str = DEFAULT_STAMP) -> None:
    os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
    with open(stamp_path, "w") as f:
        f.write(date.today().isoformat())


def clear_stamp(stamp_path: str | None = None) -> None:
    """Remove stamp file. Called on success to reset frequency tracking."""
    path = stamp_path or DEFAULT_STAMP
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _format_alert(
    exit_code: int, stage: str, stderr_tail: str, elapsed_s: float,
) -> str:
    lines = [
        f"[ashare-lab] {stage} failed",
        f"exit code: {exit_code}",
        f"elapsed: {elapsed_s:.0f}s",
    ]
    if stderr_tail.strip():
        lines.append("stderr (last 5 lines):")
        lines.append(stderr_tail.strip())
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--clear-stamp":
        clear_stamp()
        return

    if len(sys.argv) < 4:
        print(
            "Usage: alert.py EXIT_CODE STAGE ELAPSED_S [STDERR_FILE]",
            file=sys.stderr,
        )
        print("       alert.py --clear-stamp", file=sys.stderr)
        sys.exit(2)

    exit_code = int(sys.argv[1])
    stage = sys.argv[2]
    elapsed_s = float(sys.argv[3])

    stderr_tail = ""
    if len(sys.argv) >= 5 and os.path.isfile(sys.argv[4]):
        with open(sys.argv[4]) as f:
            stderr_tail = "\n".join(f.read().splitlines()[-5:])

    if not _should_alert():
        sys.exit(0)

    msg = _format_alert(exit_code, stage, stderr_tail, elapsed_s)
    if _send_qqbot(msg):
        _write_stamp()
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
