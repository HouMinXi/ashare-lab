#!/usr/bin/env python3
"""Independent pipeline failure alert -- stdlib only, no ashare_lab imports."""

import base64
import json
import logging
import os
import secrets
import struct
import subprocess
import sys
import urllib.request
import uuid
from datetime import date, timedelta

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# iLink constants (inlined from report.py, not imported)
ILINK_URL = "https://ilinkai.weixin.qq.com/ilink/bot/sendmessage"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_ID = "bot"
ILINK_APP_CLIENT_VERSION = str((2 << 16) | (2 << 8) | 0)
ITEM_TEXT = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

DEFAULT_STAMP = os.path.expanduser("~/.cache/ashare-alert-last.stamp")

# ponytail: weekday proxy for trading days -- no exchange_calendars
# available outside venv. Add XSHG holiday list if precision matters.
ALERT_INTERVAL_TRADING_DAYS = 3


def _get_pass(key: str) -> str:
    """Read a secret from pass(1)."""
    r = subprocess.run(
        ["pass", "show", key],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return r.stdout.strip()


def _random_wechat_uin() -> str:
    val = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(val).encode()).decode()


def _ilink_headers(token: str, body: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": ILINK_APP_CLIENT_VERSION,
        "Authorization": f"Bearer {token}",
    }


def _send_ilink(token: str, chat_id: str, text: str) -> bool:
    """Send a text message via iLink. Returns True on success."""
    payload = {
        "msg": {
            "from_user_id": "",
            "to_user_id": chat_id,
            "client_id": str(uuid.uuid4()),
            "message_type": MSG_TYPE_BOT,
            "message_state": MSG_STATE_FINISH,
            "item_list": [
                {"type": ITEM_TEXT, "text_item": {"text": text}},
            ],
        },
        "base_info": {"channel_version": CHANNEL_VERSION},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    headers = _ilink_headers(token, body)
    req = urllib.request.Request(
        ILINK_URL,
        data=body.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return True
            logger.warning("iLink HTTP %d", resp.status)
            return False
    except Exception as exc:
        logger.warning("iLink send failed: %s", exc)
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

    try:
        token = _get_pass("ashare/weixin-token")
        chat_id = _get_pass("ashare/weixin-chat-id")
    except Exception as exc:
        logger.warning("Cannot read secrets: %s", exc)
        sys.exit(1)

    msg = _format_alert(exit_code, stage, stderr_tail, elapsed_s)
    if _send_ilink(token, chat_id, msg):
        _write_stamp()
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
