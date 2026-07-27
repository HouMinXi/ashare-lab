"""Shared alert-bridge sender for ashare-lab.

Transport: POST http://192.168.100.10:8377/alert
Header: X-Bridge-Token
Body: {"title": str, "body": str}

Token resolution (shared by all callers):
  1. X_BRIDGE_TOKEN environment variable
  2. ~/.secrets/bridge-token (POSIX)
  3. H:\\.secrets\\bridge-token (gpu-win)

Uses stdlib urllib to avoid importing requests/aiohttp in the predict
path.  Callers that need chunking split externally and call this per
chunk.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_BRIDGE_URL = "http://192.168.100.10:8377/alert"
_BRIDGE_TOKEN_PATHS = (
    r"H:\.secrets\bridge-token",  # gpu-win
    os.path.expanduser("~/.secrets/bridge-token"),  # POSIX
)


def bridge_token() -> str:
    """Read the alert-bridge shared token: env first, then token files.

    Returns empty string if no token is available.
    """
    tok = os.environ.get("X_BRIDGE_TOKEN", "").strip()
    if tok:
        return tok
    for p in _BRIDGE_TOKEN_PATHS:
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            continue
    return ""


def send_bridge_alert(title: str, body: str, timeout: int = 10) -> bool:
    """Send one alert via the bridge.  Returns True on success.

    Fail-open: logs warning and returns False on any error.  Never raises.
    """
    token = bridge_token()
    if not token:
        logger.warning("bridge: no token available, skipping alert")
        return False

    headers = {
        "Content-Type": "application/json",
        "X-Bridge-Token": token,
    }
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(_BRIDGE_URL, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            logger.info("bridge: alert sent (%s)", resp.read().decode("utf-8")[:100])
        return True
    except Exception as exc:
        logger.warning("bridge: alert failed: %s", exc)
        return False
