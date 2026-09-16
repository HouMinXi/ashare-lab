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

Outbound is QQ via X500 alert-bridge (:8377) then hermes send --to qqbot.
iLink / weixin send is not used.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_BRIDGE_URL = "http://192.168.100.10:8377/alert"
_BRIDGE_TOKEN_PATHS = (
    r"H:\.secrets\bridge-token",  # gpu-win
    os.path.expanduser("~/.secrets/bridge-token"),  # POSIX
)

_HERMES_BIN_CANDIDATES = (
    os.path.expanduser("~/code/hermes-agent/venv/bin/hermes"),
    os.path.expanduser("~/code/hermes-agent/hermes"),
)


def _read_secret(env_key: str, paths: tuple[str, ...]) -> str:
    """Read a secret: env first, then file paths.  Empty if unavailable."""
    val = os.environ.get(env_key, "").strip()
    if val:
        return val
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            continue
    return ""


def bridge_token() -> str:
    """Read the alert-bridge shared token: env first, then token files.

    Returns empty string if no token is available.
    """
    return _read_secret("X_BRIDGE_TOKEN", _BRIDGE_TOKEN_PATHS)


def send_qqbot_text(text: str, timeout: int = 30) -> bool:
    """Send one text to QQBot home channel via hermes CLI. Never raises."""
    hermes = next((p for p in _HERMES_BIN_CANDIDATES if os.path.isfile(p)), "")
    if not hermes:
        logger.warning("bridge: hermes CLI missing, skip QQ send")
        return False
    try:
        r = subprocess.run(
            [hermes, "send", "-t", "qqbot", "--json"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[:200]
            logger.warning("bridge: QQ send rc=%s: %s", r.returncode, err)
            return False
        data = json.loads(r.stdout.decode("utf-8", errors="replace"))
        if data.get("success"):
            logger.info("bridge: QQ sent")
            return True
        logger.warning("bridge: QQ reply success=false: %s", data)
        return False
    except Exception as exc:
        logger.warning("bridge: QQ send failed: %s", exc)
        return False


def _send_via_gateway(title: str, body: str, timeout: int) -> bool:
    """Fallback when :8377 is down: hermes send --to qqbot."""
    return send_qqbot_text(f"{title}\n{body}", timeout=timeout)


def send_bridge_alert(title: str, body: str, timeout: int = 10) -> bool:
    """Send one alert via the bridge.  Returns True on success.

    Fail-open: logs warning and returns False on any error.  Never raises.
    Fallback: on transport failure (URLError/timeout/5xx), hermes send qqbot.
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
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            # 4xx: bridge alive but rejecting; don't fallback (avoid duplicates)
            logger.warning("bridge: alert rejected (HTTP %d): %s", exc.code, exc)
            return False
        # 5xx: bridge unhealthy; try fallback
        logger.warning("bridge: alert failed (HTTP %d), trying gateway fallback: %s", exc.code, exc)
        return _send_via_gateway(title, body, timeout)
    except Exception as exc:
        # URLError / timeout / other transport failure; try fallback
        logger.warning("bridge: alert failed, trying gateway fallback: %s", exc)
        return _send_via_gateway(title, body, timeout)
