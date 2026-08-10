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
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_BRIDGE_URL = "http://192.168.100.10:8377/alert"
_BRIDGE_TOKEN_PATHS = (
    r"H:\.secrets\bridge-token",  # gpu-win
    os.path.expanduser("~/.secrets/bridge-token"),  # POSIX
)

_GATEWAY_URL = "http://192.168.100.10:8642/api/weixin/send"
_GATEWAY_TOKEN_PATHS = (
    r"H:\.secrets\hermes-gateway-token",
    os.path.expanduser("~/.secrets/hermes-gateway-token"),
)
_GATEWAY_CHAT_ID_PATHS = (
    r"H:\.secrets\hermes-weixin-chat-id",
    os.path.expanduser("~/.secrets/hermes-weixin-chat-id"),
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


def _send_via_gateway(title: str, body: str, timeout: int) -> bool:
    """Fallback: send alert via hermes-gateway /api/weixin/send."""
    gw_token = _read_secret("HERMES_GATEWAY_TOKEN", _GATEWAY_TOKEN_PATHS)
    chat_id = _read_secret("HERMES_WEIXIN_CHAT_ID", _GATEWAY_CHAT_ID_PATHS)
    if not gw_token or not chat_id:
        logger.warning("bridge: gateway config missing (token=%s, chat_id=%s), skipping fallback",
                       bool(gw_token), bool(chat_id))
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gw_token}",
    }
    payload = json.dumps({
        "chat_id": chat_id,
        "message": f"{title}\n{body}",
    }).encode("utf-8")
    req = urllib.request.Request(_GATEWAY_URL, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("success"):
            logger.info("bridge: fallback alert sent via gateway")
            return True
        logger.warning("bridge: gateway returned success=false: %s", result)
        return False
    except Exception as exc:
        logger.warning("bridge: gateway fallback failed: %s", exc)
        return False


def send_bridge_alert(title: str, body: str, timeout: int = 10) -> bool:
    """Send one alert via the bridge.  Returns True on success.

    Fail-open: logs warning and returns False on any error.  Never raises.
    Fallback: on transport failure (URLError/timeout/5xx), try hermes-gateway.
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
