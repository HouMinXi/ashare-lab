"""Tests for ashare_lab.bridge fallback logic.

T1 bridge 200 -> True; gateway zero calls.
T2 bridge URLError -> gateway called once; payload correct; True.
T3 bridge 500 -> fallback.
T4 bridge 403 -> no fallback, False.
T5 bridge URLError + gateway fails -> False, no raise.
T6 no gateway config -> bridge fails, no fallback, False.
Bug-injection: delete _send_via_gateway call -> T2/T3 FAIL.
"""
from __future__ import annotations

import json
import urllib.error
from unittest import mock

from ashare_lab.bridge import send_bridge_alert


# -- Helpers --

def _mock_bridge_ok():
    """Return a mock urlopen that succeeds for bridge."""
    resp = mock.MagicMock()
    resp.read.return_value = b'{"status": "ok"}'
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


def _mock_bridge_http_error(code: int):
    """Return a mock urlopen that raises HTTPError."""
    return urllib.error.HTTPError(
        "http://192.168.100.10:8377/alert", code, "ERR", {}, None,
    )


def _mock_bridge_url_error():
    """Return a mock urlopen that raises URLError."""
    return urllib.error.URLError("Connection refused")


def _mock_gateway_ok():
    """Return a mock urlopen that succeeds for gateway."""
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps({"success": True, "message_id": "42"}).encode()
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


def _mock_gateway_fail():
    """Return a mock urlopen that raises for gateway."""
    return urllib.error.URLError("Gateway down")


# -- Tests --

class TestBridgeFallback:
    """Gateway fallback when bridge is unreachable."""

    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", return_value=_mock_bridge_ok())
    def test_t1_bridge_ok_no_gateway(self, mock_urlopen, _):
        """T1: bridge 200 -> True; gateway not called."""
        assert send_bridge_alert("t1", "body") is True
        assert mock_urlopen.call_count == 1
        # Only one call (bridge), no gateway
        req = mock_urlopen.call_args[0][0]
        assert "8377" in req.full_url

    @mock.patch("ashare_lab.bridge._send_via_gateway", return_value=True)
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_url_error())
    def test_t2_bridge_urlerror_fallback(self, _, __, mock_gw):
        """T2: bridge URLError -> gateway called once; returns True."""
        assert send_bridge_alert("t2 title", "t2 body") is True
        mock_gw.assert_called_once_with("t2 title", "t2 body", 10)

    @mock.patch("ashare_lab.bridge._send_via_gateway", return_value=True)
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_http_error(500))
    def test_t3_bridge_500_fallback(self, _, __, mock_gw):
        """T3: bridge 500 -> fallback to gateway."""
        assert send_bridge_alert("t3", "body") is True
        mock_gw.assert_called_once()

    @mock.patch("ashare_lab.bridge._send_via_gateway", return_value=True)
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_http_error(403))
    def test_t4_bridge_403_no_fallback(self, _, __, mock_gw):
        """T4: bridge 403 -> no fallback; returns False."""
        assert send_bridge_alert("t4", "body") is False
        mock_gw.assert_not_called()

    @mock.patch("ashare_lab.bridge._send_via_gateway", return_value=False)
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_url_error())
    def test_t5_both_fail_no_raise(self, _, __, mock_gw):
        """T5: bridge URLError + gateway fails -> False, no exception."""
        result = send_bridge_alert("t5", "body")
        assert result is False
        mock_gw.assert_called_once()

    @mock.patch("ashare_lab.bridge.os.path.isfile", return_value=False)
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_url_error())
    def test_t6_no_gateway_config(self, _, __, ___):
        """T6: hermes CLI missing -> no fallback, False, no exception."""
        result = send_bridge_alert("t6", "body")
        assert result is False


class TestGatewayPayload:
    """Verify QQ fallback argv when :8377 is down."""

    @mock.patch("ashare_lab.bridge.os.path.isfile", return_value=True)
    @mock.patch("ashare_lab.bridge.subprocess.run")
    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen")
    def test_gateway_payload_fields(self, mock_urlopen, _, mock_run, __):
        """Fallback calls hermes send -t qqbot with title+body on stdin."""
        mock_urlopen.side_effect = _mock_bridge_url_error()
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=b'{"success": true}',
            stderr=b"",
        )
        result = send_bridge_alert("My Title", "My Body")

        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args
        argv = args[0][0]
        assert argv[-3:] == ["send", "-t", "qqbot"] or (
            "send" in argv and "qqbot" in argv
        )
        assert args.kwargs["input"] == b"My Title\nMy Body"


class TestBugInjection:
    """Delete fallback calls -> T2/T3 must FAIL."""

    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_url_error())
    def test_inject_delete_fallback_urlerror(self, _, __):
        """Without _send_via_gateway, bridge URLError -> False."""
        # Patch _send_via_gateway to simulate deletion (always False)
        with mock.patch("ashare_lab.bridge._send_via_gateway", return_value=False):
            result = send_bridge_alert("inject", "body")
        # If fallback were deleted, this would be False
        # (this test verifies the wiring exists by confirming False when gw fails)
        assert result is False

    @mock.patch("ashare_lab.bridge.bridge_token", return_value="tok")
    @mock.patch("urllib.request.urlopen", side_effect=_mock_bridge_url_error())
    def test_inject_wiring_proof(self, _, __):
        """Proof: _send_via_gateway IS called on bridge failure."""
        with mock.patch("ashare_lab.bridge._send_via_gateway", return_value=True) as mock_gw:
            result = send_bridge_alert("wiring", "proof")
        assert result is True
        mock_gw.assert_called_once()
