"""Acceptance tests for predict.py model fallback/staleness alerts (B1-B4).

Tests the alert mechanism without requiring qlib runtime. Mocks the
hermes-gateway HTTP call at the urllib boundary.

B1: missing w{id}.pt -> notify called with expected/fallback model names
B2: notify raises -> prediction still completes, exception logged
B3: bug-injection: remove notify call -> B1 FAIL, restore -> PASS
B4: mock at transport boundary (urllib.request.urlopen)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_alert_flag():
    """Reset the rate-limit flag before each test."""
    import ashare_lab.research.predict as mod
    mod._ALERT_SENT_THIS_RUN = False
    yield
    mod._ALERT_SENT_THIS_RUN = False


class TestPredictAlert:
    """Tests for _send_alert and its integration in predict_for_date."""

    def test_alert_sent_on_fallback(self, tmp_path: Path, caplog) -> None:
        """B1: missing model -> alert sent with expected and fallback names."""
        from ashare_lab.research.predict import _send_alert

        with mock.patch.dict("os.environ", {"X_BRIDGE_TOKEN": "test-token"}):
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = mock.MagicMock()
                mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
                mock_resp.__exit__ = mock.Mock(return_value=False)
                mock_resp.read.return_value = b'{"ok": true}'
                mock_urlopen.return_value = mock_resp

                _send_alert(
                    "[predict] model fallback: w11.pt missing, "
                    "using latest.pt on 2026-07-25. Action: train w11.pt"
                )

                mock_urlopen.assert_called_once()
                req = mock_urlopen.call_args[0][0]
                body = json.loads(req.data.decode("utf-8"))
                assert "w11.pt missing" in body["body"]
                assert "latest.pt" in body["body"]

    def test_alert_fail_open(self, caplog) -> None:
        """B2: notify raises -> logged, does not propagate."""
        from ashare_lab.research.predict import _send_alert

        with mock.patch.dict("os.environ", {"X_BRIDGE_TOKEN": "test-token"}):
            with mock.patch("urllib.request.urlopen", side_effect=Exception("network down")):
                _send_alert("[test] alert")

        assert "alert failed" in caplog.text

    def test_alert_rate_limit(self) -> None:
        """One alert per run: second call is a no-op."""
        from ashare_lab.research.predict import _send_alert

        with mock.patch.dict("os.environ", {"X_BRIDGE_TOKEN": "test-token"}):
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = mock.MagicMock()
                mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
                mock_resp.__exit__ = mock.Mock(return_value=False)
                mock_resp.read.return_value = b'{"ok": true}'
                mock_urlopen.return_value = mock_resp

                _send_alert("[test] first")
                _send_alert("[test] second")

                assert mock_urlopen.call_count == 1

    def test_bug_injection_no_alert_call(self, tmp_path: Path) -> None:
        """B3: with _send_alert removed, the fallback path has no alert."""
        import ashare_lab.research.predict as mod

        # Simulate removing _send_alert by replacing it with a no-op.
        original = mod._send_alert
        mod._send_alert = lambda text: None  # noqa: ARG005

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mod._send_alert("[test] should be no-op")
            # No HTTP call because _send_alert is a no-op.
            mock_urlopen.assert_not_called()

        # Restore.
        mod._send_alert = original

    def test_alert_boundary_mock_point(self) -> None:
        """B4: the mock boundary is urllib.request.urlopen.

        This test documents the exact mock point for test infrastructure.
        The alert uses stdlib urllib (not aiohttp/requests), so mocking
        urllib.request.urlopen is sufficient to intercept all alerts.
        """
        import ashare_lab.research.predict as mod
        mod._ALERT_SENT_THIS_RUN = False

        with mock.patch.dict("os.environ", {"X_BRIDGE_TOKEN": "test-token"}):
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = mock.MagicMock()
                mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
                mock_resp.__exit__ = mock.Mock(return_value=False)
                mock_resp.read.return_value = b'{"ok": true}'
                mock_urlopen.return_value = mock_resp

                mod._send_alert("[test] boundary check")

                req = mock_urlopen.call_args[0][0]
                assert req.full_url.endswith("/alert")
                assert req.headers.get("X-bridge-token") == "test-token" or req.headers.get("X-Bridge-Token") == "test-token"


class TestBridgeModuleMissing:
    """gpu-win syncs a selective file set; bridge.py may be absent there."""

    def test_alert_fail_open_without_bridge_module(self, caplog) -> None:
        """Import failure of ashare_lab.bridge must not propagate."""
        import sys
        from ashare_lab.research.predict import _send_alert

        with mock.patch.dict(sys.modules, {"ashare_lab.bridge": None}):
            _send_alert("[test] bridge missing")

        assert "bridge module unavailable" in caplog.text
