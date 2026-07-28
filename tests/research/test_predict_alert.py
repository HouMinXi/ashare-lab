"""Tests for predict model-fallback alert exemption (P3)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predict_env(tmp_path: Path, window_id: int = 11,
                      expected: str | None = "w10",
                      latest_target: str = "w10.pt",
                      window_exists: bool = False,
                      use_copy: bool = False):
    """Set up a minimal environment for testing the fallback logic.

    Production layout (X500):
      models/latest.pt -> w10.pt   (symlink)
      models/w10.pt                (actual model)
      models/w11.pt                (missing = shelved)

    Production layout (gpu-win):
      models/latest.pt             (copy of w10.pt)
      models/w10.pt                (actual model)
      models/w11.pt                (missing = shelved)

    Returns (models_dir, cfg_dict).
    """
    models = tmp_path / "models"
    models.mkdir()

    # Create the target model that latest.pt will point to.
    # latest_target defaults to "w10.pt" to match expected.
    target = models / latest_target
    target.write_text("fake model content for testing")

    # latest.pt -> symlink or copy of target
    latest = models / "latest.pt"
    if use_copy:
        shutil.copy2(target, latest)
    else:
        latest.symlink_to(target)

    # Window model -- only create if window_exists=True
    if window_exists:
        w_path = models / f"w{window_id}.pt"
        w_path.write_text("fake model content for testing")

    cfg: dict = {}
    if expected:
        cfg = {"research": {"expected_live_model": expected}}

    return models, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("ashare_lab.research.predict._send_alert")
def test_window_equals_expected_alerts(mock_alert, tmp_path):
    """(a) window model == expected -> genuine missing -> alert fires.

    Window 10, expected w10, w10.pt missing, latest.pt -> w9.pt (not w10).
    This is a genuine missing model because latest.pt does NOT point to w10.
    """
    models, cfg = _make_predict_env(tmp_path, window_id=10,
                                    expected="w10",
                                    latest_target="w9.pt",
                                    window_exists=False)

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 10, "test_start": "2025-01-01",
                "test_end": "2025-06-30", "train_start": "2022-01-01",
                "train_end": "2024-12-31"}]
    model_path, win = _resolve_model_path("2025-03-15", None, windows, cfg, models)

    mock_alert.assert_called_once()
    assert "Action: train w10.pt" in mock_alert.call_args[0][0]
    assert win["window_id"] == 10


@patch("ashare_lab.research.predict._send_alert")
def test_deliberate_mismatch_no_alert(mock_alert, tmp_path, caplog):
    """(b) window!=expected + fallback==expected -> NO alert, info log."""
    models, cfg = _make_predict_env(tmp_path, window_id=11,
                                    expected="w10", window_exists=False)

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    caplog.set_level(logging.INFO, logger="ashare_lab.research.predict")
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_not_called()
    assert "not deployed (expected_live_model=w10)" in caplog.text
    assert win["window_id"] == 11


@patch("ashare_lab.research.predict._send_alert")
def test_fallback_not_expected_alerts(mock_alert, tmp_path):
    """(c) fallback != expected -> alert fires (someone swapped latest.pt)."""
    models, cfg = _make_predict_env(tmp_path, window_id=11,
                                    expected="w10",
                                    latest_target="w9.pt",
                                    window_exists=False)

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_called_once()
    assert "Action: train w11.pt" in mock_alert.call_args[0][0]


@patch("ashare_lab.research.predict._send_alert")
def test_expected_file_missing_alerts(mock_alert, tmp_path):
    """(d) expected model file itself missing -> alert fires."""
    models = tmp_path / "models"
    models.mkdir()

    latest = models / "latest.pt"
    latest.write_text("fake model")

    cfg = {"research": {"expected_live_model": "w10"}}

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_called_once()


@patch("ashare_lab.research.predict._send_alert")
def test_no_expected_config_alerts(mock_alert, tmp_path):
    """When expected_live_model is not set, alert fires as before."""
    models, cfg = _make_predict_env(tmp_path, window_id=11,
                                    expected=None, window_exists=False)

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_called_once()
    assert "Action: train w11.pt" in mock_alert.call_args[0][0]


@patch("ashare_lab.research.predict._send_alert")
def test_deliberate_mismatch_no_alert_copy(mock_alert, tmp_path, caplog):
    """(b-copy) Same as (b) but latest.pt is a regular-file copy, not a symlink.

    This is the gpu-win production layout where latest.pt is a copy of w10.pt.
    """
    models, cfg = _make_predict_env(tmp_path, window_id=11,
                                    expected="w10", window_exists=False,
                                    use_copy=True)

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    caplog.set_level(logging.INFO, logger="ashare_lab.research.predict")
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_not_called()
    assert "not deployed (expected_live_model=w10)" in caplog.text


@patch("ashare_lab.research.predict._send_alert")
def test_copy_layout_differing_content_alerts(mock_alert, tmp_path):
    """(b-copy-diff) latest.pt is a regular-file copy of a DIFFERENT model.

    Layout: w10.pt (content A), latest.pt = copy of w9.pt (content B),
    expected_live_model="w10", window_id=11, w11.pt absent.
    filecmp finds byte-content differs -> alert fires.
    """
    models = tmp_path / "models"
    models.mkdir()

    # w10.pt -- the expected live model (content A)
    (models / "w10.pt").write_text("content A: real w10 model data")

    # w9.pt -- a different model (content B), used as the source for latest.pt
    (models / "w9.pt").write_text("content B: older w9 model data")

    # latest.pt is a regular-file copy of w9.pt (content B)
    shutil.copy2(models / "w9.pt", models / "latest.pt")

    cfg = {"research": {"expected_live_model": "w10"}}

    from ashare_lab.research.predict import _resolve_model_path

    windows = [{"window_id": 11, "test_start": "2025-07-01",
                "test_end": "2025-12-31", "train_start": "2022-07-01",
                "train_end": "2025-06-30"}]
    model_path, win = _resolve_model_path("2025-09-15", None, windows, cfg, models)

    mock_alert.assert_called_once()
    assert "w11.pt missing" in mock_alert.call_args[0][0]
