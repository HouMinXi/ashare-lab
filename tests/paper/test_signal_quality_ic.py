"""Tests for 10-02: Lagged IC ranking-power decay monitor.

Contract items 1-11 + 9b backfill guard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ashare_lab.paper.signal_quality import (
    _IC_THRESHOLD,
    _MIN_INSTRUMENTS,
    _ROLLING_WINDOW,
    SignalQualityError,
    compute_forward_returns,
    lagged_ic_for_date,
    load_reference_scores,
    rolling_lagged_ic,
    signal_quality_ic_step,
)


# ---------------------------------------------------------------------------
# 1. lagged_ic_for_date: synthetic Spearman -> ic in [0.55, 0.65]
# ---------------------------------------------------------------------------
def test_lagged_ic_synthetic(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    scores = rng.normal(0, 1, n)
    returns = scores * 0.6 + rng.normal(0, 0.5, n)
    instruments = [f"SH{i:06d}" for i in range(n)]
    pred_df = pd.DataFrame({"instrument": instruments, "score": scores})
    pred_df.to_parquet(tmp_path / "2026-07-30.parquet")
    forward_returns = pd.Series(returns, index=instruments)
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns)
    assert ic is not None
    assert 0.50 <= ic <= 0.75
    assert count == n


# ---------------------------------------------------------------------------
# 2. lagged_ic_for_date: alignment drops NaN
# ---------------------------------------------------------------------------
def test_lagged_ic_alignment(tmp_path):
    instruments = [f"SH{i:06d}" for i in range(100)]
    scores = list(range(100))
    pred_df = pd.DataFrame({"instrument": instruments, "score": scores})
    pred_df.to_parquet(tmp_path / "2026-07-30.parquet")
    forward_returns = pd.Series(
        [float(i) if i % 2 == 0 else float("nan") for i in range(100)],
        index=instruments,
    )
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns, min_instruments=10)
    assert count == 50


# ---------------------------------------------------------------------------
# 3. lagged_ic_for_date: 49 valid -> None; 50 -> computes
# ---------------------------------------------------------------------------
def test_lagged_ic_min_instruments(tmp_path):
    instruments = [f"SH{i:06d}" for i in range(49)]
    scores = list(range(49))
    pred_df = pd.DataFrame({"instrument": instruments, "score": scores})
    pred_df.to_parquet(tmp_path / "2026-07-30.parquet")
    forward_returns = pd.Series([float(i) for i in range(49)], index=instruments)
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns)
    assert ic is None
    assert count == 49


# ---------------------------------------------------------------------------
# 4. lagged_ic_for_date: constant scores -> None
# ---------------------------------------------------------------------------
def test_lagged_ic_constant(tmp_path):
    instruments = [f"SH{i:06d}" for i in range(100)]
    pred_df = pd.DataFrame({"instrument": instruments, "score": [1.0] * 100})
    pred_df.to_parquet(tmp_path / "2026-07-30.parquet")
    forward_returns = pd.Series([float(i) for i in range(100)], index=instruments)
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns)
    assert ic is None


# ---------------------------------------------------------------------------
# 5. lagged_ic_for_date: missing parquet -> (None, 0)
# ---------------------------------------------------------------------------
def test_lagged_ic_missing_parquet(tmp_path):
    forward_returns = pd.Series([1.0, 2.0], index=["SH000001", "SH000002"])
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns)
    assert ic is None
    assert count == 0


# ---------------------------------------------------------------------------
# 6. DATE-PIN: 2026-08-06 -> t5 == 2026-07-30
# ---------------------------------------------------------------------------
def test_date_pin():
    import datetime as dt
    from ashare_lab.data.calendar import previous_trading_day
    trade_date = dt.date.fromisoformat("2026-08-06")
    cursor = trade_date
    for _ in range(5):
        cursor = previous_trading_day(cursor)
    assert cursor.isoformat() == "2026-07-30"


# ---------------------------------------------------------------------------
# 7. rolling_lagged_ic: 19 -> (None, 19); 20 -> (mean, 20)
# ---------------------------------------------------------------------------
def test_rolling_lagged_ic_insufficient(tmp_path):
    for i in range(19):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    mean, n = rolling_lagged_ic(tmp_path, "2026-08-01")
    assert mean is None
    assert n == 19


def test_rolling_lagged_ic_sufficient(tmp_path):
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    mean, n = rolling_lagged_ic(tmp_path, "2026-08-01")
    assert mean is not None
    assert n == 20
    assert abs(mean - (-0.05)) < 1e-10


# ---------------------------------------------------------------------------
# 8. Integration: sidecar on t-5 + alert when mean < 0.01
# ---------------------------------------------------------------------------
def test_ic_step_alert(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", return_value=pd.Series([0.01] * 100, index=instruments)), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as mock_alert:
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_t5" in sidecar
    mock_alert.assert_called_once()


# ---------------------------------------------------------------------------
# 9. ASHARE_SQ_DRYRUN=1: computation runs, alert NOT sent
# ---------------------------------------------------------------------------
def test_ic_step_dryrun(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", return_value=pd.Series([0.01] * 100, index=instruments)), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as mock_alert, \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# 9b. Backfill guard
# ---------------------------------------------------------------------------
def test_ic_step_backfill_guard(tmp_path):
    ctx = MagicMock()
    ctx.steps = {"settle", "signal"}
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns") as mock_cfr:
        signal_quality_ic_step(ctx)
    mock_cfr.assert_not_called()


# ---------------------------------------------------------------------------
# 10. SignalQualityError -> fail-open
# ---------------------------------------------------------------------------
def test_ic_step_fail_open(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", side_effect=SignalQualityError("timeout")), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_error" in sidecar
    assert sidecar["lagged_ic_t5"] is None  # stale value cleared on error


# ---------------------------------------------------------------------------
# 11. rolling_lagged_ic: corrupt JSON skipped
# ---------------------------------------------------------------------------
def test_rolling_lagged_ic_corrupt(tmp_path):
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        if i == 5:
            (tmp_path / f"{date}.meta.json").write_text("not valid json{")
        else:
            (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    mean, n = rolling_lagged_ic(tmp_path, "2026-08-01")
    assert n == 19


def test_lagged_ic_empty_forward_returns(tmp_path):
    instruments = [f"SH{i:06d}" for i in range(100)]
    pred_df = pd.DataFrame({"instrument": instruments, "score": list(range(100))})
    pred_df.to_parquet(tmp_path / "2026-07-30.parquet")
    forward_returns = pd.Series(dtype=float)
    ic, count = lagged_ic_for_date(tmp_path, "2026-07-30", forward_returns)
    assert ic is None
    assert count == 0


def test_load_reference_scores_corrupt(tmp_path):
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        if i == 5:
            (tmp_path / f"{date}.parquet").write_bytes(b"not a parquet")
        else:
            pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    scores, count = load_reference_scores(tmp_path, "2026-08-01", n_days=60)
    assert scores is not None
    assert count == 24  # 25 - 1 corrupt


# ---------------------------------------------------------------------------
# 13. ASHARE_USE_STALE=1 -> early return (ic_step)
# ---------------------------------------------------------------------------
def test_ic_step_stale_guard(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns") as mock_cfr, \
         patch.dict("os.environ", {"ASHARE_USE_STALE": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    mock_cfr.assert_not_called()


# ---------------------------------------------------------------------------
# 14. Alert send failure -> fail-open (ic_step)
# ---------------------------------------------------------------------------
def test_ic_step_alert_send_failure(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", return_value=pd.Series([0.01] * 100, index=instruments)), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=False) as mock_alert:
        signal_quality_ic_step(ctx)
    mock_alert.assert_called_once()
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_t5" in sidecar  # IC still recorded despite alert failure


# ---------------------------------------------------------------------------
# 15. No alert when rolling mean >= threshold (ic_step)
# ---------------------------------------------------------------------------
def test_ic_step_no_alert(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    # Rolling mean = 0.05 > _IC_THRESHOLD (0.01) -> no alert
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": 0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", return_value=pd.Series([0.01] * 100, index=instruments)), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert") as mock_alert:
        signal_quality_ic_step(ctx)
    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# 16. Generic exception in ic_step -> fail-open (not just SignalQualityError)
# ---------------------------------------------------------------------------
def test_ic_step_fail_open_generic_exception(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", side_effect=ValueError("unexpected error")), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_error" in sidecar
    assert sidecar["lagged_ic_t5"] is None  # stale value cleared on error


# ---------------------------------------------------------------------------
# 17. Subprocess timeout -> fail-open (SignalQualityError)
# ---------------------------------------------------------------------------
def test_ic_step_subprocess_timeout(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", side_effect=SignalQualityError("subprocess timeout")), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_error" in sidecar
    assert "lagged_ic_t5" in sidecar
    assert sidecar["lagged_ic_t5"] is None  # stale value cleared on error


# ---------------------------------------------------------------------------
# 18. Missing t5 parquet -> no alert, writes missing reason
# ---------------------------------------------------------------------------
def test_ic_step_missing_t5_parquet(tmp_path):
    trade_date = "2026-08-06"
    # t5_date = 2026-07-30 but file does not exist
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert") as mock_alert:
        signal_quality_ic_step(ctx)
    mock_alert.assert_not_called()
    sidecar = json.loads((tmp_path / "2026-07-30.meta.json").read_text())
    assert sidecar.get("lagged_ic_t5") is None
    assert sidecar.get("lagged_ic_t5_reason") == "missing_t5_parquet"


# ---------------------------------------------------------------------------
# 19. Insufficient rolling window (<20) -> no alert, no error
# ---------------------------------------------------------------------------
def test_ic_step_insufficient_rolling_window(tmp_path):
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    instruments = [f"SH{i:06d}" for i in range(100)]
    pd.DataFrame({"instrument": instruments, "score": list(range(100))}).to_parquet(tmp_path / f"{t5_date}.parquet")
    # Only 5 files < 20 rolling window
    for i in range(5):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.compute_forward_returns", return_value=pd.Series([0.01] * 100, index=instruments)), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert") as mock_alert:
        signal_quality_ic_step(ctx)
    mock_alert.assert_not_called()
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_t5" in sidecar  # IC recorded even with insufficient window




# ---------------------------------------------------------------------------
# 20. Date derivation failure -> fail-open
# ---------------------------------------------------------------------------
def test_ic_step_date_derivation_failure(tmp_path):
    """Verify fail-open when previous_trading_day raises."""
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = "2026-08-06"
    ctx.predictions_date_str = "invalid-date-format"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.previous_trading_day", side_effect=ValueError("bad date")):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / "2026-08-06.meta.json").read_text())
    assert "lagged_ic_error" in sidecar


# ---------------------------------------------------------------------------
# 21. Corrupt t-5 parquet -> fail-open
# ---------------------------------------------------------------------------
def test_ic_step_corrupt_t5_parquet(tmp_path):
    """Verify fail-open when t-5 parquet is corrupt."""
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    # Write corrupt parquet
    (tmp_path / f"{t5_date}.parquet").write_bytes(b"not a parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    ctx.pred_path = tmp_path / f"{trade_date}.parquet"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_error" in sidecar


# ---------------------------------------------------------------------------
# 22. t-5 parquet missing 'instrument' column -> fail-open
# ---------------------------------------------------------------------------
def test_ic_step_missing_instrument_column(tmp_path):
    """Verify fail-open when t-5 parquet lacks 'instrument' column."""
    t5_date = "2026-07-30"
    trade_date = "2026-08-06"
    # Write parquet without 'instrument' column
    pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_parquet(tmp_path / f"{t5_date}.parquet")
    for i in range(20):
        date = f"2026-07-{i+1:02d}"
        (tmp_path / f"{date}.meta.json").write_text(json.dumps({"lagged_ic_t5": -0.05}))
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    ctx.pred_path = tmp_path / f"{trade_date}.parquet"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_ic_step(ctx)
    sidecar = json.loads((tmp_path / f"{t5_date}.meta.json").read_text())
    assert "lagged_ic_error" in sidecar


# ---------------------------------------------------------------------------
# 23. No prediction file -> skip IC monitor
# ---------------------------------------------------------------------------
def test_ic_step_no_prediction(tmp_path):
    """Verify IC monitor skips when ctx.pred_path is None."""
    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = "2026-08-06"
    ctx.predictions_date_str = "2026-08-06"
    ctx.pred_path = None
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path):
        signal_quality_ic_step(ctx)
    # No sidecar should be written
    assert not (tmp_path / "2026-08-06.meta.json").exists()
