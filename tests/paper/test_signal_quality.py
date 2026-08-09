"""Tests for 10-01: PSI distribution-drift monitor.

Contract items 1-12 + 11b backfill guard.
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
    _MIN_REFERENCE_FILES,
    _N_BUCKETS,
    _PSI_THRESHOLD,
    _REFERENCE_DAYS,
    compute_psi,
    load_reference_scores,
    signal_quality_psi_step,
)


# ---------------------------------------------------------------------------
# 1. compute_psi: identical distributions -> ~0.0
# ---------------------------------------------------------------------------
def test_compute_psi_identical():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 10000)
    psi = compute_psi(data, data)
    assert psi is not None
    assert psi <= 0.01


# ---------------------------------------------------------------------------
# 2. compute_psi: known shift -> positive PSI
# ---------------------------------------------------------------------------
def test_compute_psi_known_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 10000)
    cur = rng.normal(0.5, 1, 10000)
    psi = compute_psi(ref, cur)
    assert psi is not None
    assert psi > 0.05


# ---------------------------------------------------------------------------
# 3. compute_psi: degenerate reference -> None
# ---------------------------------------------------------------------------
def test_compute_psi_degenerate():
    ref = np.full(100, 5.0)
    cur = np.array([1.0, 2.0, 3.0])
    psi = compute_psi(ref, cur)
    assert psi is None


# ---------------------------------------------------------------------------
# 4. compute_psi: empty arrays -> None
# ---------------------------------------------------------------------------
def test_compute_psi_empty_ref():
    psi = compute_psi(np.array([]), np.array([1.0, 2.0]))
    assert psi is None


def test_compute_psi_empty_cur():
    psi = compute_psi(np.array([1.0, 2.0]), np.array([]))
    assert psi is None


# ---------------------------------------------------------------------------
# 5. load_reference_scores: sufficient files -> (array, count)
# ---------------------------------------------------------------------------
def test_load_reference_scores_sufficient(tmp_path):
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        df = pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)})
        df.to_parquet(tmp_path / f"{date}.parquet")
    scores, count = load_reference_scores(tmp_path, "2026-08-01", n_days=60)
    assert scores is not None
    assert count == 25
    assert len(scores) == 25 * 100


# ---------------------------------------------------------------------------
# 6. load_reference_scores: insufficient files -> (None, count)
# ---------------------------------------------------------------------------
def test_load_reference_scores_insufficient(tmp_path):
    for i in range(15):
        date = f"2026-07-{i+1:02d}"
        df = pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)})
        df.to_parquet(tmp_path / f"{date}.parquet")
    scores, count = load_reference_scores(tmp_path, "2026-08-01", n_days=60)
    assert scores is None
    assert count == 15


# ---------------------------------------------------------------------------
# 7. load_reference_scores: filters by date + corrupt skipped
# ---------------------------------------------------------------------------
def test_load_reference_scores_date_filter(tmp_path):
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        df = pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)})
        df.to_parquet(tmp_path / f"{date}.parquet")
    scores, count = load_reference_scores(tmp_path, "2026-07-20", n_days=60)
    assert scores is None  # 19 files < _MIN_REFERENCE_FILES (20)
    assert count == 19  # 07-01 to 07-19


# ---------------------------------------------------------------------------
# 8. Integration: fresh day -> sidecar + alert
# ---------------------------------------------------------------------------
def test_psi_step_fresh_day_alert(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pd.DataFrame({"score": np.random.default_rng(0).normal(0.5, 1, 100)}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as mock_alert, \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_psi_step(ctx)
    mock_alert.assert_not_called()  # dry-run suppresses alerts
    sidecar = json.loads(pred_path.with_suffix(".meta.json").read_text())
    assert "psi" in sidecar
    assert sidecar["psi"] is not None
    assert sidecar["psi_reference_days"] == 25


# ---------------------------------------------------------------------------
# 9. Integration: uses pred_path date (not ctx.trade_date)
# ---------------------------------------------------------------------------
def test_psi_step_uses_pred_date(tmp_path):
    pred_path = tmp_path / "2026-07-15.parquet"
    pd.DataFrame({"score": np.random.default_rng(0).normal(0, 1, 100)}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-07-15"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_psi_step(ctx)
    sidecar = json.loads(pred_path.with_suffix(".meta.json").read_text())
    assert "psi" in sidecar
    assert sidecar["psi_reference_days"] == 14  # 07-01 to 07-14 (14 days before 07-15)


# ---------------------------------------------------------------------------
# 10. Integration: not drifted -> no alert
# ---------------------------------------------------------------------------
def test_psi_step_no_alert(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    # Use same distribution for reference and current -> PSI ~ 0 (no drift)
    rng = np.random.default_rng(42)
    scores = rng.normal(0, 1, 100)
    pd.DataFrame({"score": scores}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": scores}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as mock_alert:
        signal_quality_psi_step(ctx)
    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# 11. Integration: exception -> fail-open + psi_error
# ---------------------------------------------------------------------------
def test_psi_step_fail_open(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_parquet(pred_path)
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.load_reference_scores", side_effect=Exception("load error")), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_psi_step(ctx)
    sidecar = json.loads(pred_path.with_suffix(".meta.json").read_text())
    assert "psi_error" in sidecar


# ---------------------------------------------------------------------------
# 11b. Backfill guard
# ---------------------------------------------------------------------------
def test_psi_step_backfill_guard(tmp_path):
    ctx = MagicMock()
    ctx.steps = {"settle", "signal"}
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.load_reference_scores") as mock_load:
        signal_quality_psi_step(ctx)
    mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# 12. pred_path=None -> records null PSI
# ---------------------------------------------------------------------------
def test_psi_step_no_prediction(tmp_path):
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = None
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch.dict("os.environ", {"ASHARE_SQ_DRYRUN": "1"}, clear=False):
        signal_quality_psi_step(ctx)
    sidecar_path = tmp_path / "2026-08-06.meta.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["psi"] is None
    assert sidecar["psi_reason"] == "no_prediction"


# ---------------------------------------------------------------------------
# 13. Alert send failure -> fail-open (psi_step)
# ---------------------------------------------------------------------------
def test_psi_step_alert_send_failure(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pd.DataFrame({"score": np.random.default_rng(0).normal(0.5, 1, 100)}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=False) as mock_alert:
        signal_quality_psi_step(ctx)
    mock_alert.assert_called_once()
    sidecar = json.loads(pred_path.with_suffix(".meta.json").read_text())
    assert "psi" in sidecar  # PSI still recorded despite alert failure


# ---------------------------------------------------------------------------
# 14. ASHARE_USE_STALE=1 -> early return
# ---------------------------------------------------------------------------
def test_psi_step_stale_guard(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_parquet(pred_path)
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.load_reference_scores") as mock_load, \
         patch.dict("os.environ", {"ASHARE_USE_STALE": "1"}, clear=False):
        signal_quality_psi_step(ctx)
    mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# 15. Integration: compute_psi returns None (degenerate reference)
# ---------------------------------------------------------------------------
def test_psi_step_degenerate_reference(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    # All identical scores -> compute_psi returns None
    pd.DataFrame({"score": [5.0] * 100}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": [5.0] * 100}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path):
        signal_quality_psi_step(ctx)
    sidecar = json.loads((tmp_path / "2026-08-06.meta.json").read_text())
    assert sidecar["psi"] is None


# ---------------------------------------------------------------------------
# 16. Corrupt prediction parquet -> fail-open + psi_error
# ---------------------------------------------------------------------------
def test_psi_step_corrupt_parquet(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pred_path.write_bytes(b"not a parquet")
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path):
        signal_quality_psi_step(ctx)
    sidecar = json.loads((tmp_path / "2026-08-06.meta.json").read_text())
    assert "psi_error" in sidecar


# ---------------------------------------------------------------------------
# 17. Insufficient reference files -> no alert (load_reference_scores returns None)
# ---------------------------------------------------------------------------
def test_psi_step_insufficient_reference(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    pd.DataFrame({"score": np.random.default_rng(0).normal(0, 1, 100)}).to_parquet(pred_path)
    # Only 10 files < 20 minimum
    for i in range(10):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert") as mock_alert:
        signal_quality_psi_step(ctx)
    mock_alert.assert_not_called()
    sidecar = json.loads((tmp_path / "2026-08-06.meta.json").read_text())
    assert sidecar.get("psi") is None


# ---------------------------------------------------------------------------
# 18. Missing pred_path file -> fail-open + psi_error
# ---------------------------------------------------------------------------
def test_psi_step_missing_file(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    # File does not exist, but reference files exist -> read_parquet fails
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path):
        signal_quality_psi_step(ctx)
    sidecar = json.loads((tmp_path / "2026-08-06.meta.json").read_text())
    assert "psi_error" in sidecar


# ---------------------------------------------------------------------------
# 19. PSI alert triggered (not dry-run) when PSI > threshold
# ---------------------------------------------------------------------------
def test_psi_step_alert_triggered(tmp_path):
    pred_path = tmp_path / "2026-08-06.parquet"
    # Deterministic: current scores all at 10.0, reference at ~0 -> PSI >> threshold
    pd.DataFrame({"score": [10.0] * 100}).to_parquet(pred_path)
    for i in range(25):
        date = f"2026-07-{i+1:02d}"
        pd.DataFrame({"score": np.random.default_rng(i).normal(0, 1, 100)}).to_parquet(tmp_path / f"{date}.parquet")
    ctx = MagicMock()
    ctx.steps = None
    ctx.pred_path = pred_path
    ctx.predictions_date_str = "2026-08-06"
    ctx.trade_date = "2026-08-06"
    with patch("ashare_lab.paper.signal_quality.PREDICTIONS_DIR", tmp_path), \
         patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as mock_alert:
        signal_quality_psi_step(ctx)
    mock_alert.assert_called_once()
    sidecar = json.loads(pred_path.with_suffix(".meta.json").read_text())
    assert sidecar["psi"] is not None
    assert sidecar["psi"] > 0.25


# ---------------------------------------------------------------------------
# 20. PSI floor: sparse current (concentrated in few bins) -> finite PSI
# ---------------------------------------------------------------------------
def test_compute_psi_sparse_current():
    """Floor (1e-4) prevents division by zero when current is concentrated."""
    ref = np.random.default_rng(42).normal(0, 1, 10000)
    # All current scores in one narrow band
    cur = np.full(1000, 0.0)
    psi = compute_psi(ref, cur)
    assert psi is not None
    assert np.isfinite(psi)


# ---------------------------------------------------------------------------
# 21. compute_psi with NaN/Inf -> filters and returns finite PSI
# ---------------------------------------------------------------------------
def test_compute_psi_nan_inf():
    """NaN and Inf values are filtered before PSI computation."""
    ref = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf, 4.0, 5.0])
    cur = np.array([1.1, 2.1, 3.1, np.nan, np.inf, 4.1, 5.1])
    psi = compute_psi(ref, cur)
    assert psi is not None
    assert np.isfinite(psi)


# ---------------------------------------------------------------------------
# 22. compute_psi: current outside reference range (clipping)
# ---------------------------------------------------------------------------
def test_compute_psi_out_of_range():
    """Current scores outside reference range are clipped."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
    # Current scores extend beyond reference range
    cur = np.array([-10.0, 0.0, 3.0, 6.0, 20.0] * 100)
    psi = compute_psi(ref, cur)
    assert psi is not None
    assert np.isfinite(psi)
    assert psi > 0  # Should show drift due to clipping
