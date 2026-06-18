"""Unit tests for ashare_lab.research.regime.

All tests run without a qlib runtime. Mock qlib D.features to test
signal computation logic and filter behavior in isolation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ashare_lab.research.regime import (
    apply_regime_filter,
    calibrate_thresholds,
    compute_regime_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features_df(
    instruments: list[str],
    close_prices: list[float],
    ma20_values: list[float],
    date: str = "2023-06-15",
) -> pd.DataFrame:
    """Build a DataFrame mimicking qlib D.features output.

    qlib D.features returns a DataFrame with MultiIndex (instrument, datetime)
    and columns matching the requested fields.
    """
    idx = pd.MultiIndex.from_tuples(
        [(inst, pd.Timestamp(date)) for inst in instruments],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame(
        {"$close": close_prices, "Mean($close, 20)": ma20_values},
        index=idx,
    )


@pytest.fixture()
def favorable_features():
    """80% of instruments have close > MA20 (favorable regime)."""
    instruments = [f"SH60{i:04d}" for i in range(10)]
    # 8 out of 10 above MA20
    close_prices = [12.0, 15.0, 20.0, 8.0, 25.0, 30.0, 18.0, 22.0, 9.0, 35.0]
    ma20_values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    return _make_features_df(instruments, close_prices, ma20_values)


@pytest.fixture()
def unfavorable_features():
    """30% of instruments have close > MA20 (unfavorable regime)."""
    instruments = [f"SH60{i:04d}" for i in range(10)]
    # 3 out of 10 above MA20
    close_prices = [8.0, 12.0, 7.0, 6.0, 15.0, 5.0, 4.0, 3.0, 11.0, 2.0]
    ma20_values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    return _make_features_df(instruments, close_prices, ma20_values)


@pytest.fixture()
def extreme_features():
    """10% of instruments have close > MA20 (extreme / go-to-cash regime)."""
    instruments = [f"SH60{i:04d}" for i in range(10)]
    # 1 out of 10 above MA20
    close_prices = [5.0, 6.0, 7.0, 4.0, 3.0, 2.0, 1.0, 8.0, 15.0, 6.0]
    ma20_values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    return _make_features_df(instruments, close_prices, ma20_values)


@pytest.fixture()
def sample_pred():
    """A simple prediction Series with MultiIndex (datetime, instrument)."""
    instruments = [f"SH60{i:04d}" for i in range(15)]
    date = "2023-06-15"
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), inst) for inst in instruments],
        names=["datetime", "instrument"],
    )
    return pd.Series(
        np.random.default_rng(42).random(15),
        index=idx,
        dtype=float,
    )


@pytest.fixture()
def default_thresholds():
    """Standard thresholds: cash < 0.20, reduce < 0.35, else full."""
    return {"cash_threshold": 0.20, "reduce_threshold": 0.35}


# ---------------------------------------------------------------------------
# Test 1: compute_regime_signals returns dict with "ma20_above_pct" key
# ---------------------------------------------------------------------------

class TestComputeRegimeSignals:
    @patch("ashare_lab.research.regime.D")
    def test_returns_dict_with_ma20_above_pct_key(self, mock_d, favorable_features):
        """compute_regime_signals returns dict containing 'ma20_above_pct'."""
        mock_d.features.return_value = favorable_features
        result = compute_regime_signals("2023-06-15")
        assert isinstance(result, dict)
        assert "ma20_above_pct" in result
        assert "date" in result
        assert result["date"] == "2023-06-15"

    # -----------------------------------------------------------------------
    # Test 2: ma20_above_pct is between 0.0 and 1.0
    # -----------------------------------------------------------------------

    @patch("ashare_lab.research.regime.D")
    def test_ma20_above_pct_bounded_0_to_1(self, mock_d, favorable_features):
        """ma20_above_pct must be in [0.0, 1.0]."""
        mock_d.features.return_value = favorable_features
        result = compute_regime_signals("2023-06-15")
        pct = result["ma20_above_pct"]
        assert 0.0 <= pct <= 1.0, f"Expected 0..1, got {pct}"

    @patch("ashare_lab.research.regime.D")
    def test_ma20_above_pct_correct_value(self, mock_d, favorable_features):
        """80% above MA20 -> ma20_above_pct == 0.8."""
        mock_d.features.return_value = favorable_features
        result = compute_regime_signals("2023-06-15")
        assert result["ma20_above_pct"] == pytest.approx(0.8)

    # -----------------------------------------------------------------------
    # Test 7: handles missing data gracefully (returns 0.5)
    # -----------------------------------------------------------------------

    @patch("ashare_lab.research.regime.D")
    def test_missing_data_returns_neutral(self, mock_d):
        """Empty D.features result -> ma20_above_pct == 0.5 (neutral)."""
        mock_d.features.return_value = pd.DataFrame()
        result = compute_regime_signals("2023-06-15")
        assert result["ma20_above_pct"] == 0.5
        assert result["date"] == "2023-06-15"


# ---------------------------------------------------------------------------
# Test 3-5: apply_regime_filter behavior
# ---------------------------------------------------------------------------

class TestApplyRegimeFilter:
    @patch("ashare_lab.research.regime.compute_regime_signals")
    def test_favorable_returns_full_topk(
        self, mock_signals, sample_pred, default_thresholds
    ):
        """Favorable regime (ma20_above_pct=0.80) -> (pred, topk)."""
        mock_signals.return_value = {"ma20_above_pct": 0.80, "date": "2023-06-15"}
        pred_out, effective_topk = apply_regime_filter(
            sample_pred, "2023-06-15", topk=15, thresholds=default_thresholds
        )
        assert effective_topk == 15
        assert pred_out is sample_pred  # same object, unmodified

    @patch("ashare_lab.research.regime.compute_regime_signals")
    def test_unfavorable_returns_half_topk(
        self, mock_signals, sample_pred, default_thresholds
    ):
        """Unfavorable regime (ma20_above_pct=0.30) -> (pred, topk // 2)."""
        mock_signals.return_value = {"ma20_above_pct": 0.30, "date": "2023-06-15"}
        pred_out, effective_topk = apply_regime_filter(
            sample_pred, "2023-06-15", topk=15, thresholds=default_thresholds
        )
        assert effective_topk == 15 // 2
        assert pred_out is sample_pred

    @patch("ashare_lab.research.regime.compute_regime_signals")
    def test_extreme_returns_zero_topk(
        self, mock_signals, sample_pred, default_thresholds
    ):
        """Extreme regime (ma20_above_pct=0.10) -> (pred, 0) -- go to cash."""
        mock_signals.return_value = {"ma20_above_pct": 0.10, "date": "2023-06-15"}
        pred_out, effective_topk = apply_regime_filter(
            sample_pred, "2023-06-15", topk=15, thresholds=default_thresholds
        )
        assert effective_topk == 0
        assert pred_out is sample_pred


# ---------------------------------------------------------------------------
# Test 6: calibrate_thresholds uses only pre-2023 data
# ---------------------------------------------------------------------------

class TestCalibrateThresholds:
    @patch("ashare_lab.research.regime.D")
    def test_calibration_end_date_pre_2023(self, mock_d):
        """calibrate_thresholds default end_date is 2022-12-31."""
        # Build a time series of daily features for 2015-2022.
        # Simulate: generate dates and varying ma20_above_pct values.
        dates = pd.bdate_range("2015-01-05", "2022-12-30", freq="B")
        instruments = ["SH600000"]
        rows = []
        rng = np.random.default_rng(42)
        for dt in dates:
            # Simulate close and ma20 so that ma20_above_pct varies.
            close = 10.0 + rng.normal(0, 2)
            ma20 = 10.0
            rows.append((instruments[0], dt, close, ma20))

        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in rows],
            names=["instrument", "datetime"],
        )
        features_df = pd.DataFrame(
            {"$close": [r[2] for r in rows], "Mean($close, 20)": [r[3] for r in rows]},
            index=idx,
        )
        mock_d.features.return_value = features_df

        result = calibrate_thresholds()
        assert result["calibration_end"] == "2022-12-31"
        assert "reduce_threshold" in result
        assert "cash_threshold" in result
        # cash_threshold < reduce_threshold (10th pctile < 25th pctile)
        assert result["cash_threshold"] <= result["reduce_threshold"]


# ---------------------------------------------------------------------------
# Boundary: threshold edge cases
# ---------------------------------------------------------------------------

class TestRegimeFilterEdgeCases:
    @patch("ashare_lab.research.regime.compute_regime_signals")
    def test_at_reduce_boundary(self, mock_signals, sample_pred, default_thresholds):
        """Exactly at reduce_threshold -> full exposure (>= means favorable)."""
        mock_signals.return_value = {"ma20_above_pct": 0.35, "date": "2023-06-15"}
        _, effective_topk = apply_regime_filter(
            sample_pred, "2023-06-15", topk=15, thresholds=default_thresholds
        )
        assert effective_topk == 15

    @patch("ashare_lab.research.regime.compute_regime_signals")
    def test_at_cash_boundary(self, mock_signals, sample_pred, default_thresholds):
        """Exactly at cash_threshold -> halved, not cash (>= reduce range)."""
        mock_signals.return_value = {"ma20_above_pct": 0.20, "date": "2023-06-15"}
        _, effective_topk = apply_regime_filter(
            sample_pred, "2023-06-15", topk=15, thresholds=default_thresholds
        )
        assert effective_topk == 15 // 2
