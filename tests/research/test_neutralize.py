"""Unit tests for ashare_lab.research.neutralize.

All tests run without a qlib runtime. Qlib D.features is mocked in every
test to provide synthetic factor data. Covers compute_style_factors() and
neutralize_predictions() with 5 behaviour tests.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Stub qlib.data module so deferred `from qlib.data import D` resolves
# without a real qlib installation. The mock is applied per-test via
# patch("qlib.data.D").
if "qlib" not in sys.modules:
    _qlib = types.ModuleType("qlib")
    _qlib_data = types.ModuleType("qlib.data")
    _qlib_data.D = MagicMock()
    _qlib.data = _qlib_data
    sys.modules["qlib"] = _qlib
    sys.modules["qlib.data"] = _qlib_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multiindex_series(dates, instruments, values):
    """Build a MultiIndex (datetime, instrument) Series from parallel lists."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), sym) for d, sym in zip(dates, instruments)],
        names=["datetime", "instrument"],
    )
    return pd.Series(values, index=idx, dtype=float)


def _make_factor_features_df(dates, instruments, close, volume):
    """Build a DataFrame mimicking qlib D.features output for factor expressions.

    Returns DataFrame with MultiIndex (datetime, instrument) and columns
    matching the qlib expressions used by compute_style_factors.
    """
    rows = []
    for d, inst, c, v in zip(dates, instruments, close, volume):
        rows.append((pd.Timestamp(d), inst, c, v))

    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "$close": [r[2] for r in rows],
            "$volume": [r[3] for r in rows],
            "Std($close, 20)/Mean($close, 20)": [0.05 + i * 0.01 for i in range(len(rows))],
            "Ref($close, -20)/$close - 1": [0.02 + i * 0.005 for i in range(len(rows))],
        },
        index=idx,
    )


@pytest.fixture()
def synthetic_50_stocks():
    """Generate synthetic data for 50 stocks on 1 date.

    Returns (pred, mock_features_df) where:
      - pred is a MultiIndex Series (datetime, instrument) with random scores
      - mock_features_df is a DataFrame mimicking D.features output
    The size factor (log close * volume proxy) is deliberately correlated with
    pred so neutralization should remove that correlation.
    """
    rng = np.random.default_rng(42)
    date = "2024-01-15"
    n_stocks = 50
    instruments = [f"SH60{i:04d}" for i in range(n_stocks)]

    # Generate factor-like data
    size_signal = rng.standard_normal(n_stocks)
    vol_signal = rng.standard_normal(n_stocks)
    mom_signal = rng.standard_normal(n_stocks)

    # Predictions deliberately correlated with size: pred = 0.6*size + 0.4*alpha
    alpha = rng.standard_normal(n_stocks)
    pred_values = 0.6 * size_signal + 0.4 * alpha

    dates_flat = [date] * n_stocks
    pred = _make_multiindex_series(dates_flat, instruments, pred_values)

    # Build mock D.features output
    # close and volume chosen so log(close * volume) ~ size_signal after z-score
    close_vals = np.exp(size_signal + 5)  # centred around e^5 ~ 148
    volume_vals = np.full(n_stocks, 1e6)  # constant volume so size ~ log(close)

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), inst) for inst in instruments],
        names=["datetime", "instrument"],
    )
    mock_df = pd.DataFrame(
        {
            "$close": close_vals,
            "$volume": volume_vals,
            "Std($close, 20)/Mean($close, 20)": np.abs(vol_signal) * 0.05 + 0.01,
            "Ref($close, -20)/$close - 1": mom_signal * 0.05,
        },
        index=idx,
    )

    return pred, mock_df, instruments


# ---------------------------------------------------------------------------
# Tests for compute_style_factors
# ---------------------------------------------------------------------------

class TestComputeStyleFactors:
    def test_returns_three_factor_columns(self, synthetic_50_stocks):
        """compute_style_factors returns DataFrame with Size, Volatility,
        Momentum columns indexed by (datetime, instrument)."""
        _pred, mock_df, instruments = synthetic_50_stocks

        with patch("qlib.data.D") as mock_D:
            mock_D.features.return_value = mock_df
            from ashare_lab.research.neutralize import compute_style_factors

            result = compute_style_factors(
                instruments=instruments,
                start_time="2024-01-15",
                end_time="2024-01-15",
            )

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"size", "volatility", "momentum"}
        assert result.index.names == ["datetime", "instrument"]
        assert len(result) == len(mock_df)


# ---------------------------------------------------------------------------
# Tests for neutralize_predictions
# ---------------------------------------------------------------------------

class TestNeutralizePredictions:
    def test_returns_series_same_index(self, synthetic_50_stocks):
        """neutralize_predictions returns a Series with the same index as
        the input pred (after dropping NaN-factor instruments)."""
        pred, mock_df, instruments = synthetic_50_stocks

        with patch("qlib.data.D") as mock_D:
            mock_D.features.return_value = mock_df
            from ashare_lab.research.neutralize import (
                compute_style_factors,
                neutralize_predictions,
            )

            factors = compute_style_factors(
                instruments=instruments,
                start_time="2024-01-15",
                end_time="2024-01-15",
            )
            result = neutralize_predictions(pred, factors)

        assert isinstance(result, pd.Series)
        # All 50 stocks have valid factors, so index should match exactly
        pd.testing.assert_index_equal(result.index, pred.index)

    def test_neutralized_near_zero_size_correlation(self, synthetic_50_stocks):
        """After neutralization, correlation with size factor should be
        near zero (within tolerance 0.05)."""
        pred, mock_df, instruments = synthetic_50_stocks

        with patch("qlib.data.D") as mock_D:
            mock_D.features.return_value = mock_df
            from ashare_lab.research.neutralize import (
                compute_style_factors,
                neutralize_predictions,
            )

            factors = compute_style_factors(
                instruments=instruments,
                start_time="2024-01-15",
                end_time="2024-01-15",
            )
            residuals = neutralize_predictions(pred, factors)

        # Extract size factor for the single date
        date = pd.Timestamp("2024-01-15")
        size_vals = factors.xs(date, level="datetime")["size"]
        resid_vals = residuals.xs(date, level="datetime")

        # Align indices
        common = size_vals.index.intersection(resid_vals.index)
        corr = np.corrcoef(resid_vals[common].values, size_vals[common].values)[0, 1]
        assert abs(corr) < 0.05, (
            f"Neutralized predictions should have near-zero correlation "
            f"with size factor, got {corr:.4f}"
        )

    def test_returns_pred_unchanged_when_few_stocks(self):
        """When fewer than 30 stocks have valid factors, return pred unchanged."""
        date = "2024-01-15"
        n_stocks = 20  # fewer than 30
        instruments = [f"SH60{i:04d}" for i in range(n_stocks)]
        rng = np.random.default_rng(99)

        pred = _make_multiindex_series(
            [date] * n_stocks, instruments, rng.standard_normal(n_stocks)
        )

        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(date), inst) for inst in instruments],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "size": rng.standard_normal(n_stocks),
                "volatility": rng.standard_normal(n_stocks),
                "momentum": rng.standard_normal(n_stocks),
            },
            index=idx,
        )

        from ashare_lab.research.neutralize import neutralize_predictions

        result = neutralize_predictions(pred, factors)
        pd.testing.assert_series_equal(result, pred)

    def test_handles_nan_in_factors(self, synthetic_50_stocks):
        """NaN values in factors: those instruments are dropped from
        regression but result still returned for valid instruments."""
        pred, mock_df, instruments = synthetic_50_stocks

        # Inject NaN into 5 instruments' volatility factor
        mock_df_with_nan = mock_df.copy()
        nan_idx = mock_df_with_nan.index[:5]
        mock_df_with_nan.loc[nan_idx, "Std($close, 20)/Mean($close, 20)"] = np.nan

        with patch("qlib.data.D") as mock_D:
            mock_D.features.return_value = mock_df_with_nan
            from ashare_lab.research.neutralize import (
                compute_style_factors,
                neutralize_predictions,
            )

            factors = compute_style_factors(
                instruments=instruments,
                start_time="2024-01-15",
                end_time="2024-01-15",
            )
            result = neutralize_predictions(pred, factors)

        # NaN factors are z-score normalized to 0 (fillna), so all 50
        # instruments still enter regression. Result covers all instruments.
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert not result.isna().any()
