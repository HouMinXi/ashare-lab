"""Unit and integration tests for Huber M-estimator robust neutralization.

Tests cover:
1. G-Huber-1 (Outlier Immunity): 50-sigma outlier contamination; Huber beta error < 0.05 vs OLS failure.
2. G-Huber-2 (Normal Parity): Clean Gaussian data; Spearman rank correlation > 0.999 between Huber and OLS residuals.
3. Fallback on Pathological Input: Constant / collinear input; graceful degradation to OLS without crashing.
4. neutralize_predictions integration: MultiIndex DataFrame/Series end-to-end output schema, NaN handling, < _MIN_INSTRUMENTS passthrough.
5. Diagnostic logging: Verification that log messages are emitted when >10% instruments are downweighted.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

# Stub qlib.data module so deferred imports resolve without a real qlib installation.
if "qlib" not in sys.modules:
    _qlib = types.ModuleType("qlib")
    _qlib_data = types.ModuleType("qlib.data")
    _qlib_data.D = MagicMock()
    _qlib.data = _qlib_data
    sys.modules["qlib"] = _qlib
    sys.modules["qlib.data"] = _qlib_data

from ashare_lab.research.neutralize import (
    huber_m_estimator,
    neutralize_predictions,
)


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


# ---------------------------------------------------------------------------
# Unit Tests for huber_m_estimator
# ---------------------------------------------------------------------------


class TestHuberMEstimator:
    def test_g_huber_1_outlier_immunity(self):
        """Test G-Huber-1: 50-sigma outlier contamination.

        Verify beta_huber estimation error < 0.05 while OLS estimation error
        is significantly degraded (> 0.20).
        """
        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 3

        # Ground truth beta: [intercept=1.0, beta1=0.5, beta2=-0.8, beta3=1.2]
        true_beta = np.array([1.0, 0.5, -0.8, 1.2], dtype=np.float64)

        X_raw = rng.standard_normal((n_samples, n_features))
        X = np.column_stack([np.ones(n_samples), X_raw])

        # Clean noise ~ N(0, 0.1^2)
        noise = rng.normal(0.0, 0.1, size=n_samples)
        y = X @ true_beta + noise

        # Inject 5% 50-sigma outliers (10 points out of 200)
        outlier_idx = rng.choice(n_samples, size=10, replace=False)
        y[outlier_idx] += 50.0 * 0.1  # 50-sigma positive shock

        # OLS estimation
        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ols_error = np.max(np.abs(beta_ols - true_beta))

        # Huber estimation
        beta_huber, residuals_huber, weights_huber = huber_m_estimator(X, y, c=1.345)
        huber_error = np.max(np.abs(beta_huber - true_beta))

        assert huber_error < 0.05, f"Huber beta error too large: {huber_error:.4f}"
        assert ols_error > 0.20, f"OLS was expected to fail, got error: {ols_error:.4f}"
        # Outlier weights should be significantly downweighted
        assert np.all(weights_huber[outlier_idx] < 0.5), "Outliers were not downweighted"

    def test_g_huber_2_normal_parity(self):
        """Test G-Huber-2: Clean Gaussian data.

        Verify Spearman rank correlation between Huber residuals and OLS
        residuals is > 0.999 on uncorrupted normal data.
        """
        rng = np.random.default_rng(123)
        n_samples = 500
        n_features = 3

        true_beta = np.array([0.0, 1.5, -2.0, 0.7], dtype=np.float64)
        X_raw = rng.standard_normal((n_samples, n_features))
        X = np.column_stack([np.ones(n_samples), X_raw])

        noise = rng.normal(0.0, 1.0, size=n_samples)
        y = X @ true_beta + noise

        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals_ols = y - X @ beta_ols

        beta_huber, residuals_huber, weights_huber = huber_m_estimator(X, y, c=1.345)

        rho, _ = spearmanr(residuals_huber, residuals_ols)
        assert rho > 0.999, f"Spearman correlation with OLS residuals {rho:.5f} < 0.999"
        # Theoretical P(|Z| < 1.345/0.95) for N(0,1) is ~84.3%
        assert np.mean(weights_huber > 0.95) > 0.80

    def test_zero_variance_perfect_fit(self):
        """When y = X @ beta exactly (zero residuals), sigma < 1e-8 triggers early break."""
        X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]], dtype=np.float64)
        beta_true = np.array([2.0, 3.0], dtype=np.float64)
        y = X @ beta_true

        beta, residuals, weights = huber_m_estimator(X, y)
        np.testing.assert_allclose(beta, beta_true, atol=1e-6)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-6)
        np.testing.assert_allclose(weights, 1.0, atol=1e-6)

    def test_fallback_on_max_iter_non_convergence(self):
        """If max_iter is 0 (or fails to converge), fallback to OLS."""
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
        y = rng.standard_normal(50)

        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta, residuals, weights = huber_m_estimator(X, y, max_iter=0)

        np.testing.assert_allclose(beta, beta_ols)
        np.testing.assert_allclose(residuals, y - X @ beta_ols)
        np.testing.assert_allclose(weights, np.ones_like(y))

    def test_fallback_on_pathological_collinear_input(self):
        """Singular or rank-deficient matrix should execute robustly without unhandled exception."""
        n_samples = 40
        X = np.ones((n_samples, 3))  # Perfectly collinear columns
        y = np.ones(n_samples)

        beta, residuals, weights = huber_m_estimator(X, y)
        assert beta.shape == (3,)
        assert residuals.shape == (n_samples,)
        assert weights.shape == (n_samples,)


# ---------------------------------------------------------------------------
# Integration Tests for neutralize_predictions
# ---------------------------------------------------------------------------


class TestNeutralizePredictionsHuberIntegration:
    @pytest.fixture()
    def synthetic_dataset(self):
        rng = np.random.default_rng(42)
        date = "2024-01-15"
        n_stocks = 60
        instruments = [f"SH60{i:04d}" for i in range(n_stocks)]

        size_signal = rng.standard_normal(n_stocks)
        vol_signal = rng.standard_normal(n_stocks)
        mom_signal = rng.standard_normal(n_stocks)

        alpha = rng.standard_normal(n_stocks)
        pred_values = 0.5 * size_signal - 0.3 * vol_signal + 0.2 * mom_signal + alpha

        dates_flat = [date] * n_stocks
        pred = _make_multiindex_series(dates_flat, instruments, pred_values)

        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(date), inst) for inst in instruments],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "size": size_signal,
                "volatility": vol_signal,
                "momentum": mom_signal,
            },
            index=idx,
        )
        return pred, factors, instruments, date

    def test_end_to_end_schema_and_types(self, synthetic_dataset):
        """neutralize_predictions returns Series with exact MultiIndex and float dtype."""
        pred, factors, instruments, _ = synthetic_dataset
        result = neutralize_predictions(pred, factors)

        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, pred.index)
        assert result.dtype == np.float64 or result.dtype == float
        assert not result.isna().any()

    def test_outlier_downweighting_diagnostic_logging(self, synthetic_dataset, caplog):
        """When >10% of instruments are heavy outliers, diagnostic log message is triggered."""
        pred, factors, instruments, date = synthetic_dataset

        # Inject extreme outliers into 15% of stocks (9 stocks)
        pred_corrupted = pred.copy()
        outlier_instruments = instruments[:9]
        for inst in outlier_instruments:
            pred_corrupted.loc[(pd.Timestamp(date), inst)] += 100.0

        with caplog.at_level(logging.INFO):
            result = neutralize_predictions(pred_corrupted, factors)

        assert isinstance(result, pd.Series)
        # Check that warning/info message regarding downweighted instruments was logged
        log_records = [r.message for r in caplog.records]
        assert any("downweighted" in msg.lower() or "huber" in msg.lower() for msg in log_records), (
            f"Expected diagnostic log for downweighted outliers, got: {log_records}"
        )

    def test_passthrough_when_under_min_instruments(self):
        """When fewer than _MIN_INSTRUMENTS (30) exist, predictions pass through unchanged."""
        date = "2024-01-15"
        n_stocks = 25
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

        result = neutralize_predictions(pred, factors)
        pd.testing.assert_series_equal(result, pred)
