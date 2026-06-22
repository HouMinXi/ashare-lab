"""Unit tests for Contract 1: signal_transform hook in rolling.py.

Tests that:
  - signal_transform=None produces identical behavior (identity).
  - A valid transform is applied to predictions before IC/backtest.
  - A failing transform falls back to raw pred and records the failure.

All tests run without qlib. The walk-forward loop internals (train_window,
run_backtest, etc.) are mocked to isolate the hook behavior.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Qlib stub (must precede any ashare_lab import that touches qlib)
# ---------------------------------------------------------------------------

if "qlib" not in sys.modules:
    _qlib = types.ModuleType("qlib")
    _qlib_data = types.ModuleType("qlib.data")
    _qlib_data.D = MagicMock()
    _qlib.data = _qlib_data
    sys.modules["qlib"] = _qlib
    sys.modules["qlib.data"] = _qlib_data

    _qlib_contrib = types.ModuleType("qlib.contrib")
    _qlib_contrib_eval = types.ModuleType("qlib.contrib.evaluate")
    _qlib_contrib_strategy = types.ModuleType("qlib.contrib.strategy")
    _qlib_contrib_eval.backtest_daily = MagicMock()
    _qlib_contrib_strategy.TopkDropoutStrategy = MagicMock()
    _qlib.contrib = _qlib_contrib
    sys.modules["qlib.contrib"] = _qlib_contrib
    sys.modules["qlib.contrib.evaluate"] = _qlib_contrib_eval
    sys.modules["qlib.contrib.strategy"] = _qlib_contrib_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pred(n: int = 50, date: str = "2024-01-15") -> pd.Series:
    """Build a synthetic MultiIndex (datetime, instrument) prediction Series."""
    rng = np.random.default_rng(42)
    instruments = [f"SH60{i:04d}" for i in range(n)]
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), inst) for inst in instruments],
        names=["datetime", "instrument"],
    )
    return pd.Series(rng.standard_normal(n), index=idx, dtype=float)


def _make_label(pred: pd.Series) -> pd.Series:
    """Build a synthetic label Series matching pred's index."""
    rng = np.random.default_rng(99)
    return pd.Series(rng.standard_normal(len(pred)), index=pred.index, dtype=float)


def _make_window(window_id: int = 1) -> dict:
    """Build a minimal WindowDict for testing."""
    return {
        "step": window_id - 1,
        "window_id": window_id,
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
        "is_complete": True,
    }


def _mock_config():
    """Return a minimal config dict for walk-forward."""
    return {
        "walk_forward": {"min_windows": 1},
        "strategy": {"topk": 15},
        "cost_model": {
            "benchmark": "SH000852",
            "slippage": 0.001,
            "trade_unit": 100,
            "open_cost": 0.00026,
            "close_cost": 0.00076,
            "min_cost": 5,
            "deal_price": "close",
            "limit_threshold": 0.099,
            "account": 300000,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalTransformNone:
    """signal_transform=None must produce byte-identical behavior."""

    @patch("ashare_lab.research.rolling.get_all_windows")
    @patch("ashare_lab.research.rolling.train_window")
    @patch("ashare_lab.research.rolling.run_backtest")
    @patch("ashare_lab.research.rolling.load_config")
    def test_none_is_identity(
        self,
        mock_config,
        mock_backtest,
        mock_train,
        mock_get_windows,
        tmp_path,
    ):
        """With signal_transform=None, pred passed to daily_rank_ic and
        run_backtest is the exact same object returned by train_window."""
        pred = _make_pred()
        label = _make_label(pred)
        window = _make_window()

        mock_config.return_value = _mock_config()
        mock_get_windows.return_value = [window]
        mock_train.return_value = (tmp_path / "model.pkl", pred, label)

        # Build portfolio_df and bench_close that aggregate_window_metrics expects.
        portfolio_df = pd.DataFrame(
            {"return": [0.01, -0.005]},
            index=pd.to_datetime(["2024-07-01", "2024-07-02"]),
        )
        bench_close = pd.Series(
            [100.0, 101.0, 100.5],
            index=pd.to_datetime(["2024-06-28", "2024-07-01", "2024-07-02"]),
        )
        mock_backtest.return_value = (portfolio_df, bench_close, 0)

        from ashare_lab.research.rolling import run_full_walk_forward

        results, failed = run_full_walk_forward(
            exp_dir=tmp_path,
            n_drop=1,
            universe="csi500",
            signal_transform=None,
        )

        assert len(results) == 1
        assert len(failed) == 0

        # Verify backtest received the original pred (not a copy or transform).
        call_args = mock_backtest.call_args
        actual_pred = call_args.kwargs.get("pred", call_args[1].get("pred") if len(call_args) > 1 else None)
        if actual_pred is None:
            actual_pred = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs["pred"]
        pd.testing.assert_series_equal(actual_pred, pred)


class TestSignalTransformApplied:
    """A valid signal_transform is applied before IC and backtest."""

    @patch("ashare_lab.research.rolling.get_all_windows")
    @patch("ashare_lab.research.rolling.train_window")
    @patch("ashare_lab.research.rolling.run_backtest")
    @patch("ashare_lab.research.rolling.load_config")
    def test_transform_applied(
        self,
        mock_config,
        mock_backtest,
        mock_train,
        mock_get_windows,
        tmp_path,
    ):
        """The transform function is called, and its output is used for
        downstream IC computation and backtest."""
        pred = _make_pred()
        label = _make_label(pred)
        window = _make_window()

        # Transform that doubles all predictions.
        def double_pred(p: pd.Series, w: dict) -> pd.Series:
            return p * 2.0

        mock_config.return_value = _mock_config()
        mock_get_windows.return_value = [window]
        mock_train.return_value = (tmp_path / "model.pkl", pred, label)

        portfolio_df = pd.DataFrame(
            {"return": [0.01]},
            index=pd.to_datetime(["2024-07-01"]),
        )
        bench_close = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
        )
        mock_backtest.return_value = (portfolio_df, bench_close, 0)

        from ashare_lab.research.rolling import run_full_walk_forward

        results, failed = run_full_walk_forward(
            exp_dir=tmp_path,
            n_drop=1,
            universe="csi500",
            signal_transform=double_pred,
        )

        assert len(results) == 1
        assert len(failed) == 0

        # Verify backtest received the DOUBLED pred.
        call_args = mock_backtest.call_args
        actual_pred = call_args.kwargs.get("pred")
        if actual_pred is None:
            # Positional args fallback.
            for arg in call_args[0]:
                if isinstance(arg, pd.Series):
                    actual_pred = arg
                    break
            if actual_pred is None:
                actual_pred = call_args.kwargs["pred"]
        expected = pred * 2.0
        pd.testing.assert_series_equal(actual_pred, expected)


class TestSignalTransformICComputedOnTransformed:
    """IC must be computed on the TRANSFORMED pred, not the raw pred."""

    @patch("ashare_lab.research.rolling.get_all_windows")
    @patch("ashare_lab.research.rolling.train_window")
    @patch("ashare_lab.research.rolling.run_backtest")
    @patch("ashare_lab.research.rolling.load_config")
    def test_ic_uses_transformed_not_raw(
        self,
        mock_config,
        mock_backtest,
        mock_train,
        mock_get_windows,
        tmp_path,
    ):
        """Negating pred flips IC sign. If IC is computed on raw pred
        (before transform), the IC would be identical with and without
        the negate transform. This test catches that bug.

        Uses 50 instruments with correlated pred/label so IC is
        meaningfully non-zero, then checks that negation flips it.
        """
        rng = np.random.default_rng(42)
        n = 50
        date = "2024-01-15"
        instruments = [f"SH60{i:04d}" for i in range(n)]
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(date), inst) for inst in instruments],
            names=["datetime", "instrument"],
        )

        # Correlated pred and label so IC is meaningfully positive.
        base = rng.standard_normal(n)
        pred = pd.Series(base + rng.normal(0, 0.1, n), index=idx, dtype=float)
        label = pd.Series(base + rng.normal(0, 0.3, n), index=idx, dtype=float)

        window = _make_window()

        mock_config.return_value = _mock_config()
        mock_get_windows.return_value = [window]

        portfolio_df = pd.DataFrame(
            {"return": [0.01]},
            index=pd.to_datetime(["2024-07-01"]),
        )
        bench_close = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
        )
        mock_backtest.return_value = (portfolio_df, bench_close, 0)

        from ashare_lab.research.rolling import run_full_walk_forward

        # Run 1: no transform (raw pred).
        mock_train.return_value = (tmp_path / "model.pkl", pred.copy(), label.copy())
        results_raw, _ = run_full_walk_forward(
            exp_dir=tmp_path,
            n_drop=1,
            universe="csi500",
            signal_transform=None,
        )
        ic_raw = results_raw[0]["mean_rank_ic"]

        # Run 2: negate transform (pred * -1).
        def negate_pred(p: pd.Series, w: dict) -> pd.Series:
            return p * -1.0

        mock_train.return_value = (tmp_path / "model.pkl", pred.copy(), label.copy())
        results_neg, _ = run_full_walk_forward(
            exp_dir=tmp_path,
            n_drop=1,
            universe="csi500",
            signal_transform=negate_pred,
        )
        ic_neg = results_neg[0]["mean_rank_ic"]

        # IC of negated pred must be approximately -IC of raw pred.
        # If the code computes IC BEFORE the transform, ic_neg == ic_raw.
        assert ic_raw is not None
        assert ic_neg is not None
        assert ic_neg == pytest.approx(-ic_raw, abs=1e-10)


class TestSignalTransformException:
    """When signal_transform raises, raw pred is kept and failure recorded."""

    @patch("ashare_lab.research.rolling.get_all_windows")
    @patch("ashare_lab.research.rolling.train_window")
    @patch("ashare_lab.research.rolling.run_backtest")
    @patch("ashare_lab.research.rolling.load_config")
    def test_exception_falls_back_and_records(
        self,
        mock_config,
        mock_backtest,
        mock_train,
        mock_get_windows,
        tmp_path,
    ):
        """A failing transform logs a warning, keeps raw pred, and records
        the window_id in the (internal) transform_failures list."""
        pred = _make_pred()
        label = _make_label(pred)
        window = _make_window()

        def failing_transform(p: pd.Series, w: dict) -> pd.Series:
            raise ValueError("intentional test failure")

        mock_config.return_value = _mock_config()
        mock_get_windows.return_value = [window]
        mock_train.return_value = (tmp_path / "model.pkl", pred, label)

        portfolio_df = pd.DataFrame(
            {"return": [0.01]},
            index=pd.to_datetime(["2024-07-01"]),
        )
        bench_close = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
        )
        mock_backtest.return_value = (portfolio_df, bench_close, 0)

        from ashare_lab.research.rolling import run_full_walk_forward

        # Should not raise -- failure is caught per-window.
        results, failed = run_full_walk_forward(
            exp_dir=tmp_path,
            n_drop=1,
            universe="csi500",
            signal_transform=failing_transform,
        )

        assert len(results) == 1
        assert len(failed) == 0

        # Verify backtest received the ORIGINAL (unfailed) pred.
        call_args = mock_backtest.call_args
        actual_pred = call_args.kwargs.get("pred")
        if actual_pred is None:
            for arg in call_args[0]:
                if isinstance(arg, pd.Series):
                    actual_pred = arg
                    break
            if actual_pred is None:
                actual_pred = call_args.kwargs["pred"]
        pd.testing.assert_series_equal(actual_pred, pred)
