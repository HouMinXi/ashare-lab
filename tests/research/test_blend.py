"""Unit tests for Contract 2: blend_tra_ntra in blend.py.

Tests that:
  - 60/40 weighting is exact on synthetic data.
  - Index alignment drops non-shared index entries.
  - Empty pred returns empty Series with name="score".

All tests run without qlib. compute_style_factors is mocked to
provide synthetic factor data.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Qlib stub
# ---------------------------------------------------------------------------

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


def _make_multiindex_series(
    dates: list[str],
    instruments: list[str],
    values: list[float],
) -> pd.Series:
    """Build a MultiIndex (datetime, instrument) Series."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), sym) for d, sym in zip(dates, instruments)],
        names=["datetime", "instrument"],
    )
    return pd.Series(values, index=idx, dtype=float)


def _make_factors_df(
    dates: list[str],
    instruments: list[str],
    n: int,
) -> pd.DataFrame:
    """Build a synthetic factors DataFrame for compute_style_factors mock."""
    rng = np.random.default_rng(42)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), inst) for d, inst in zip(dates, instruments)],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "size": rng.standard_normal(n),
            "volatility": rng.standard_normal(n),
            "momentum": rng.standard_normal(n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlendWeighting:
    """The 60/40 blend must produce exactly tra_weight * pred + (1-tra_weight) * ntra."""

    @patch("ashare_lab.research.blend.compute_style_factors")
    @patch("ashare_lab.research.blend.neutralize_predictions")
    def test_exact_weighting(self, mock_neutralize, mock_factors):
        """With 2 points, verify 0.6 * pred + 0.4 * ntra exactly."""
        date = "2024-01-15"
        instruments = ["SH600001", "SH600002"]
        pred_values = [1.0, 2.0]
        ntra_values = [0.5, 1.5]

        pred = _make_multiindex_series(
            [date, date], instruments, pred_values,
        )
        pred_ntra = _make_multiindex_series(
            [date, date], instruments, ntra_values,
        )

        mock_factors.return_value = _make_factors_df(
            [date, date], instruments, 2,
        )
        mock_neutralize.return_value = pred_ntra

        from ashare_lab.research.blend import blend_tra_ntra

        result = blend_tra_ntra(pred, {}, tra_weight=0.60)

        expected_values = [
            0.60 * 1.0 + 0.40 * 0.5,   # 0.8
            0.60 * 2.0 + 0.40 * 1.5,   # 1.8
        ]
        expected = _make_multiindex_series(
            [date, date], instruments, expected_values,
        )
        expected.name = "score"

        pd.testing.assert_series_equal(result, expected)
        assert result.name == "score"

    @patch("ashare_lab.research.blend.compute_style_factors")
    @patch("ashare_lab.research.blend.neutralize_predictions")
    def test_custom_weight(self, mock_neutralize, mock_factors):
        """tra_weight=0.50 produces exact 50/50 blend."""
        date = "2024-01-15"
        instruments = ["SH600001", "SH600002"]

        pred = _make_multiindex_series([date, date], instruments, [10.0, 20.0])
        pred_ntra = _make_multiindex_series([date, date], instruments, [5.0, 15.0])

        mock_factors.return_value = _make_factors_df([date, date], instruments, 2)
        mock_neutralize.return_value = pred_ntra

        from ashare_lab.research.blend import blend_tra_ntra

        result = blend_tra_ntra(pred, {}, tra_weight=0.50)

        expected = _make_multiindex_series(
            [date, date], instruments, [7.5, 17.5],
        )
        expected.name = "score"
        pd.testing.assert_series_equal(result, expected)


class TestBlendAlignment:
    """Index intersection drops entries not in both pred and ntra."""

    @patch("ashare_lab.research.blend.compute_style_factors")
    @patch("ashare_lab.research.blend.neutralize_predictions")
    def test_alignment_drops_non_shared(self, mock_neutralize, mock_factors):
        """When ntra is missing an instrument, the result drops it."""
        date = "2024-01-15"
        pred = _make_multiindex_series(
            [date, date, date],
            ["SH600001", "SH600002", "SH600003"],
            [1.0, 2.0, 3.0],
        )
        # ntra only has 2 of 3 instruments.
        pred_ntra = _make_multiindex_series(
            [date, date],
            ["SH600001", "SH600002"],
            [0.5, 1.5],
        )

        mock_factors.return_value = _make_factors_df(
            [date, date, date],
            ["SH600001", "SH600002", "SH600003"],
            3,
        )
        mock_neutralize.return_value = pred_ntra

        from ashare_lab.research.blend import blend_tra_ntra

        result = blend_tra_ntra(pred, {})

        # Only 2 instruments in result (SH600003 dropped).
        assert len(result) == 2
        assert "SH600003" not in result.index.get_level_values("instrument")


class TestBlendEmpty:
    """Empty pred returns empty Series with name='score'."""

    def test_empty_pred(self):
        """blend_tra_ntra on empty pred returns empty with name='score'."""
        pred = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_tuples(
                [], names=["datetime", "instrument"],
            ),
        )

        from ashare_lab.research.blend import blend_tra_ntra

        result = blend_tra_ntra(pred, {})
        assert len(result) == 0
        assert result.name == "score"


class TestBlendDefaultWeight:
    """The default tra_weight must be 0.60 when caller omits the kwarg."""

    @patch("ashare_lab.research.blend.compute_style_factors")
    @patch("ashare_lab.research.blend.neutralize_predictions")
    def test_default_weight_is_060(self, mock_neutralize, mock_factors):
        """Calling blend_tra_ntra WITHOUT tra_weight kwarg must use 0.60."""
        date = "2024-01-15"
        instruments = ["SH600001", "SH600002"]
        pred_values = [1.0, 2.0]
        ntra_values = [0.5, 1.5]

        pred = _make_multiindex_series(
            [date, date], instruments, pred_values,
        )
        pred_ntra = _make_multiindex_series(
            [date, date], instruments, ntra_values,
        )

        mock_factors.return_value = _make_factors_df(
            [date, date], instruments, 2,
        )
        mock_neutralize.return_value = pred_ntra

        from ashare_lab.research.blend import blend_tra_ntra

        # Call WITHOUT tra_weight -- must use default 0.60.
        result = blend_tra_ntra(pred, {})

        expected_values = [
            0.60 * 1.0 + 0.40 * 0.5,   # 0.8
            0.60 * 2.0 + 0.40 * 1.5,   # 1.8
        ]
        expected = _make_multiindex_series(
            [date, date], instruments, expected_values,
        )
        expected.name = "score"

        pd.testing.assert_series_equal(result, expected)


class TestBlendVerdictIC:
    """Verdict IC must come from walk-forward, not a hardcoded proxy."""

    @patch("ashare_lab.research.blend_verdict.load_config")
    @patch("ashare_lab.research.blend_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.blend_verdict.write_verdict")
    def test_verdict_ic_matches_walk_forward(
        self,
        mock_write,
        mock_walk_forward,
        mock_config,
        tmp_path,
    ):
        """run_blend_verdict must produce verdict whose mean_rank_ic
        equals the mean of window ICs from run_full_walk_forward."""
        mock_config.return_value = {
            "universe": {"primary": "csi500"},
            "strategy": {"main": {"n_drop": 1}, "topk": 15},
            "gate": {
                "min_mean_rank_ic": 0.03,
                "min_positive_excess_pct": 0.50,
                "border_pass_ic_upper": 0.05,
                "border_pass_excess_upper": 0.70,
            },
        }

        # Synthetic window results with known ICs.
        window_results = [
            {
                "window_id": 1,
                "mean_rank_ic": 0.08,
                "cumulative_excess_return": 0.10,
                "is_positive_excess": True,
                "lot_skip_count": 0,
            },
            {
                "window_id": 2,
                "mean_rank_ic": 0.04,
                "cumulative_excess_return": 0.05,
                "is_positive_excess": True,
                "lot_skip_count": 0,
            },
        ]
        mock_walk_forward.return_value = (window_results, [])

        from ashare_lab.research.blend_verdict import run_blend_verdict

        verdict = run_blend_verdict(exp_dir=tmp_path, results_dir=tmp_path)

        # Verdict mean_rank_ic must be the mean of the window ICs.
        expected_mean_ic = (0.08 + 0.04) / 2.0
        assert verdict["mean_rank_ic"] == pytest.approx(expected_mean_ic)

        # Each window detail must carry the IC from walk-forward.
        assert verdict["window_details"][0]["mean_rank_ic"] == pytest.approx(0.08)
        assert verdict["window_details"][1]["mean_rank_ic"] == pytest.approx(0.04)


class TestBlendExceptionPropagation:
    """Exceptions from neutralize_predictions propagate (Contract 1 catches them)."""

    @patch("ashare_lab.research.blend.compute_style_factors")
    def test_neutralize_exception_propagates(self, mock_factors):
        """When compute_style_factors raises, the exception propagates
        to the caller (not caught inside blend_tra_ntra)."""
        date = "2024-01-15"
        pred = _make_multiindex_series(
            [date, date], ["SH600001", "SH600002"], [1.0, 2.0],
        )

        mock_factors.side_effect = RuntimeError("qlib not initialized")

        from ashare_lab.research.blend import blend_tra_ntra

        with pytest.raises(RuntimeError, match="qlib not initialized"):
            blend_tra_ntra(pred, {})
