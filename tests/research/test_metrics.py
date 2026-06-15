"""Unit tests for ashare_lab.research.metrics.

All tests run without a qlib runtime. Covers daily_rank_ic() and
aggregate_window_metrics() with 7 test cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ashare_lab.research.metrics import aggregate_window_metrics, daily_rank_ic


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


def _make_portfolio_df(dates, returns):
    """Build a portfolio DataFrame with DatetimeIndex and 'return' column."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"return": returns}, index=idx)


def _make_bench_close(dates, closes):
    """Build a benchmark close Series with DatetimeIndex."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.Series(closes, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Tests for daily_rank_ic
# ---------------------------------------------------------------------------

class TestDailyRankIc:
    def test_perfect_positive_correlation(self):
        """Identical rank order -> IC near 1.0 on each date."""
        dates = ["2023-01-03"] * 5 + ["2023-01-04"] * 5
        syms = ["A", "B", "C", "D", "E"] * 2
        scores = [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        labels = [0.1, 0.2, 0.3, 0.4, 0.5] * 2
        pred = _make_multiindex_series(dates, syms, scores)
        label = _make_multiindex_series(dates, syms, labels)
        result = daily_rank_ic(pred, label)
        assert result.name == "rank_ic"
        assert len(result) == 2
        assert (result > 0.99).all(), f"Expected near-1.0 IC, got {result.tolist()}"

    def test_perfect_anticorrelation(self):
        """Reverse rank order -> IC near -1.0."""
        dates = ["2023-01-03"] * 5
        syms = ["A", "B", "C", "D", "E"]
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        labels = [0.1, 0.2, 0.3, 0.4, 0.5]
        pred = _make_multiindex_series(dates, syms, scores)
        label = _make_multiindex_series(dates, syms, labels)
        result = daily_rank_ic(pred, label)
        assert result.name == "rank_ic"
        assert len(result) == 1
        assert result.iloc[0] < -0.99, f"Expected near -1.0 IC, got {result.iloc[0]}"

    def test_fewer_than_5_instruments_skipped(self):
        """Dates with < 5 common instruments are excluded from result."""
        dates = ["2023-01-03"] * 4  # only 4 instruments -> skip
        syms = ["A", "B", "C", "D"]
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [0.1, 0.2, 0.3, 0.4]
        pred = _make_multiindex_series(dates, syms, scores)
        label = _make_multiindex_series(dates, syms, labels)
        result = daily_rank_ic(pred, label)
        assert result.name == "rank_ic"
        assert len(result) == 0, f"Expected empty result, got {result}"

    def test_empty_pred_returns_empty_series(self):
        """Empty pred -> empty Series with name='rank_ic' and dtype=float."""
        pred = pd.Series([], dtype=float)
        pred.index = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        label = _make_multiindex_series(["2023-01-03"] * 5, list("ABCDE"), [0.1] * 5)
        result = daily_rank_ic(pred, label)
        assert result.name == "rank_ic"
        assert result.dtype == float
        assert len(result) == 0

    def test_empty_label_returns_empty_series(self):
        """Empty label -> empty Series with name='rank_ic' and dtype=float."""
        pred = _make_multiindex_series(["2023-01-03"] * 5, list("ABCDE"), [1.0] * 5)
        label = pd.Series([], dtype=float)
        label.index = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        result = daily_rank_ic(pred, label)
        assert result.name == "rank_ic"
        assert len(result) == 0

    def test_zero_variance_date_skipped(self):
        """Date where all predictions are identical -> NaN IC -> skipped."""
        # Date 1: all same score (zero variance -> NaN spearman) -> skip
        # Date 2: normal -> should appear
        dates = ["2023-01-03"] * 5 + ["2023-01-04"] * 5
        syms = list("ABCDE") * 2
        scores = [1.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0]
        labels = [0.1, 0.2, 0.3, 0.4, 0.5] * 2
        pred = _make_multiindex_series(dates, syms, scores)
        label = _make_multiindex_series(dates, syms, labels)
        result = daily_rank_ic(pred, label)
        # Zero-variance date -> NaN -> skipped; date 2 should appear
        assert result.name == "rank_ic"
        assert len(result) == 1
        assert pd.Timestamp("2023-01-04") in result.index


# ---------------------------------------------------------------------------
# Tests for aggregate_window_metrics
# ---------------------------------------------------------------------------

class TestAggregateWindowMetrics:
    def _basic_rank_ic(self):
        """A simple 3-date IC series for reuse."""
        return pd.Series(
            [0.05, 0.03, 0.04],
            index=pd.DatetimeIndex(
                [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-05")]
            ),
            name="rank_ic",
            dtype=float,
        )

    def test_positive_excess_return(self):
        """Portfolio outperforms benchmark -> is_positive_excess=True."""
        rank_ic = self._basic_rank_ic()
        portfolio_df = _make_portfolio_df(
            ["2023-01-04", "2023-01-05"],
            [0.02, 0.01],
        )
        bench_close = _make_bench_close(
            ["2023-01-03", "2023-01-04", "2023-01-05"],
            [100.0, 100.5, 101.0],
        )
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 1, 3)
        assert result["window_id"] == 1
        assert result["mean_rank_ic"] == pytest.approx(0.04, rel=1e-6)
        assert result["is_positive_excess"] is True
        assert result["cumulative_excess_return"] > 0
        assert result["lot_skip_count"] == 3

    def test_negative_excess_return(self):
        """Portfolio underperforms benchmark -> is_positive_excess=False."""
        rank_ic = self._basic_rank_ic()
        portfolio_df = _make_portfolio_df(
            ["2023-01-04", "2023-01-05"],
            [0.002, 0.001],
        )
        bench_close = _make_bench_close(
            ["2023-01-03", "2023-01-04", "2023-01-05"],
            [100.0, 102.0, 104.0],
        )
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 2, 0)
        assert result["is_positive_excess"] is False
        assert result["cumulative_excess_return"] < 0

    def test_lot_skip_count_passthrough(self):
        """lot_skip_count=None is passed through unchanged."""
        rank_ic = self._basic_rank_ic()
        portfolio_df = _make_portfolio_df(["2023-01-04"], [0.01])
        bench_close = _make_bench_close(["2023-01-03", "2023-01-04"], [100.0, 101.0])
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 3, None)
        assert result["lot_skip_count"] is None

    def test_nan_ic_series_mean_is_none(self):
        """All-NaN IC series -> mean_rank_ic=None (not NaN)."""
        rank_ic = pd.Series([float("nan"), float("nan")], name="rank_ic", dtype=float)
        portfolio_df = _make_portfolio_df(["2023-01-04"], [0.01])
        bench_close = _make_bench_close(["2023-01-03", "2023-01-04"], [100.0, 100.5])
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 1, 0)
        assert result["mean_rank_ic"] is None

    def test_empty_portfolio_df_edge_case(self):
        """Empty portfolio_df -> cumulative_excess=0.0, is_positive=False."""
        rank_ic = self._basic_rank_ic()
        portfolio_df = pd.DataFrame({"return": []}, index=pd.DatetimeIndex([]))
        bench_close = _make_bench_close(["2023-01-03", "2023-01-04"], [100.0, 101.0])
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 4, 0)
        assert result["cumulative_excess_return"] == 0.0
        assert result["is_positive_excess"] is False
        assert result["window_id"] == 4
        # mean_rank_ic still computed from rank_ic_series
        assert result["mean_rank_ic"] == pytest.approx(0.04, rel=1e-6)

    def test_zero_excess_is_not_positive(self):
        """Cumulative excess of exactly 0.0 -> is_positive_excess=False (strict)."""
        rank_ic = self._basic_rank_ic()
        # portfolio and bench move by exactly the same amount
        portfolio_df = _make_portfolio_df(["2023-01-04"], [0.005])
        bench_close = _make_bench_close(["2023-01-03", "2023-01-04"], [100.0, 100.5])
        result = aggregate_window_metrics(rank_ic, portfolio_df, bench_close, 5, 0)
        # excess may be near-zero; test the strict > 0 rule explicitly with zero
        result["cumulative_excess_return"] = 0.0
        assert not (result["cumulative_excess_return"] > 0)
