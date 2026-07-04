"""Per-window RankIC and aggregate metrics for walk-forward research.

Provides daily_rank_ic() and aggregate_window_metrics(), which are the
primary metric functions consumed by rolling.py.

All qlib imports are deferred inside function bodies so this module is
importable without a qlib runtime (required for unit test isolation).
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# Minimum instruments per date to compute IC (fewer => unreliable correlation).
_MIN_INSTRUMENTS = 5

# Formal JSONL record schema for matrix experiment results.
# matrix_runner.py writes these keys; analyze_matrix.py reads them.
CELL_SCHEMA_KEYS = ["model", "window", "ic", "excess", "maxdd", "completed_at"]


def daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    """Compute per-date Spearman rank IC between predictions and labels.

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float. Prediction
            score from the model for each (date, instrument) pair.
        label: MultiIndex Series (datetime, instrument) -> float. Realized
            next-period return label for each (date, instrument) pair.

    Returns:
        Series with DatetimeIndex, values = Spearman IC per date, name="rank_ic".
        Dates with fewer than _MIN_INSTRUMENTS common instruments or zero
        variance are dropped (NaN IC skipped). Returns empty Series with
        name="rank_ic" and dtype=float when pred or label is empty.
    """
    if pred.empty or label.empty:
        return pd.Series(name="rank_ic", dtype=float)

    pred_df = pred.unstack("instrument")
    label_df = label.unstack("instrument")
    common_dates = pred_df.index.intersection(label_df.index)

    ic_values: dict = {}
    for date in common_dates:
        p = pred_df.loc[date].dropna()
        lbl = label_df.loc[date].reindex(p.index).dropna()
        common_idx = p.index.intersection(lbl.index)
        if len(common_idx) < _MIN_INSTRUMENTS:
            continue
        rho, _ = spearmanr(p[common_idx].values, lbl[common_idx].values)
        # spearmanr returns NaN when variance is zero (all identical values)
        if pd.isna(rho):
            log.debug("skipping zero-variance date %s", date)
            continue
        ic_values[date] = float(rho)

    result = pd.Series(ic_values, name="rank_ic", dtype=float)
    return result


def compute_max_drawdown(portfolio_df: pd.DataFrame) -> float:
    """Compute maximum drawdown from daily portfolio returns.

    Args:
        portfolio_df: DataFrame with a "return" column of daily returns.

    Returns:
        Maximum drawdown as a negative float (e.g. -0.15 for 15%).
        Returns 0.0 if portfolio_df is empty or all returns are non-negative
        such that no drawdown occurs.
    """
    if portfolio_df.empty:
        return 0.0
    cumulative = (1 + portfolio_df["return"]).cumprod()
    running_max = cumulative.cummax()
    # Guard: cumulative can hit 0 when a return equals -1.0 (total loss).
    # running_max / running_max would be 0/0 -> NaN, violating the contract.
    if (running_max == 0).any():
        return -1.0
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def aggregate_window_metrics(
    rank_ic_series: pd.Series,
    portfolio_df: pd.DataFrame,
    bench_close: pd.Series,
    window_id: int,
    lot_skip_count: int | None,
) -> dict:
    """Compute the metric subset of WindowResult for one walk-forward window.

    Args:
        rank_ic_series: Series of per-date Spearman IC (from daily_rank_ic).
            Name should be "rank_ic". May be empty.
        portfolio_df: DataFrame with single DatetimeIndex and a "return" column
            (daily portfolio return). May be empty (0 rows = cash-only period).
        bench_close: Series with single DatetimeIndex of benchmark close prices
            (already xs-unpacked from MultiIndex by run_backtest).
        window_id: Integer walk-forward window identifier (1-based).
        lot_skip_count: Number of orders skipped due to lot-size rounding.
            None if the indicator structure was unrecognised.

    Returns:
        dict with 5 keys: window_id, mean_rank_ic, cumulative_excess_return,
        is_positive_excess, lot_skip_count. rolling.py merges the remaining
        WindowResult schema fields (universe, n_drop, train_end, test_start,
        test_end, pred_path).
    """
    # mean_rank_ic: explicit NaN guard (Series.mean() returns NaN not None for
    # all-NaN input).
    raw_mean = rank_ic_series.mean()
    mean_rank_ic = None if pd.isna(raw_mean) else float(raw_mean)

    # Edge case: no portfolio dates (cash-only with no trading days).
    if portfolio_df.empty:
        return {
            "window_id": window_id,
            "mean_rank_ic": mean_rank_ic,
            "cumulative_excess_return": 0.0,
            "is_positive_excess": False,
            "lot_skip_count": lot_skip_count,
        }

    # bench_return: pct_change of benchmark close, drop first NaN row.
    bench_return = bench_close.pct_change().dropna()

    # Trim benchmark to portfolio period.  The 14-day pre-fetch adds extra
    # leading rows that must not inflate excess.  pandas label-based slicing
    # handles a missing start label gracefully (inclusive range).
    bench_return = bench_return.loc[
        portfolio_df.index[0] : portfolio_df.index[-1]
    ]

    # Daily excess: reindex portfolio returns to benchmark index, fill missing
    # portfolio dates with zero (cash days during suspension etc.).
    # Reindex target is bench_return.index (NOT bench_close.index) to avoid
    # the NaN from the dropped first pct_change row.
    daily_excess = (
        portfolio_df["return"].reindex(bench_return.index).fillna(0) - bench_return
    )

    # Compounding excess return.
    cumulative_excess_return = float((1 + daily_excess).prod() - 1)
    is_positive_excess = cumulative_excess_return > 0  # strict; zero is not positive

    return {
        "window_id": window_id,
        "mean_rank_ic": mean_rank_ic,
        "cumulative_excess_return": cumulative_excess_return,
        "is_positive_excess": is_positive_excess,
        "lot_skip_count": lot_skip_count,
    }
