"""Signal transform functions for the walk-forward pipeline.

Provides regime_mask_transform(), a signal_transform-compatible function
that masks predictions based on market-breadth regime signals. Operates
as a "shadow track" -- NaN-masked predictions cause TopkDropoutStrategy
to hold fewer positions (or go to cash) without retraining.

All qlib imports are deferred inside regime.py, so this module is
importable without a qlib runtime.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ashare_lab.research.regime import compute_regime_signals

log = logging.getLogger(__name__)

# Default thresholds for regime masking.
_DEFAULT_CASH_THRESHOLD = 0.20
_DEFAULT_REDUCE_THRESHOLD = 0.35
_DEFAULT_REDUCE_TOPN = 8


def regime_mask_transform(
    pred: pd.Series,
    window: dict,
    *,
    cash_threshold: float = _DEFAULT_CASH_THRESHOLD,
    reduce_threshold: float = _DEFAULT_REDUCE_THRESHOLD,
    reduce_topn: int = _DEFAULT_REDUCE_TOPN,
    universe: str = "csi1000",
) -> pd.Series:
    """NaN-mask predictions based on market-breadth regime signals.

    For each date in pred:
      - If ma20_above_pct < cash_threshold: set all predictions to NaN
        (go to cash).
      - If ma20_above_pct < reduce_threshold: keep only the top-N
        predictions by score, mask the rest to NaN (reduce exposure).
      - Otherwise: keep all predictions unchanged (full exposure).

    This is the NaN-mask approach from run_regime_backtest, adapted
    as a signal_transform hook. NaN predictions cause the backtest
    strategy to skip those instruments.

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float.
        window: WindowDict (accepted for signal_transform compatibility;
            universe is derived from the keyword arg, not window).
        cash_threshold: Breadth fraction below which all positions
            are masked (go to cash).
        reduce_threshold: Breadth fraction below which only the top
            reduce_topn positions are kept.
        reduce_topn: Number of top-scoring positions to keep when
            in "reduce" regime.
        universe: Qlib universe for regime signal computation.

    Returns:
        Series with same index as pred, with NaN values where the
        regime filter masked predictions.
    """
    if pred.empty:
        return pred.copy()

    result = pred.copy()
    dates = pred.index.get_level_values("datetime").unique()

    n_cash = 0
    n_reduce = 0

    for date in dates:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        signals = compute_regime_signals(date_str, universe)
        ma20_pct = signals["ma20_above_pct"]

        # Get the cross-section for this date.
        date_mask = pred.index.get_level_values("datetime") == date

        if ma20_pct < cash_threshold:
            # Go to cash: mask all predictions for this date.
            result.loc[date_mask] = np.nan
            n_cash += 1
        elif ma20_pct < reduce_threshold:
            # Reduce: keep only top-N by score, mask rest.
            date_pred = pred.loc[date_mask]
            if len(date_pred) > reduce_topn:
                top_n_idx = date_pred.nlargest(reduce_topn).index
                mask_idx = date_pred.index.difference(top_n_idx)
                result.loc[mask_idx] = np.nan
            n_reduce += 1

    if n_cash > 0 or n_reduce > 0:
        log.info(
            "regime_mask: %d dates -> %d cash, %d reduce, %d full",
            len(dates),
            n_cash,
            n_reduce,
            len(dates) - n_cash - n_reduce,
        )

    return result
