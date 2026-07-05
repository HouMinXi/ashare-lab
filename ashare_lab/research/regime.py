"""Regime detection: market-breadth signal and position-sizing filter.

Provides compute_regime_signals(), apply_regime_filter(), and
calibrate_thresholds(). Addresses W5/W6 crowding-driven drawdowns by
reducing portfolio exposure when market breadth deteriorates.

Primary signal: ma20_above_pct -- percentage of universe constituents
with close > 20-day moving average. Single well-understood breadth
indicator, chosen to avoid overfitting per P1 review consensus.

All qlib imports are deferred inside function bodies so this module is
importable without a qlib runtime (required for unit test isolation).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_regime_signals(
    date: str,
    universe: str = "csi1000",
) -> dict:
    """Compute market-breadth regime signals for a given date.

    Fetches close prices and 20-day moving average for all universe
    constituents via qlib D.features, then computes the fraction of
    instruments whose close is above their MA(20).

    Args:
        date: Trading date in YYYY-MM-DD format.
        universe: Qlib universe string (e.g. "csi1000").

    Returns:
        Dict with keys:
            ma20_above_pct: float in [0.0, 1.0], fraction of instruments
                with close > MA(20). Returns 0.5 (neutral) when data is
                missing or empty.
            date: The input date string.
    """
    from qlib.data import D  # noqa: PLC0415

    try:
        features_df = D.features(
            instruments=universe,
            fields=["$close", "Mean($close, 20)"],
            start_time=date,
            end_time=date,
        )
    except Exception:
        log.warning("regime: D.features failed for %s; returning neutral 0.5", date)
        return {"ma20_above_pct": 0.5, "date": date}

    if features_df is None or features_df.empty:
        log.debug("regime: no data for %s; returning neutral 0.5", date)
        return {"ma20_above_pct": 0.5, "date": date}

    close = features_df["$close"]
    ma20 = features_df["Mean($close, 20)"]

    # Drop rows where either close or MA20 is NaN.
    valid_mask = close.notna() & ma20.notna()
    total = int(valid_mask.sum())

    if total == 0:
        return {"ma20_above_pct": 0.5, "date": date}

    above_count = int((close[valid_mask] > ma20[valid_mask]).sum())
    ma20_above_pct = float(above_count / total)

    return {"ma20_above_pct": ma20_above_pct, "date": date}


def apply_regime_filter(
    pred: pd.Series,
    date: str,
    topk: int,
    thresholds: dict,
) -> tuple[pd.Series, int]:
    """Adjust effective topk based on regime signal.

    Does NOT retrain the model. Only adjusts position sizing:
      - ma20_above_pct < cash_threshold: go to cash (effective_topk = 0)
      - ma20_above_pct < reduce_threshold: halve (effective_topk = topk // 2)
      - otherwise: full exposure (effective_topk = topk)

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float. Model
            predictions for the given date. Returned unchanged.
        date: Trading date in YYYY-MM-DD format.
        topk: Original topk from config (e.g. 15).
        thresholds: Dict with "cash_threshold" and "reduce_threshold" keys.

    Returns:
        Tuple (pred, effective_topk):
            pred: The original prediction Series (unmodified).
            effective_topk: Adjusted topk based on regime signal.
    """
    signals = compute_regime_signals(date)
    ma20_pct = signals["ma20_above_pct"]

    cash_threshold = thresholds["cash_threshold"]
    reduce_threshold = thresholds["reduce_threshold"]

    if ma20_pct < cash_threshold:
        log.info(
            "regime: %s ma20_above_pct=%.3f < cash=%.3f -> topk=0",
            date, ma20_pct, cash_threshold,
        )
        return pred, 0

    if ma20_pct < reduce_threshold:
        effective = topk // 2
        log.info(
            "regime: %s ma20_above_pct=%.3f < reduce=%.3f -> topk=%d",
            date, ma20_pct, reduce_threshold, effective,
        )
        return pred, effective

    return pred, topk


def calibrate_thresholds(
    universe: str = "csi1000",
    end_date: str = "2022-12-31",
) -> dict:
    """Calibrate regime thresholds on pre-2023 historical data.

    Computes ma20_above_pct for every trading day from 2015-01-01 to
    end_date, then derives thresholds from the distribution:
      - reduce_threshold = 25th percentile of ma20_above_pct
      - cash_threshold = 10th percentile of ma20_above_pct

    Args:
        universe: Qlib universe string (e.g. "csi1000").
        end_date: Last date for calibration (inclusive). Defaults to
            "2022-12-31" to prevent look-ahead into W5/W6 test periods.

    Returns:
        Dict with keys:
            reduce_threshold: float, 25th percentile.
            cash_threshold: float, 10th percentile.
            calibration_end: str, the end_date used.
    """
    from qlib.data import D  # noqa: PLC0415

    start_date = "2015-01-01"

    features_df = D.features(
        instruments=universe,
        fields=["$close", "Mean($close, 20)"],
        start_time=start_date,
        end_time=end_date,
    )

    if features_df is None or features_df.empty:
        log.warning(
            "calibrate: no data for %s..%s; returning conservative defaults",
            start_date, end_date,
        )
        return {
            "reduce_threshold": 0.35,
            "cash_threshold": 0.20,
            "calibration_end": end_date,
        }

    # Extract unique trading dates from the MultiIndex.
    # D.features MultiIndex is (instrument, datetime) -- get datetime level.
    datetime_level = "datetime"
    if datetime_level not in features_df.index.names:
        # Fallback: use whichever level contains Timestamps.
        for lvl_name in features_df.index.names:
            lvl_values = features_df.index.get_level_values(lvl_name)
            if hasattr(lvl_values, "date"):
                datetime_level = lvl_name
                break

    all_dates = features_df.index.get_level_values(datetime_level).unique()

    daily_pcts: list[float] = []
    for dt in all_dates:
        day_data = features_df.xs(dt, level=datetime_level)
        close = day_data["$close"]
        ma20 = day_data["Mean($close, 20)"]

        valid = close.notna() & ma20.notna()
        total = int(valid.sum())
        if total == 0:
            continue

        above = int((close[valid] > ma20[valid]).sum())
        daily_pcts.append(float(above / total))

    if len(daily_pcts) == 0:
        return {
            "reduce_threshold": 0.35,
            "cash_threshold": 0.20,
            "calibration_end": end_date,
        }

    pcts_array = np.array(daily_pcts)
    reduce_threshold = float(np.percentile(pcts_array, 25))
    cash_threshold = float(np.percentile(pcts_array, 10))

    log.info(
        "calibrate: %d trading days, reduce=%.4f (p25), cash=%.4f (p10)",
        len(daily_pcts), reduce_threshold, cash_threshold,
    )

    return {
        "reduce_threshold": reduce_threshold,
        "cash_threshold": cash_threshold,
        "calibration_end": end_date,
    }
