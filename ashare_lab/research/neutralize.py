"""Factor/style neutralization for prediction scores.

Implements Barra-style cross-sectional neutralization: regress prediction
scores against Size, Volatility, and Momentum factors, then use residuals
as the neutralized alpha signal for Top-K portfolio construction.

All qlib imports are deferred inside function bodies so this module is
importable without a qlib runtime (required for unit test isolation).

Exports:
    compute_style_factors: Fetch and z-score normalize style factors.
    neutralize_predictions: Cross-sectional OLS neutralization per date.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum instruments per date to run cross-sectional regression.
# With 3 regressors + intercept, fewer than 30 observations produces
# unreliable OLS estimates; return predictions unchanged.
_MIN_INSTRUMENTS = 30


def compute_style_factors(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """Fetch and z-score normalize three Barra-style risk factors.

    Factors:
        size: log($close * $volume) -- market-cap proxy (no shares outstanding
            in qlib; close * volume approximates daily turnover value).
        volatility: Std($close, 20) / Mean($close, 20) -- 20-day coefficient
            of variation.
        momentum: Ref($close, -20) / $close - 1 -- 20-day price momentum.

    Each factor is cross-sectionally z-score normalized per date (mean=0,
    std=1) so regression coefficients are comparable.

    Args:
        instruments: List of qlib instrument codes.
        start_time: Start date string (YYYY-MM-DD).
        end_time: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with MultiIndex (datetime, instrument), columns
        ["size", "volatility", "momentum"]. NaN values are preserved
        (caller decides how to handle).
    """
    from qlib.data import D  # noqa: PLC0415

    fields = [
        "$close",
        "$volume",
        "Std($close, 20)/Mean($close, 20)",
        "Ref($close, -20)/$close - 1",
    ]

    raw = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
    )

    if raw is None or raw.empty:
        log.warning("compute_style_factors: no data returned from D.features")
        return pd.DataFrame(
            columns=["size", "volatility", "momentum"],
            index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]),
        )

    # Ensure index order is (datetime, instrument) for consistency.
    if raw.index.names[0] != "datetime":
        raw = raw.swaplevel().sort_index()

    # Compute raw factors.
    close = raw["$close"]
    volume = raw["$volume"]
    size_raw = np.log(close * volume)
    volatility_raw = raw["Std($close, 20)/Mean($close, 20)"]
    momentum_raw = raw["Ref($close, -20)/$close - 1"]

    factors = pd.DataFrame(
        {
            "size": size_raw,
            "volatility": volatility_raw,
            "momentum": momentum_raw,
        },
        index=raw.index,
    )

    # Cross-sectional z-score normalization per date.
    grouped = factors.groupby(level="datetime")
    means = grouped.transform("mean")
    stds = grouped.transform("std")
    # Avoid division by zero: if std is 0 (all identical), factor becomes 0.
    stds = stds.replace(0.0, np.nan)
    factors = (factors - means) / stds
    factors = factors.fillna(0.0)

    return factors


def neutralize_predictions(
    pred: pd.Series,
    factors: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional OLS neutralization of predictions against style factors.

    For each date, runs OLS regression:
        pred_i = alpha + beta_size * Size_i + beta_vol * Vol_i
                 + beta_mom * Mom_i + residual_i

    Returns residuals as the neutralized prediction scores (pure alpha,
    orthogonal to the style factors).

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float.
        factors: DataFrame from compute_style_factors with columns
            ["size", "volatility", "momentum"] and matching MultiIndex.

    Returns:
        Series with same structure as pred. For dates with fewer than
        _MIN_INSTRUMENTS valid observations, predictions are returned
        unchanged (no regression attempted).
    """
    if pred.empty:
        return pred.copy()

    results = []

    # Get unique dates from pred.
    dates = pred.index.get_level_values("datetime").unique()

    for date in dates:
        # Extract cross-section for this date.
        try:
            pred_cs = pred.xs(date, level="datetime")
        except KeyError:
            continue

        # Get factors for this date; skip if date not in factors.
        try:
            factors_cs = factors.xs(date, level="datetime")
        except KeyError:
            # No factor data for this date; keep predictions unchanged.
            idx = pd.MultiIndex.from_tuples(
                [(date, inst) for inst in pred_cs.index],
                names=["datetime", "instrument"],
            )
            results.append(pd.Series(pred_cs.values, index=idx, dtype=float))
            continue

        # Align on common instruments and drop NaN.
        common_inst = pred_cs.index.intersection(factors_cs.index)
        pred_aligned = pred_cs.reindex(common_inst)
        factors_aligned = factors_cs.reindex(common_inst)

        # Drop rows where any factor is NaN.
        valid_mask = factors_aligned.notna().all(axis=1) & pred_aligned.notna()
        pred_valid = pred_aligned[valid_mask]
        factors_valid = factors_aligned[valid_mask]

        if len(pred_valid) < _MIN_INSTRUMENTS:
            # Too few instruments; return predictions unchanged for this date.
            idx = pd.MultiIndex.from_tuples(
                [(date, inst) for inst in pred_cs.index],
                names=["datetime", "instrument"],
            )
            results.append(pd.Series(pred_cs.values, index=idx, dtype=float))
            continue

        # OLS via numpy.linalg.lstsq: y = X @ beta, residuals = y - X @ beta.
        y = pred_valid.values.astype(np.float64)
        x_mat = np.column_stack([
            np.ones(len(pred_valid)),  # intercept
            factors_valid["size"].values,
            factors_valid["volatility"].values,
            factors_valid["momentum"].values,
        ]).astype(np.float64)

        beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        fitted = x_mat @ beta
        residuals = y - fitted

        # Build result for valid instruments only.
        idx = pd.MultiIndex.from_tuples(
            [(date, inst) for inst in pred_valid.index],
            names=["datetime", "instrument"],
        )
        results.append(pd.Series(residuals, index=idx, dtype=float))

    if not results:
        return pred.copy()

    return pd.concat(results)
