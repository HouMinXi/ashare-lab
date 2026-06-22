"""60/40 TRA + neutralized-TRA blend transform.

Provides blend_tra_ntra(), a signal_transform-compatible function that
blends raw TRA predictions with style-neutralized predictions at a fixed
60/40 ratio.

The blend ratio (tra_weight=0.60) is the Phase 2.1 lock. The parameter
exists only for unit-test verification; it is never exposed to config
or CLI. The driver always calls with the default.

All qlib imports are deferred inside function bodies (via neutralize.py)
so this module is importable without a qlib runtime.
"""

from __future__ import annotations

import logging

import pandas as pd

from ashare_lab.research.neutralize import (
    compute_style_factors,
    neutralize_predictions,
)

log = logging.getLogger(__name__)


def blend_tra_ntra(
    pred: pd.Series,
    window: dict,
    *,
    tra_weight: float = 0.60,
) -> pd.Series:
    """Blend raw TRA predictions with style-neutralized predictions.

    Computes neutralized predictions (nTRA) by regressing pred against
    Barra-style size/volatility/momentum factors, then linearly blends:

        blended = tra_weight * pred + (1 - tra_weight) * pred_ntra

    Instruments and date range are derived from pred's MultiIndex
    (self-contained; no extra config needed).

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float.
            Raw TRA prediction scores from train_window.
        window: WindowDict from get_window(). Accepted for
            signal_transform compatibility but not used; all
            information is derived from pred's index.
        tra_weight: Weight for raw TRA predictions. Default 0.60
            is the Phase 2.1 locked ratio. Exposed only for tests.

    Returns:
        Blended Series with name="score", indexed on the intersection
        of pred and pred_ntra indices. Dates where neutralization falls
        back (fewer than 30 instruments) contribute unchanged predictions
        to the blend (neutralize_predictions returns pred unchanged for
        those dates).

    Raises:
        Any exception from compute_style_factors or neutralize_predictions
        propagates to the caller (Contract 1's try/except catches it and
        falls back to raw pred with a warning).
    """
    if pred.empty:
        result = pred.copy()
        result.name = "score"
        return result

    # Derive instruments and date range from pred's index.
    instruments = pred.index.get_level_values("instrument").unique().tolist()
    dates = pred.index.get_level_values("datetime")
    start = str(dates.min().date())
    end = str(dates.max().date())

    log.info(
        "blend: %d instruments, %s..%s, tra_weight=%.2f",
        len(instruments),
        start,
        end,
        tra_weight,
    )

    # Compute style factors and neutralize.
    factors = compute_style_factors(instruments, start, end)
    pred_ntra = neutralize_predictions(pred, factors)

    # Align on intersection to avoid NaN from index mismatch.
    idx = pred.index.intersection(pred_ntra.index)
    blended = tra_weight * pred.loc[idx] + (1 - tra_weight) * pred_ntra.loc[idx]
    blended.name = "score"

    log.info(
        "blend: %d predictions blended (%d raw, %d neutralized, %d intersection)",
        len(blended),
        len(pred),
        len(pred_ntra),
        len(idx),
    )

    return blended
