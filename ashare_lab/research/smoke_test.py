"""Single-window (W1) smoke test: Alpha158 + LGBModel on CSI500.

Provides get_window(), get_all_windows(), and run_smoke_test().
Window date arithmetic uses dateutil.relativedelta for exact month offsets.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from ashare_lab.config import load_config
from ashare_lab.data.calendar import latest_trading_day
from ashare_lab.research.train import ALPHA158_WARMUP_START, apply_price_filter

log = logging.getLogger(__name__)

# Minimum test span in calendar days to mark a window as "complete".
# Derived from step_months=6 minus weekends/holidays (~170 days).
_MIN_TEST_DAYS = 170


def get_window(step: int) -> dict:
    """Return a WindowDict for the given 0-based walk-forward step.

    Uses expanding window: train_start is fixed, train_end grows by
    step_months per step. valid and test each span step_months. test_end
    is clamped to latest_trading_day().

    Args:
        step: 0-based walk-forward step index.

    Returns:
        WindowDict with keys: step, window_id, train_start, train_end,
        valid_start, valid_end, test_start, test_end, is_complete.
        All date fields are ISO-format strings (YYYY-MM-DD).
    """
    cfg = load_config()
    wf = cfg["walk_forward"]
    train_start: str = wf["train_start"]
    base_train_end: str = wf["base_train_end"]
    step_months: int = wf["step_months"]

    base_end = dt.date.fromisoformat(base_train_end)
    train_window_years: int | None = wf.get("train_window_years")  # None = expanding

    train_end = base_end + relativedelta(months=step * step_months)
    if train_window_years:
        # Rolling window: train_start = train_end - N years (floored by safety floor)
        rolling_start = train_end - relativedelta(years=train_window_years)
        safety_floor = dt.date.fromisoformat(train_start)
        train_start = str(max(rolling_start, safety_floor))
    valid_start = train_end + relativedelta(days=1)
    valid_end = valid_start + relativedelta(months=step_months) - relativedelta(days=1)
    test_start = valid_end + relativedelta(days=1)
    test_end_raw = test_start + relativedelta(months=step_months) - relativedelta(days=1)

    # Clamp test_end to latest trading day (inclusive).
    latest = latest_trading_day()
    test_end = min(test_end_raw, latest)

    test_span_days = (test_end - test_start).days
    is_complete = test_span_days >= _MIN_TEST_DAYS

    return {
        "step": step,
        "window_id": step + 1,
        "train_start": train_start,
        "train_end": str(train_end),
        "valid_start": str(valid_start),
        "valid_end": str(valid_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "is_complete": is_complete,
    }


def get_all_windows() -> list[dict]:
    """Return all WindowDicts where test_start <= latest_trading_day().

    Iterates steps starting at 0, stopping when test_start would exceed
    the latest trading day.

    Returns:
        list of WindowDict (may be empty if no valid windows exist).
    """
    latest = latest_trading_day()
    windows = []
    step = 0
    while True:
        w = get_window(step)
        test_start = dt.date.fromisoformat(w["test_start"])
        if test_start > latest:
            break
        windows.append(w)
        step += 1
    return windows


def run_smoke_test(provider_uri: Path | None = None) -> dict:
    """Run a W1 smoke test: Alpha158 + LGBModel on CSI500.

    Initialises qlib, builds Alpha158 handler on CSI500 with warmup
    start 2017-01-01, wraps in DatasetH with W1 train/valid/test segments,
    trains LGBModel, predicts on test set, applies price filter, and
    computes Spearman RankIC inline.

    Args:
        provider_uri: Path to qlib data root. If None, falls back to
            DEFAULT_PROVIDER_URI from ashare_lab.data.update.

    Returns:
        SmokeResult dict:
            window_id (int): always 1 (W1 only)
            rank_ic (float): Spearman RankIC for the W1 test period
            n_predictions (int): number of predictions after price filter
            ok (bool): rank_ic is finite and non-NaN
    """
    import qlib  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
    from qlib.contrib.model.gbdt import LGBModel  # noqa: PLC0415
    from qlib.data.dataset import DatasetH  # noqa: PLC0415
    from scipy.stats import spearmanr  # noqa: PLC0415

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415

    if provider_uri is None:
        provider_uri = DEFAULT_PROVIDER_URI

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)

    w1 = get_window(0)

    log.info(
        "W1 smoke test: train %s..%s, valid %s..%s, test %s..%s",
        w1["train_start"],
        w1["train_end"],
        w1["valid_start"],
        w1["valid_end"],
        w1["test_start"],
        w1["test_end"],
    )

    # Alpha158 is a DataHandlerLP (NOT a Dataset); wrap in DatasetH.
    handler = Alpha158(
        instruments="csi500",
        start_time=ALPHA158_WARMUP_START,
        end_time=w1["test_end"],
        fit_start_time=w1["train_start"],
        fit_end_time=w1["train_end"],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (w1["train_start"], w1["train_end"]),
            "valid": (w1["valid_start"], w1["valid_end"]),
            "test": (w1["test_start"], w1["test_end"]),
        },
    )

    model = LGBModel(random_state=42, verbose=-1)
    model.fit(dataset)

    pred = model.predict(dataset, segment="test")
    # pred is a MultiIndex Series (datetime, instrument) -> score

    # Extract labels for the test period.
    label = dataset.prepare("test", col_set="label").iloc[:, 0]

    # Apply price filter via shared helper from train.py (threshold read from config).
    pred = apply_price_filter(pred, w1["test_start"], w1["test_end"], universe="csi500")

    n_predictions = len(pred)
    log.info("predictions after price filter: %d", n_predictions)

    # Inline Spearman RankIC across all (date, instrument) pairs.
    pred_df = pred.unstack("instrument")
    label_df = label.unstack("instrument")
    common_dates = pred_df.index.intersection(label_df.index)

    ic_values = []
    for date in common_dates:
        p = pred_df.loc[date].dropna()
        lbl = label_df.loc[date].reindex(p.index).dropna()
        common_idx = p.index.intersection(lbl.index)
        if len(common_idx) >= 5:
            rho, _ = spearmanr(p[common_idx].values, lbl[common_idx].values)
            if not math.isnan(rho):
                ic_values.append(rho)

    rank_ic = float(pd.Series(ic_values).mean()) if ic_values else float("nan")
    ok = not math.isnan(rank_ic) and math.isfinite(rank_ic)

    log.info("W1 RankIC = %.6f, ok = %s", rank_ic if ok else float("nan"), ok)

    return {
        "window_id": 1,
        "rank_ic": rank_ic,
        "n_predictions": n_predictions,
        "ok": ok,
    }
