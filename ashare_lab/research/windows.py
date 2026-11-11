"""Re-anchor scheduled train, validation and test calendar spans."""
from __future__ import annotations

import datetime as dt

from dateutil.relativedelta import relativedelta

from ashare_lab.config import load_config
from ashare_lab.data.calendar import latest_trading_day


def rebase_window(
    window: dict,
    train_end: str | None = None,
    valid_months: int | None = None,
    test_months: int | None = None,
) -> dict:
    """Move a training cutoff without silently changing default span lengths.

    Validation starts the following day. Each segment lasts the requested number
    of calendar months, defaulting independently to walk_forward.step_months.
    A rolling fit retains its configured years and earliest allowed start.
    No overrides returns an unchanged copy, including the legacy completeness flag.

    Raises ValueError for non-positive spans or a test that has not started yet.
    Zero validation is not supported: early stopping requires validation samples.
    """
    if train_end is None and valid_months is None and test_months is None:
        return dict(window)

    cfg = load_config()["walk_forward"]
    vm = cfg["step_months"] if valid_months is None else valid_months
    tm = cfg["step_months"] if test_months is None else test_months
    if not isinstance(vm, int) or isinstance(vm, bool) or vm < 1:
        raise ValueError("valid_months must be at least 1: the trainer needs a validation set")
    if not isinstance(tm, int) or isinstance(tm, bool) or tm < 1:
        raise ValueError("test_months must be a positive integer")

    end = dt.date.fromisoformat(window["train_end"] if train_end is None else train_end)
    valid_start = end + dt.timedelta(days=1)
    test_start = valid_start + relativedelta(months=vm)
    test_end_raw = test_start + relativedelta(months=tm) - dt.timedelta(days=1)
    test_end = min(test_end_raw, latest_trading_day())
    if test_end < test_start:
        raise ValueError("test segment has not started: test_start is after the latest trading day")

    out = dict(window)
    out.update(
        train_end=end.isoformat(),
        valid_start=valid_start.isoformat(),
        valid_end=(test_start - dt.timedelta(days=1)).isoformat(),
        test_start=test_start.isoformat(),
        test_end=test_end.isoformat(),
        is_complete=test_end == test_end_raw,
    )
    years = cfg.get("train_window_years")
    if years:
        floor = dt.date.fromisoformat(cfg["train_start"])
        out["train_start"] = max(end - relativedelta(years=years), floor).isoformat()
    if dt.date.fromisoformat(out["train_start"]) > end:
        raise ValueError("train_start is after train_end")
    return out
