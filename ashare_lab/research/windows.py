"""Window date arithmetic the training driver applies after window selection.

`get_window` pins the schedule: a training span, a validation gap of `step_months`,
then a test span of the same length. That shape cannot express the one experiment a
regime question needs -- a model trained up to the present with a short validation
gap and a test span that reaches today -- because the gap is fixed and the test
start is anchored to it. `rebase_window` moves the anchor and re-derives the three
segments from it.
"""

from __future__ import annotations

import datetime as dt

from dateutil.relativedelta import relativedelta

from ashare_lab.config import load_config
from ashare_lab.data.calendar import latest_trading_day

MIN_VALID_MONTHS = 1


def months_between(start: dt.date, end: dt.date) -> int:
    """Whole months from start to end, rounded down."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def rebase_window(
    window: dict,
    train_end: str | None = None,
    valid_months: int | None = None,
    test_months: int | None = None,
) -> dict:
    """Re-derive valid and test segments from a new training end date.

    With no override the window is returned unchanged. A rolling configuration
    moves train_start with the anchor so the span length in the recipe is kept; an
    expanding configuration leaves train_start where it is.

    Raises:
        ValueError: if valid_months is below one -- the trainer needs a validation
        set for early stopping, and an empty segment fails deep inside qlib.
    """
    if train_end is None and valid_months is None and test_months is None:
        return dict(window)

    end = dt.date.fromisoformat(train_end) if train_end else dt.date.fromisoformat(window["train_end"])
    span_months = (
        months_between(dt.date.fromisoformat(window["train_end"]), dt.date.fromisoformat(window["valid_end"]))
        or MIN_VALID_MONTHS
    )

    vm = span_months if valid_months is None else valid_months
    tm = span_months if test_months is None else test_months
    if vm < MIN_VALID_MONTHS:
        raise ValueError(f"valid_months must be at least {MIN_VALID_MONTHS}: the trainer needs a validation set")

    out = dict(window)
    out["train_end"] = end.isoformat()
    out["valid_start"] = (end + relativedelta(days=1)).isoformat()
    out["valid_end"] = (end + relativedelta(months=vm)).isoformat()
    out["test_start"] = (end + relativedelta(months=vm, days=1)).isoformat()
    out["test_end"] = min(end + relativedelta(months=vm + tm), latest_trading_day()).isoformat()
    out["is_complete"] = out["test_start"] <= out["test_end"]

    cfg = load_config().get("walk_forward", {})
    years = cfg.get("train_window_years")
    if years:
        floor = dt.date.fromisoformat(cfg["train_start"])
        out["train_start"] = max(end - relativedelta(years=years), floor).isoformat()
    return out
