"""Literal calendar contracts for re-anchoring a scheduled research window."""
from __future__ import annotations

import copy
import datetime as dt

import pytest

from ashare_lab.research import windows
from ashare_lab.research.windows import rebase_window

BASE = {
    "step": 10,
    "window_id": 11,
    "train_start": "2022-12-31",
    "train_end": "2025-12-31",
    "valid_start": "2026-01-01",
    "valid_end": "2026-06-30",
    "test_start": "2026-07-01",
    "test_end": "2026-10-30",
    "is_complete": False,
}


@pytest.fixture(autouse=True)
def frozen_environment(monkeypatch):
    config = {"walk_forward": {
        "step_months": 6, "train_start": "2018-01-01", "train_window_years": 3,
    }}
    monkeypatch.setattr(windows, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    return config


def test_no_override_leaves_the_window_alone():
    out = rebase_window(BASE)
    assert out == BASE
    assert out is not BASE


def test_a_new_anchor_moves_every_segment():
    original = dict(BASE)
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=5)
    assert BASE == original
    assert (out["train_end"], out["valid_start"], out["valid_end"], out["test_start"], out["test_end"]) == (
        "2026-05-31", "2026-06-01", "2026-06-30", "2026-07-01", "2026-11-06",
    )
    assert out["is_complete"] is False


def test_the_test_end_clamps_to_the_latest_session():
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=120)
    assert out["test_end"] == "2026-11-06"


@pytest.mark.parametrize("years, expected", [
    (1, "2025-05-31"), (2, "2024-05-31"), (3, "2023-05-31"),
    (5, "2021-05-31"), (20, "2018-01-01"),
    (None, "2022-12-31"), (0, "2022-12-31"),
])
def test_training_start_follows_configured_rolling_or_expanding_mode(frozen_environment, years, expected):
    frozen_environment["walk_forward"]["train_window_years"] = years
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1)
    assert out["train_start"] == expected


def test_a_zero_length_validation_is_refused():
    with pytest.raises(ValueError, match="validation set"):
        rebase_window(BASE, train_end="2026-05-31", valid_months=0)


@pytest.mark.parametrize("cutoff, valid_start, valid_end, test_start, test_end", [
    ("2025-12-31", "2026-01-01", "2026-06-30", "2026-07-01", "2026-12-31"),
    ("2024-02-29", "2024-03-01", "2024-08-31", "2024-09-01", "2025-02-28"),
    ("2026-05-31", "2026-06-01", "2026-11-30", "2026-12-01", "2027-05-31"),
    ("2024-08-30", "2024-08-31", "2025-02-27", "2025-02-28", "2025-08-27"),
])
def test_default_spans_keep_successive_calendar_months(
    monkeypatch, cutoff, valid_start, valid_end, test_start, test_end,
):
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2030, 1, 1))
    out = rebase_window(BASE, train_end=cutoff)
    assert (out["valid_start"], out["valid_end"], out["test_start"], out["test_end"]) == (
        valid_start, valid_end, test_start, test_end,
    )
    assert out["is_complete"] is True


@pytest.mark.parametrize("months", [0, -1, True, 1.5, "1"])
def test_non_positive_or_non_integer_test_span_is_refused(months):
    with pytest.raises(ValueError, match="test_months"):
        rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=months)


def test_same_anchor_preserves_scheduled_dates(monkeypatch):
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2026, 10, 30))
    assert rebase_window(BASE, train_end=BASE["train_end"]) == BASE


def test_validation_override_keeps_default_test_length(monkeypatch):
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2030, 1, 1))
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1)
    assert (out["valid_end"], out["test_start"], out["test_end"]) == (
        "2026-06-30", "2026-07-01", "2026-12-31",
    )


def test_configured_quarter_is_the_default_for_both_spans(monkeypatch, frozen_environment):
    frozen_environment["walk_forward"]["step_months"] = 3
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2030, 1, 1))
    out = rebase_window(BASE, train_end="2026-05-31")
    assert (out["valid_end"], out["test_start"], out["test_end"]) == (
        "2026-08-31", "2026-09-01", "2026-11-30",
    )


def test_future_test_span_is_refused(monkeypatch):
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2026, 6, 30))
    with pytest.raises(ValueError, match="test"):
        rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=5)
