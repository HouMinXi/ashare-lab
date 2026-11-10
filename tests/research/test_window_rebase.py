"""Rebasing a window must move all three segments together, and clamp the end.

Fixture windows are hand-written so the assertions do not depend on today's
calendar except where the clamp is the thing under test.
"""

from __future__ import annotations

import pytest

from ashare_lab.data.calendar import latest_trading_day
from ashare_lab.research.windows import months_between, rebase_window

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


def test_no_override_leaves_the_window_alone():
    assert rebase_window(BASE) == BASE


def test_a_new_anchor_moves_every_segment():
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=5)
    assert out["train_end"] == "2026-05-31"
    assert out["valid_start"] == "2026-06-01"
    assert out["valid_end"] == "2026-06-30"
    assert out["test_start"] == "2026-07-01"
    assert out["test_end"] == min("2026-11-30", latest_trading_day().isoformat())


def test_the_test_end_clamps_to_the_latest_session():
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1, test_months=120)
    assert out["test_end"] == latest_trading_day().isoformat()


def test_the_rolling_start_follows_the_anchor():
    out = rebase_window(BASE, train_end="2026-05-31", valid_months=1)
    assert out["train_start"] == "2023-05-31", "three-year rolling span must keep its length"


def test_a_zero_length_validation_is_refused():
    with pytest.raises(ValueError, match="validation set"):
        rebase_window(BASE, train_end="2026-05-31", valid_months=0)


def test_months_between_rounds_down():
    assert months_between(__import__("datetime").date(2025, 12, 31), __import__("datetime").date(2026, 6, 30)) == 5
