"""Unit tests for ashare_lab.research.smoke_test.

Six tests exercise window date arithmetic without qlib runtime:
1. Step 0 train_end is 2022-12-31
2. Step 0 test span >= 170 calendar days
3. Step 5 test_end is clamped to latest_trading_day
4. is_complete is True for a full window (step 0)
5. is_complete is False for a short last window (mocked)
6. get_all_windows returns >= 5 windows
"""

from __future__ import annotations

import datetime as dt
import unittest.mock as mock

from ashare_lab.research.smoke_test import get_all_windows, get_window


# ---------------------------------------------------------------------------
# Test 1: step 0 train_end is base_train_end (2022-12-31)
# ---------------------------------------------------------------------------

def test_step0_train_end() -> None:
    """W1 (step=0) train_end must equal base_train_end from config."""
    w = get_window(0)
    assert w["train_end"] == "2022-12-31"
    assert w["window_id"] == 1
    assert w["step"] == 0
    # With train_window_years=3, W1 train_start = base_train_end - 3yr.
    assert w["train_start"] == "2019-12-31"


# ---------------------------------------------------------------------------
# Test 2: step 0 test span >= 170 calendar days
# ---------------------------------------------------------------------------

def test_step0_test_span() -> None:
    """W1 test segment must span >= 170 calendar days (full 6-month window)."""
    w = get_window(0)
    test_start = dt.date.fromisoformat(w["test_start"])
    test_end = dt.date.fromisoformat(w["test_end"])
    span_days = (test_end - test_start).days
    assert span_days >= 170, (
        f"Expected test span >= 170 days, got {span_days} "
        f"({w['test_start']} to {w['test_end']})"
    )


# ---------------------------------------------------------------------------
# Test 3: step 5 test_end is clamped to latest_trading_day
# ---------------------------------------------------------------------------

def test_step5_test_end_clamped() -> None:
    """W6 (step=5) test_end must not exceed latest_trading_day()."""
    from ashare_lab.data.calendar import latest_trading_day  # noqa: PLC0415

    w = get_window(5)
    test_end = dt.date.fromisoformat(w["test_end"])
    latest = latest_trading_day()
    assert test_end <= latest, (
        f"test_end {test_end} exceeds latest_trading_day {latest}"
    )


# ---------------------------------------------------------------------------
# Test 4: is_complete True for full window (step 0)
# ---------------------------------------------------------------------------

def test_is_complete_true_for_full_window() -> None:
    """W1 is a full 6-month test window; is_complete must be True."""
    w = get_window(0)
    assert w["is_complete"] is True


# ---------------------------------------------------------------------------
# Test 5: is_complete False for a short last window
# ---------------------------------------------------------------------------

def test_is_complete_false_for_short_window() -> None:
    """is_complete must be False when test_end is clamped to a near date.

    We mock latest_trading_day to a date only 30 days after test_start
    for step 5, ensuring test span < 170 days.
    """
    from ashare_lab.research import smoke_test as st  # noqa: PLC0415

    # W6 (step=5) test_start is 2026-01-01 (unclamped test_end 2026-06-30).
    # Mock latest_trading_day to return 2026-03-01 (59 calendar days after
    # test_start, well below the 170-day threshold).
    fake_latest = dt.date(2026, 3, 1)
    with mock.patch.object(st, "latest_trading_day", return_value=fake_latest):
        # Use st.get_window so the patched latest_trading_day is resolved
        # via the smoke_test module namespace (not the test's own binding).
        w = st.get_window(5)

    test_start = dt.date.fromisoformat(w["test_start"])
    test_end = dt.date.fromisoformat(w["test_end"])
    span_days = (test_end - test_start).days
    assert span_days < 170, f"Expected short span, got {span_days}"
    assert w["is_complete"] is False


# ---------------------------------------------------------------------------
# Test 6: get_all_windows returns >= 5 windows
# ---------------------------------------------------------------------------

def test_get_all_windows_min_count() -> None:
    """At 2026-06-15, get_all_windows must return at least 5 windows."""
    windows = get_all_windows()
    assert len(windows) >= 5, (
        f"Expected >= 5 walk-forward windows, got {len(windows)}"
    )
    # All window_ids must be sequential starting from 1.
    for i, w in enumerate(windows):
        assert w["window_id"] == i + 1
        assert w["step"] == i
