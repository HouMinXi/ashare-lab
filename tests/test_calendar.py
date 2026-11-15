"""Tests for ashare_lab.data.calendar."""
import datetime as dt
import pytest
from ashare_lab.data.calendar import (
    is_trading_day,
    latest_trading_day,
    trading_days_between,
    next_trading_day,
    previous_trading_day,
)


def test_is_trading_day_weekday():
    # 2026-06-12 is a Friday (trading day)
    assert is_trading_day(dt.date(2026, 6, 12)) is True


def test_is_trading_day_saturday():
    assert is_trading_day(dt.date(2026, 6, 7)) is False


def test_is_trading_day_pre_calendar():
    # Before XSHG calendar start (2006-06-13)
    assert is_trading_day(dt.date(2000, 1, 1)) is False


def test_latest_trading_day_returns_date():
    result = latest_trading_day()
    assert isinstance(result, dt.date)


def test_latest_trading_day_is_trading_day():
    result = latest_trading_day()
    assert is_trading_day(result)


def test_latest_trading_day_on_weekend():
    # Saturday -> should return Friday
    saturday = dt.date(2026, 6, 7)
    result = latest_trading_day(saturday)
    assert result == dt.date(2026, 6, 5)  # Friday


def test_latest_trading_day_pre_calendar_raises():
    with pytest.raises(ValueError, match="predates calendar start"):
        latest_trading_day(dt.date(1980, 1, 1))


def test_latest_trading_day_future_is_refused():
    # It used to clamp to the packaged calendar's last session. Past that
    # session every weekday reports as a holiday, so the pipeline skipped the
    # day and exited 0 instead of saying the calendar was stale.
    with pytest.raises(RuntimeError, match="calendar data needs updating"):
        latest_trading_day(dt.date(2099, 1, 1))


def test_trading_days_between_span():
    start = dt.date(2026, 5, 1)
    end = dt.date(2026, 5, 31)
    days = trading_days_between(start, end)
    assert len(days) > 15
    assert all(is_trading_day(d) for d in days)
    assert days == sorted(days)


def test_trading_days_between_empty():
    # Saturday to Sunday -> no trading days
    days = trading_days_between(dt.date(2026, 6, 7), dt.date(2026, 6, 7))
    assert days == []


def test_next_trading_day_from_trading_day():
    result = next_trading_day(dt.date(2026, 6, 12))  # Friday
    assert result == dt.date(2026, 6, 15)  # 2026-06-13 is Saturday


def test_previous_trading_day_from_trading_day():
    result = previous_trading_day(dt.date(2026, 6, 12))  # Friday
    assert is_trading_day(result)
    assert result < dt.date(2026, 6, 12)


def test_a_date_past_the_packaged_calendar_is_refused():
    from ashare_lab.data.calendar import _get_calendar

    beyond = _get_calendar().last_session.date() + dt.timedelta(days=1)

    with pytest.raises(RuntimeError, match="calendar data needs updating"):
        is_trading_day(beyond)
    with pytest.raises(RuntimeError, match="calendar data needs updating"):
        latest_trading_day(beyond)


def test_every_entry_point_reports_a_calendar_that_ran_out_the_same_way():
    import datetime as dt

    from ashare_lab.data import calendar as cal_mod

    cal = cal_mod._get_calendar()
    last = cal.last_session.date()
    beyond = last + dt.timedelta(days=4)

    for call in (
        lambda: cal_mod.next_trading_day(last),
        lambda: cal_mod.previous_trading_day(beyond),
        lambda: cal_mod.trading_days_between(last, beyond),
        lambda: cal_mod.latest_trading_day(beyond),
    ):
        with pytest.raises(RuntimeError, match="calendar data needs updating"):
            call()
