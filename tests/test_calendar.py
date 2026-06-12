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


def test_latest_trading_day_future_clamps():
    far_future = dt.date(2099, 1, 1)
    result = latest_trading_day(far_future)
    assert isinstance(result, dt.date)


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
