"""A-share trading calendar backed by exchange_calendars XSHG."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
EXCHANGE_ID = "XSHG"


@lru_cache(maxsize=1)
def _get_calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar(EXCHANGE_ID)


def is_trading_day(date: dt.date) -> bool:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    if ts < cal.first_session or ts > cal.last_session:
        return False
    return cal.is_session(ts)


def latest_trading_day(as_of: dt.date | None = None) -> dt.date:
    """Most recent completed trading day on or before *as_of* (default: today Shanghai)."""
    if as_of is None:
        as_of = dt.datetime.now(TZ_SHANGHAI).date()
    cal = _get_calendar()
    ts = pd.Timestamp(as_of)
    if ts > cal.last_session:
        return cal.last_session.date()
    if ts < cal.first_session:
        return cal.first_session.date()
    if cal.is_session(ts):
        return ts.date()
    prev = cal.previous_close(ts).normalize()
    return prev.date()


def trading_days_between(
    start: dt.date, end: dt.date
) -> list[dt.date]:
    cal = _get_calendar()
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [s.date() for s in sessions]


def next_trading_day(date: dt.date) -> dt.date:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    if cal.is_session(ts):
        return cal.session_offset(ts, 1).date()
    nxt = cal.next_open(ts).normalize()
    return nxt.date()


def previous_trading_day(date: dt.date) -> dt.date:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    if cal.is_session(ts):
        return cal.session_offset(ts, -1).date()
    prev = cal.previous_close(ts).normalize()
    return prev.date()
