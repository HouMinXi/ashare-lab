"""A-share trading calendar backed by exchange_calendars XSHG."""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

log = logging.getLogger(__name__)

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
EXCHANGE_ID = "XSHG"


@lru_cache(maxsize=1)
def _get_calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar(EXCHANGE_ID)


_CALENDAR_RUNWAY_DAYS = 60
_runway_warned = False


def _warn_on_short_runway(cal, as_of: dt.date) -> None:
    """Say so while there is still time to update the packaged calendar."""
    global _runway_warned
    if _runway_warned:
        return
    remaining = (cal.last_session.date() - as_of).days
    if 0 <= remaining < _CALENDAR_RUNWAY_DAYS:
        _runway_warned = True
        log.warning(
            "exchange calendar ends %s, %d days after %s; update "
            "exchange_calendars before the next session past it",
            cal.last_session.date(), remaining, as_of,
        )


def is_trading_day(date: dt.date) -> bool:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    _warn_on_short_runway(cal, date)
    if ts > cal.last_session:
        # Past the packaged calendar every weekday looks like a holiday, so a
        # stale calendar would silently skip every session instead of failing.
        raise RuntimeError(
            f"exchange calendar ends at {cal.last_session.date()}; {date} is "
            "beyond it, so the calendar data needs updating"
        )
    if ts < cal.first_session:
        return False
    return cal.is_session(ts)


def market_today(now: dt.datetime | None = None) -> dt.date:
    """Today's date on the exchange, not on the machine.

    ``now`` is injectable so a test can pin the instant instead of replacing the
    datetime module.
    """
    moment = now if now is not None else dt.datetime.now(TZ_SHANGHAI)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TZ_SHANGHAI)
    return moment.astimezone(TZ_SHANGHAI).date()


def latest_trading_day(as_of: dt.date | None = None) -> dt.date:
    """Most recent completed trading day on or before *as_of* (default: today Shanghai)."""
    if as_of is None:
        as_of = market_today()
    cal = _get_calendar()
    ts = pd.Timestamp(as_of)
    _warn_on_short_runway(cal, as_of)
    if ts > cal.last_session:
        raise RuntimeError(
            f"exchange calendar ends at {cal.last_session.date()}; {as_of} is "
            "beyond it, so the calendar data needs updating"
        )
    if ts < cal.first_session:
        raise ValueError(
            f"as_of {as_of} predates calendar start {cal.first_session.date()}"
        )
    if cal.is_session(ts):
        return ts.date()
    prev = cal.previous_close(ts).normalize()
    return prev.date()


def _require_in_range(cal, moment: pd.Timestamp, what: str) -> None:
    """Every entry point reports the same way once the package runs out."""
    if moment > cal.last_session:
        raise RuntimeError(
            f"exchange calendar ends at {cal.last_session.date()}; {what} is "
            "beyond it, so the calendar data needs updating"
        )


def trading_days_between(
    start: dt.date, end: dt.date
) -> list[dt.date]:
    cal = _get_calendar()
    _require_in_range(cal, pd.Timestamp(end), f"{end}")
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [s.date() for s in sessions]


def next_trading_day(date: dt.date) -> dt.date:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    _require_in_range(cal, ts, f"{date}")
    if ts >= cal.last_session:
        # The input is inside the package; the answer is not. Say so the way
        # every other entry point does instead of leaking the library's error.
        raise RuntimeError(
            f"exchange calendar ends at {cal.last_session.date()}; there is no "
            f"session after {date}, so the calendar data needs updating"
        )
    if cal.is_session(ts):
        return cal.session_offset(ts, 1).date()
    nxt = cal.next_open(ts).normalize()
    return nxt.date()


def previous_trading_day(date: dt.date) -> dt.date:
    cal = _get_calendar()
    ts = pd.Timestamp(date)
    _require_in_range(cal, ts, f"{date}")
    if cal.is_session(ts):
        return cal.session_offset(ts, -1).date()
    prev = cal.previous_close(ts).normalize()
    return prev.date()
