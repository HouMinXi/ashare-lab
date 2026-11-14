"""Tests for market_today, the golden snapshot month, and the commands that use them.

Both features exist because the host clock and the exchange clock disagree.

Two lessons from the review are baked in here:

* the stub date must fall in a month that is NOT the current real month,
  otherwise a host-clock implementation satisfies the same assertion;
* the test that the exchange clock is used must be able to tell Shanghai from
  any other +8/+9 zone, so the stub instant is chosen where they disagree and
  the recorded tz is checked.
"""
from __future__ import annotations

import datetime as real_dt
from argparse import Namespace

import pytest

from ashare_lab import cli
from ashare_lab.data import calendar as cal_mod
from ashare_lab.paper import ledger as ledger_mod


class _StubDatetime:
    seen_tz: object = None

    @classmethod
    def now(cls, tz: object = None) -> real_dt.datetime:
        cls.seen_tz = tz
        return real_dt.datetime(2026, 11, 1, 18, 30, tzinfo=real_dt.timezone.utc).astimezone(tz)


class _StubDt:
    datetime = _StubDatetime
    date = real_dt.date


def test_market_today_uses_the_exchange_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_mod, "dt", _StubDt)

    assert cal_mod.market_today() == real_dt.date(2026, 11, 2)


def test_market_today_asks_for_shanghai_specifically(monkeypatch: pytest.MonkeyPatch) -> None:
    """At 15:30 UTC Shanghai is still 11-01 while Tokyo has already turned over,
    so a neighbouring zone cannot satisfy this."""
    class _Afternoon(_StubDatetime):
        @classmethod
        def now(cls, tz: object = None) -> real_dt.datetime:
            cls.seen_tz = tz
            return real_dt.datetime(2026, 11, 1, 15, 30, tzinfo=real_dt.timezone.utc).astimezone(tz)

    monkeypatch.setattr(cal_mod, "dt", type("_S", (), {"datetime": _Afternoon, "date": real_dt.date}))

    assert cal_mod.market_today() == real_dt.date(2026, 11, 1)
    assert _Afternoon.seen_tz == cal_mod.TZ_SHANGHAI


def test_market_today_differs_from_the_host_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the helper: at this instant a UTC-4 host still says 11-01."""
    monkeypatch.setattr(cal_mod, "dt", _StubDt)

    host_date = real_dt.datetime(2026, 11, 1, 18, 30, tzinfo=real_dt.timezone.utc).astimezone(
        real_dt.timezone(real_dt.timedelta(hours=-4))
    ).date()

    assert host_date == real_dt.date(2026, 11, 1)
    assert cal_mod.market_today() != host_date


# The stub month must not coincide with the real current month: when it did, a
# host-clock implementation returned the same string and the tests below passed
# with the defect in place.
STUB_MARKET_DAY = real_dt.date(2027, 3, 5)


def test_the_stub_month_is_not_the_current_month() -> None:
    assert STUB_MARKET_DAY.strftime("%Y-%m") != real_dt.date.today().strftime("%Y-%m"), (
        "pick another month, or these tests stop distinguishing the two clocks"
    )


def _patch_market_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_mod, "market_today", lambda: STUB_MARKET_DAY)
    monkeypatch.setattr(ledger_mod, "market_today", lambda: STUB_MARKET_DAY, raising=False)


def test_golden_period_follows_the_exchange_month(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_market_today(monkeypatch)

    assert ledger_mod._golden_period() == "2027-03"


def test_golden_period_is_not_the_host_month(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_market_today(monkeypatch)

    assert ledger_mod._golden_period() != real_dt.date.today().strftime("%Y-%m")


@pytest.mark.parametrize("command", ["cmd_fetch_today", "cmd_chenditc_snapshot"])
def test_commands_take_their_date_from_the_exchange(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Reverting either command to the host date leaves is_trading_day with the
    host date, which this records and rejects."""
    seen: list[real_dt.date] = []
    monkeypatch.setattr(cal_mod, "market_today", lambda: STUB_MARKET_DAY)
    monkeypatch.setattr(cal_mod, "is_trading_day", lambda day: (seen.append(day), False)[1])

    assert getattr(cli, command)(Namespace()) == 0
    assert seen == [STUB_MARKET_DAY], "not a trading day path must receive the exchange date"


def test_the_diff_command_does_not_pass_a_host_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """latest_trading_day() already defaults to the exchange; passing today()
    overrides it with the host day."""
    calls: list[tuple] = []

    def _record(*args, **kwargs):
        calls.append(args)
        return STUB_MARKET_DAY

    monkeypatch.setattr(cal_mod, "latest_trading_day", _record)

    cli.cmd_chenditc_diff(Namespace())
    assert calls == [()], "the command must let the calendar choose the day"
