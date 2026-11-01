"""Tests for market_today and the golden snapshot month.

Both features exist because the host clock and the exchange clock disagree.
The fixture time is chosen so they land on different dates: 2026-11-01 18:30 UTC
is 02:30 on 11-02 in Shanghai and 14:30 on 11-01 in UTC-4.
"""
from __future__ import annotations

import datetime as real_dt

import pytest

from ashare_lab.data import calendar as cal_mod
from ashare_lab.paper import ledger as ledger_mod


class _StubDatetime:
    @classmethod
    def now(cls, tz: object = None) -> real_dt.datetime:
        return real_dt.datetime(2026, 11, 1, 18, 30, tzinfo=real_dt.timezone.utc).astimezone(tz)


class _StubDt:
    datetime = _StubDatetime
    date = real_dt.date


def test_market_today_uses_the_exchange_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_mod, "dt", _StubDt)

    assert cal_mod.market_today() == real_dt.date(2026, 11, 2)


def test_market_today_differs_from_the_host_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the helper: at this instant a UTC-4 host still says 11-01."""
    monkeypatch.setattr(cal_mod, "dt", _StubDt)

    host_date = real_dt.datetime(2026, 11, 1, 18, 30, tzinfo=real_dt.timezone.utc).astimezone(
        real_dt.timezone(real_dt.timedelta(hours=-4))
    ).date()

    assert host_date == real_dt.date(2026, 11, 1)
    assert cal_mod.market_today() != host_date


def test_golden_period_follows_the_exchange_month(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_mod, "market_today", lambda: real_dt.date(2026, 11, 2))

    assert ledger_mod._golden_period() == "2026-11"


def test_golden_period_is_not_the_host_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """On 11-01 a UTC-4 host would file the snapshot under 2026-10."""
    monkeypatch.setattr(cal_mod, "market_today", lambda: real_dt.date(2026, 11, 2))

    assert ledger_mod._golden_period() != "2026-10"