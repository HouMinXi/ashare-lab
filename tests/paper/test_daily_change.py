"""The price record settle consumes is built from two closes."""

from __future__ import annotations

import pytest

from ashare_lab.paper.engine_settle import _is_limit_down
from ashare_lab.paper.pipeline import _daily_change, _price_row

ROW = {
    "close": 6.70,
    "ref_close": 6.91,
    "change": -0.302,
    "volume": 1_000_000.0,
    "factor": 0.5,
}


def test_an_ex_dividend_row_is_not_read_as_a_limit_down():
    """sz300980 on 2026-09-30: stored -30.2%, actual -3.0%."""
    record = _price_row(ROW, "sz300980", {"sz300980"})

    assert record["change"] == pytest.approx(-0.0304, abs=1e-3)
    assert not _is_limit_down(record["change"], record["threshold"]), "a -3% session is not limit-down"
    assert _is_limit_down(ROW["change"], record["threshold"]), "the stored field did flag it"


def test_the_price_is_converted_to_cny():
    record = _price_row(ROW, "sz300980", set())

    assert record["close"] == pytest.approx(13.40), "6.70 stored at factor 0.5 is 13.40 CNY"
    assert record["factor"] == 0.5


def test_a_missing_factor_is_not_claimed_as_adjusted():
    record = _price_row({**ROW, "factor": None}, "sz300980", set())

    assert record["adjusted"] is False
    assert record["factor"] == 1.0


def test_a_real_limit_down_survives_the_new_path():
    row = {**ROW, "close": 9.00, "ref_close": 10.00, "change": 0.0}

    record = _price_row(row, "sz300980", set())

    assert _is_limit_down(record["change"], record["threshold"])


def test_a_missing_previous_close_falls_back_to_the_stored_field():
    assert _daily_change(close=10.0, ref_close=None, stored=0.05) == 0.05


def test_a_zero_previous_close_falls_back_rather_than_dividing():
    assert _daily_change(close=10.0, ref_close=0.0, stored=-0.01) == -0.01


def test_a_suspended_previous_row_falls_back():
    assert _daily_change(close=10.0, ref_close=float("nan"), stored=0.02) == 0.02
