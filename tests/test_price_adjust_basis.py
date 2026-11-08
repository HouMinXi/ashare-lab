"""The daily write must keep one adjustment basis, not append a second one."""

from __future__ import annotations

import struct

import pandas as pd
import pytest

from ashare_lab.data.adjust import (
    PrevState,
    adjusted_return,
    next_factor,
    scale_prices,
    systemic_break,
)
from ashare_lab.data.fetcher import _apply_store_basis

# Real values from the store on 2026-10-08 and the raw tape on 2026-10-09,
# when the daily writer appended raw prices with factor 1.0.
PUDONG = PrevState(close=9.70 * 0.679638, factor=0.679638)
PUDONG_RAW_1008 = 9.70
PUDONG_RAW_1009 = 9.54


def _write_bin(path, first_slot: int, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<f", float(first_slot))
    payload += b"".join(struct.pack("<f", float(v)) for v in values)
    path.write_bytes(payload)


def _provider(tmp_path, symbols: dict[str, tuple[float, float]], dates: list[str]):
    """A store holding one close/factor row per symbol on the last date."""
    root = tmp_path / "cn_data"
    cal = root / "calendars"
    cal.mkdir(parents=True, exist_ok=True)
    (cal / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")
    for symbol, (close, factor) in symbols.items():
        for field, value in (("close", close), ("factor", factor)):
            _write_bin(root / "features" / symbol.lower() / f"{field}.day.bin", 0, [value])
    return root


def _row(open_, high, low, close, pre_close):
    return pd.DataFrame(
        [
            {
                "date": "2026-10-09",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1.0,
                "factor": 1.0,
                "change": 0.0,
                "pre_close": pre_close,
            }
        ]
    )


# --- the arithmetic ---------------------------------------------------------


def test_no_corporate_action_leaves_the_factor_alone():
    assert next_factor(PUDONG, PUDONG_RAW_1008) == pytest.approx(PUDONG.factor, rel=1e-9)


def test_a_dividend_is_absorbed_by_the_factor():
    dividend = 0.44
    factor = next_factor(PUDONG, PUDONG_RAW_1008 - dividend)
    # The factor must move by exactly the ex-dividend ratio, so that the
    # adjusted series shows no drop on the ex date.
    assert factor == pytest.approx(PUDONG.factor * PUDONG_RAW_1008 / (PUDONG_RAW_1008 - dividend))
    assert factor > PUDONG.factor


def test_non_positive_inputs_are_refused():
    with pytest.raises(ValueError):
        next_factor(PrevState(close=1.0, factor=0.0), 1.0)
    with pytest.raises(ValueError):
        next_factor(PUDONG, 0.0)


def test_scale_prices_covers_every_price_field():
    scaled = scale_prices({"open": 9.7, "high": 9.74, "low": 9.54, "close": 9.54}, 0.679638)
    assert scaled["close"] == pytest.approx(6.4837, abs=1e-4)
    assert scaled["high"] > scaled["close"]
    assert scaled["low"] <= scaled["close"]


# --- the defect this change exists to remove --------------------------------


def test_the_day_after_a_basis_change_is_not_a_forty_percent_move(tmp_path):
    """2026-10-09 as it should be written: continuity, not a fake rally."""
    root = _provider(tmp_path, {"SH600000": (PUDONG.close, PUDONG.factor)}, ["2026-10-08"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, PUDONG_RAW_1009, PUDONG_RAW_1008)}

    out = _apply_store_basis(rows, "2026-10-09", root)
    written = out["SH600000"].iloc[0]

    assert written["factor"] == pytest.approx(PUDONG.factor)
    assert written["close"] == pytest.approx(PUDONG_RAW_1009 * PUDONG.factor, abs=1e-4)
    move = adjusted_return(PUDONG, float(written["close"]))
    assert abs(move) < 0.05, f"the written row still jumps {move:.2%}"

    # The old behaviour, kept as the reference the assertion above rules out.
    old_move = PUDONG_RAW_1009 / PUDONG.close - 1
    assert old_move > 0.40


def test_a_new_symbol_starts_its_own_series(tmp_path):
    root = _provider(tmp_path, {}, ["2026-10-08"])
    rows = {"SH600999": _row(10.0, 10.2, 9.9, 10.1, 10.0)}

    out = _apply_store_basis(rows, "2026-10-09", root)

    assert out["SH600999"].iloc[0]["factor"] == 1.0
    assert out["SH600999"].iloc[0]["close"] == 10.1


def test_a_rerun_writes_nothing_for_a_stored_day(tmp_path):
    """The day is already in the store; returning the raw row would rewrite it."""
    root = _provider(tmp_path, {"SH600000": (6.4837, PUDONG.factor)}, ["2026-10-09"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, 9.54, 9.70)}

    out = _apply_store_basis(rows, "2026-10-09", root)

    assert out == {}, "a stored day must not be handed back to the writer"


# --- the guard --------------------------------------------------------------


def test_systemic_break_flags_a_market_wide_fault():
    returns = {f"SH{600000 + i}": 0.45 for i in range(100)}
    assert len(systemic_break(returns)) == 100


def test_systemic_break_tolerates_a_single_outlier():
    returns = {f"SH{600000 + i}": 0.01 for i in range(5000)}
    returns["SH600000"] = 0.8
    assert systemic_break(returns) == []


def test_systemic_break_ignores_samples_too_small_to_judge():
    assert systemic_break({"SH600000": 0.9, "SH600001": 0.9}) == []


def test_a_broken_batch_is_refused_before_it_is_written(tmp_path):
    symbols = {f"SH{600000 + i:06d}": (10.0, 0.68) for i in range(100)}
    root = _provider(tmp_path, symbols, ["2026-10-08"])
    rows = {
        sym: _row(10.0, 10.0, 10.0, 20.0, 10.0) for sym in symbols
    }

    with pytest.raises(RuntimeError, match="refusing to write"):
        _apply_store_basis(rows, "2026-10-09", root)

# --- the closed paths the first review asked for ----------------------------


def test_a_nan_pre_close_is_refused_not_written(tmp_path):
    root = _provider(tmp_path, {"SH600000": (PUDONG.close, PUDONG.factor)}, ["2026-10-08"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, 9.54, float("nan"))}

    # The message names the input: the factor check downstream would also catch
    # a NaN, and a test that accepts either would not notice the first guard
    # being removed.
    with pytest.raises(ValueError, match="pre_close must be finite"):
        _apply_store_basis(rows, "2026-10-09", root)


def test_unreadable_history_is_refused_not_treated_as_new(tmp_path):
    """Files exist but hold nothing usable: guessing factor 1.0 is the defect."""
    root = tmp_path / "cn_data"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2026-10-08\n", encoding="utf-8")
    _write_bin(root / "features" / "sh600000" / "close.day.bin", 0, [float("nan")])
    _write_bin(root / "features" / "sh600000" / "factor.day.bin", 0, [PUDONG.factor])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, 9.54, 9.70)}

    with pytest.raises(RuntimeError, match="no readable row"):
        _apply_store_basis(rows, "2026-10-09", root)


def test_a_truncated_bin_is_not_read_as_a_price(tmp_path):
    root = tmp_path / "cn_data"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2026-10-08\n", encoding="utf-8")
    field = root / "features" / "sh600000" / "close.day.bin"
    field.parent.mkdir(parents=True)
    field.write_bytes(struct.pack("<f", 0.0) + struct.pack("<f", 6.59) + b"\x00\x00")
    _write_bin(root / "features" / "sh600000" / "factor.day.bin", 0, [PUDONG.factor])

    from ashare_lab.data.store_state import last_state

    assert last_state(root, "SH600000") is None


def test_last_state_honours_a_non_zero_first_slot(tmp_path):
    root = tmp_path / "cn_data"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text(
        "\n".join(["2026-09-01", "2026-09-02", "2026-09-03"]) + "\n", encoding="utf-8"
    )
    _write_bin(root / "features" / "sh601318" / "close.day.bin", 2, [3.71])
    _write_bin(root / "features" / "sh601318" / "factor.day.bin", 2, [0.0704])

    from ashare_lab.data.store_state import last_state

    state, date = last_state(root, "SH601318")
    assert date == "2026-09-03"
    assert state.close == pytest.approx(3.71, abs=1e-6)


def test_field_drift_pairs_on_the_shared_slot(tmp_path):
    """A short factor file must not be paired with a later close value."""
    root = tmp_path / "cn_data"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2026-10-08\n2026-10-09\n", encoding="utf-8")
    _write_bin(root / "features" / "sh600000" / "close.day.bin", 0, [6.59, 6.90])
    _write_bin(root / "features" / "sh600000" / "factor.day.bin", 0, [0.679638])

    from ashare_lab.data.store_state import last_state

    state, date = last_state(root, "SH600000")
    assert date == "2026-10-08"
    assert state.close == pytest.approx(6.59, abs=1e-6)


def test_every_price_field_moves_with_the_factor(tmp_path):
    root = _provider(tmp_path, {"SH600000": (PUDONG.close, PUDONG.factor)}, ["2026-10-08"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, 9.54, 9.70)}

    written = _apply_store_basis(rows, "2026-10-09", root)["SH600000"].iloc[0]

    for field, raw in (("open", 9.70), ("high", 9.74), ("low", 9.54), ("close", 9.54)):
        assert written[field] == pytest.approx(raw * PUDONG.factor, abs=1e-4), field
    assert written["high"] >= written["close"] >= written["low"]


def test_no_corporate_action_keeps_the_factor_bit_identical():
    """The paper pipeline records a corporate action when the factor differs.

    Rounding in a real pre_close moves the ratio by a few 1e-8 every day. Without
    the tolerance those no-action days read as corporate actions and the paper
    book records one daily for every position it holds.
    """
    assert next_factor(PUDONG, PUDONG.raw_close) == PUDONG.factor
    nudged = PUDONG.raw_close * (1 + 3e-8)
    assert next_factor(PUDONG, nudged) == PUDONG.factor
    # A real action, two orders of magnitude larger, must still move it.
    assert next_factor(PUDONG, PUDONG.raw_close * 0.99) != PUDONG.factor


def test_guard_thresholds_are_pinned():
    from ashare_lab.data.adjust import (
        BREAK_RETURN,
        BREAK_SHARE,
        MIN_SYMBOLS_FOR_BREAK,
    )

    assert BREAK_RETURN == 0.30
    assert BREAK_SHARE == 0.05
    assert MIN_SYMBOLS_FOR_BREAK == 50

    # 4% of the market at 40%: under the share threshold, so it passes.
    soft = {f"SH{600000 + i}": 0.40 if i < 200 else 0.01 for i in range(5000)}
    assert systemic_break(soft) == []
    # 6% at 40%: a broken write.
    hard = {f"SH{600000 + i}": 0.40 if i < 300 else 0.01 for i in range(5000)}
    assert len(systemic_break(hard)) == 300
    # One name cannot trip it however extreme.
    assert systemic_break({"SH600000": 5.0, **{f"SZ{i:06d}": 0.0 for i in range(4999)}}) == []


# --- holes and re-runs (second review round) --------------------------------


def test_a_gap_behind_today_is_refused_not_absorbed(tmp_path):
    """Three sessions of market move must not be folded into the factor."""
    root = _provider(tmp_path, {"SH600000": (10.00 * 0.5, 0.5)}, ["2026-11-02"])
    cal = root / "calendars" / "day.txt"
    cal.write_text("2026-11-02\n2026-11-03\n2026-11-04\n2026-11-05\n2026-11-06\n", encoding="utf-8")
    rows = {"SH600000": _row(11.05, 11.05, 11.05, 11.05, 11.00)}

    with pytest.raises(RuntimeError, match="lag the calendar"):
        _apply_store_basis(rows, "2026-11-06", root)


def test_a_suspended_stock_resuming_is_not_a_gap(tmp_path):
    """pre_close is the last price it traded, so the ratio is one."""
    root = _provider(tmp_path, {"SH600000": (10.00 * 0.5, 0.5)}, ["2026-11-02"])
    cal = root / "calendars" / "day.txt"
    cal.write_text("2026-11-02\n2026-11-03\n2026-11-04\n2026-11-05\n2026-11-06\n", encoding="utf-8")
    rows = {"SH600000": _row(10.05, 10.05, 10.05, 10.05, 10.00)}

    written = _apply_store_basis(rows, "2026-11-06", root)["SH600000"].iloc[0]

    assert written["factor"] == pytest.approx(0.5, abs=1e-9)
    assert written["close"] == pytest.approx(5.025, abs=1e-4)


def test_a_day_already_stored_for_every_symbol_returns_nothing(tmp_path):
    """A re-run must be a no-op the caller can skip, not an empty write."""
    root = _provider(tmp_path, {"SH600000": (6.4837, PUDONG.factor)}, ["2026-10-09"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, 9.54, 9.70)}

    assert _apply_store_basis(rows, "2026-10-09", root) == {}


def test_a_nan_bin_header_is_not_a_date(tmp_path):
    root = tmp_path / "cn_data"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2026-10-08\n", encoding="utf-8")
    for field, value in (("close", 6.59), ("factor", 0.68)):
        path = root / "features" / "sh600000" / f"{field}.day.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<f", float("nan")) + struct.pack("<f", value))

    from ashare_lab.data.store_state import last_state

    assert last_state(root, "SH600000") is None


# --- round three: the guard must fire on the path production takes -----------


def test_a_store_calendar_behind_the_exchange_is_refused(tmp_path):
    """On a first run the date is not in the calendar yet; the guard must still fire."""
    root = _provider(tmp_path, {"SH600000": (10.00 * 0.5, 0.5)}, ["2026-11-04"])
    rows = {"SH600000": _row(10.10, 10.10, 10.10, 10.10, 10.05)}

    with pytest.raises(RuntimeError, match="backfill the hole before appending"):
        _apply_store_basis(rows, "2026-11-06", root)


def test_a_symbol_suspended_across_sessions_is_still_written(tmp_path):
    root = _provider(tmp_path, {"SH600000": (10.00 * 0.5, 0.5)}, ["2026-11-02"])
    cal = root / "calendars" / "day.txt"
    cal.write_text("2026-11-02\n2026-11-03\n2026-11-04\n2026-11-05\n", encoding="utf-8")
    rows = {"SH600000": _row(10.05, 10.05, 10.05, 10.05, 10.00)}

    written = _apply_store_basis(rows, "2026-11-05", root)["SH600000"].iloc[0]

    assert written["factor"] == pytest.approx(0.5, abs=1e-9)


def test_an_implausible_corporate_action_is_refused():
    from ashare_lab.data.adjust import next_factor

    with pytest.raises(ValueError, match="implausible"):
        next_factor(PUDONG, PUDONG.raw_close * 0.4)


def test_a_nan_price_is_refused_before_it_reaches_the_bin(tmp_path):
    root = _provider(tmp_path, {"SH600000": (PUDONG.close, PUDONG.factor)}, ["2026-10-08"])
    rows = {"SH600000": _row(9.70, 9.74, 9.54, float("nan"), 9.70)}

    with pytest.raises(ValueError, match="not a usable price"):
        _apply_store_basis(rows, "2026-10-09", root)
