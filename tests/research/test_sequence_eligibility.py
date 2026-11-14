"""Window admission uses a caller-supplied calendar and each stock's own rows."""

from typing import cast

import numpy as np
import pandas as pd
import pytest


CATEGORIES = ["eligible", "insufficient-own-history", "calendar-gap"]


def _call(entry, index, calendar):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    if entry == "structure":
        return classify_sequence_windows(index, calendar, 3)
    assert entry == "maturity"
    return label_maturity_mask(index, calendar, horizon=2, cutoff=cast(pd.Timestamp, pd.Timestamp("2020-02-01")))


def _assert_mask(mask, index, expected):
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.dtype(bool)
    assert mask.ndim == 1
    assert mask.shape == (len(index),)
    assert mask.tolist() == expected
    selected = [i for i, valid in enumerate(expected) if valid]
    assert index[mask].equals(index.take(selected))


def test_labels_require_the_whole_future_horizon_at_inclusive_cutoff():
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.bdate_range("2020-01-01", periods=9)
    index = pd.MultiIndex.from_product([["A"], calendar[2:5]], names=["instrument", "datetime"])
    for position, expected in [(4, [True, False, False]), (5, [True, True, False])]:
        mask = label_maturity_mask(index, calendar, horizon=2, cutoff=cast(pd.Timestamp, calendar[position]))
        _assert_mask(mask, index, expected)


@pytest.mark.parametrize("horizon", [0, -1, True, np.bool_(False), 2.0, "2", None, np.nan])
def test_invalid_horizon_is_rejected(horizon):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A"], calendar[:2]], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="horizon"):
        label_maturity_mask(index, calendar, horizon=horizon, cutoff=cast(pd.Timestamp, calendar[-1]))


@pytest.mark.parametrize(
    "cutoff", [None, pd.NaT, "2020-01-06", 0, pd.Timestamp("2020-01-06", tz="UTC"), pd.Timestamp("2020-01-06 00:00:01")]
)
def test_invalid_cutoff_is_rejected(cutoff):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A"], calendar[:2]], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="cutoff"):
        label_maturity_mask(index, calendar, horizon=2, cutoff=cutoff)


@pytest.mark.parametrize("horizon", [1, 2, np.int64(2), np.uint64(2), 2**63 - 1, 2**100])
@pytest.mark.parametrize("cutoff", ["2019-12-31", "2020-01-03", "2020-01-04", "2020-01-05", "2030-01-01"])
def test_maturity_respects_calendar_coverage_and_cutoff(horizon, cutoff):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    cutoff = cast(pd.Timestamp, pd.Timestamp(cutoff))
    h = int(horizon)
    expected = [pos + h < len(calendar) and calendar[pos + h] <= cutoff for pos in range(len(calendar))]
    _assert_mask(label_maturity_mask(index, calendar, horizon=horizon, cutoff=cutoff), index, expected)


@pytest.mark.parametrize("empty_calendar", [False, True])
def test_maturity_empty_result_is_boolean(empty_calendar):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.DatetimeIndex([]) if empty_calendar else pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["instrument", "datetime"]
    )
    _assert_mask(
        label_maturity_mask(index, calendar, horizon=2, cutoff=cast(pd.Timestamp, pd.Timestamp("2020-01-10"))),
        index,
        [],
    )


def _assert_reasons(actual, index, expected):
    assert isinstance(actual, pd.Series)
    assert isinstance(actual.dtype, pd.CategoricalDtype)
    assert actual.cat.categories.tolist() == CATEGORIES
    assert actual.cat.ordered is False
    assert actual.shape == (len(index),)
    assert not actual.isna().any()
    assert actual.index.equals(index)
    assert actual.name == "window_reason"
    assert actual.tolist() == expected


def test_each_stock_needs_its_own_consecutive_history():
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    calendar = pd.bdate_range("2020-01-01", periods=9)
    index = pd.MultiIndex.from_tuples(
        [("A", date) for date in calendar[:5]] + [("B", date) for date in calendar[[0, 1, 4, 5, 6]]],
        names=["instrument", "datetime"],
    )
    expected = [
        "insufficient-own-history",
        "insufficient-own-history",
        "eligible",
        "eligible",
        "eligible",
        "insufficient-own-history",
        "insufficient-own-history",
        "calendar-gap",
        "calendar-gap",
        "eligible",
    ]
    _assert_reasons(classify_sequence_windows(index, calendar, 3), index, expected)


@pytest.mark.parametrize("length", [0, -1, True, np.bool_(True), 3.0, "3", None, np.nan])
def test_invalid_sequence_length_is_rejected(length):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    calendar = pd.bdate_range("2020-01-01", periods=3)
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="seq_len"):
        classify_sequence_windows(index, calendar, length)


@pytest.mark.parametrize("length", [1, 3, np.int64(3), np.uint64(3), 4, 2**100])
def test_window_length_boundaries(length):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    calendar = pd.bdate_range("2020-01-01", periods=4)
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    expected = ["eligible" if count >= length else "insufficient-own-history" for count in range(1, 5)]
    _assert_reasons(classify_sequence_windows(index, calendar, length), index, expected)


@pytest.mark.parametrize("empty_calendar", [False, True])
def test_empty_typed_index_retains_reason_categories(empty_calendar):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    calendar = pd.DatetimeIndex([]) if empty_calendar else pd.bdate_range("2020-01-01", periods=4)
    index = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["instrument", "datetime"]
    )
    _assert_reasons(classify_sequence_windows(index, calendar, 3), index, [])


@pytest.mark.parametrize("kind", ["none", "list", "object", "duplicate", "reverse", "nat", "tz", "intraday", "empty"])
@pytest.mark.parametrize("entry", ["structure", "maturity"])
def test_invalid_calendar_is_rejected(kind, entry):

    good = pd.bdate_range("2020-01-01", periods=3)
    index = pd.MultiIndex.from_product([["A"], good], names=["instrument", "datetime"])
    bad = {
        "none": None,
        "list": list(good),
        "object": good.astype(str),
        "duplicate": good.append(good[-1:]),
        "reverse": good[::-1],
        "nat": good.append(pd.DatetimeIndex([pd.NaT])),
        "tz": good.tz_localize("UTC"),
        "intraday": good + pd.Timedelta(hours=1),
        "empty": pd.DatetimeIndex([]),
    }[kind]
    with pytest.raises(ValueError, match="calendar"):
        _call(entry, index, bad)


@pytest.mark.parametrize(
    "kind",
    [
        "none",
        "flat",
        "wrong-names",
        "reversed-levels",
        "extra-level",
        "duplicate",
        "reverse",
        "missing-instrument",
        "empty-instrument",
        "numeric-instrument",
        "string-date",
        "missing-date",
        "tz",
        "intraday",
        "absent-date",
        "malformed-empty",
    ],
)
@pytest.mark.parametrize("entry", ["structure", "maturity"])
def test_invalid_index_is_rejected(kind, entry):

    calendar = pd.bdate_range("2020-01-01", periods=3)
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    if kind in {"none", "flat", "wrong-names", "reversed-levels", "extra-level", "duplicate", "reverse"}:
        index = {
            "none": None,
            "flat": pd.Index(["A"]),
            "wrong-names": index.set_names(["stock", "day"]),
            "reversed-levels": index.swaplevel(),
            "extra-level": pd.MultiIndex.from_product([["A"], calendar, [1]]),
            "duplicate": index.append(index[-1:]),
            "reverse": index[::-1],
        }[kind]
    else:
        instrument, dates = {
            "missing-instrument": ([None], calendar[:1]),
            "empty-instrument": ([""], calendar[:1]),
            "numeric-instrument": ([42], calendar[:1]),
            "string-date": (["A"] * 3, calendar.astype(str)),
            "missing-date": (["A"], pd.DatetimeIndex([pd.NaT])),
            "tz": (["A"] * 3, calendar.tz_localize("UTC")),
            "intraday": (["A"] * 3, calendar + pd.Timedelta(hours=1)),
            "absent-date": (["A"], pd.DatetimeIndex(["2019-12-31"])),
            "malformed-empty": ([], pd.Index([], dtype=object)),
        }[kind]
        index = pd.MultiIndex.from_arrays([instrument, dates], names=["instrument", "datetime"])
    with pytest.raises(ValueError):
        _call(entry, index, calendar)


@pytest.mark.parametrize("entry", ["structure", "maturity"])
def test_matching_intraday_index_and_calendar_still_reject(entry):
    calendar = pd.bdate_range("2020-01-01", periods=5) + pd.Timedelta(hours=1)
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="midnight"):
        _call(entry, index, calendar)


def test_unobserved_market_session_is_still_a_hole():
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    calendar = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A", "B"], calendar[[0, 1, 3, 4]]], names=["instrument", "datetime"])
    expected = ["insufficient-own-history"] * 2 + ["calendar-gap"] * 2
    _assert_reasons(classify_sequence_windows(index, calendar, 3), index, expected * 2)


@pytest.mark.parametrize("entry", ["structure", "maturity"])
@pytest.mark.parametrize("reject", [False, True])
def test_inputs_remain_unchanged(entry, reject):
    calendar = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A", "B"], calendar], names=["instrument", "datetime"])
    if reject:
        index = index.append(index[-1:])
    index_before = index.copy(deep=True)
    calendar_before = calendar.copy(deep=True)
    if reject:
        with pytest.raises(ValueError):
            _call(entry, index, calendar)
    else:
        result = _call(entry, index, calendar)
        if entry == "structure":
            _assert_reasons(
                result, index, [CATEGORIES[1]] * 2 + [CATEGORIES[0]] * 3 + [CATEGORIES[1]] * 2 + [CATEGORIES[0]] * 3
            )
        else:
            _assert_mask(result, index, [True, True, True, False, False] * 2)
    pd.testing.assert_index_equal(index, index_before, exact=True)
    pd.testing.assert_index_equal(calendar, calendar_before, exact=True)


@pytest.mark.parametrize("entry", ["structure", "maturity"])
def test_malformed_empty_input_is_not_an_early_success(entry):
    index = pd.MultiIndex.from_arrays([[], pd.DatetimeIndex([])], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="calendar"):
        _call(entry, index, pd.Index([], dtype=object))


def _scalar_reasons(index, calendar, length):
    own = {}
    result = []
    for instrument, date in index:
        history = own.setdefault(instrument, [])
        history.append(date)
        if len(history) < length:
            result.append("insufficient-own-history")
        else:
            end = calendar.get_loc(date)
            expected = list(calendar[max(0, end - length + 1) : end + 1])
            result.append("eligible" if history[-length:] == expected else "calendar-gap")
    return result


@pytest.mark.parametrize("case", range(100))
@pytest.mark.parametrize("length", [1, 3, 20])
def test_windows_match_independent_calendar_slice_oracle(case, length):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows

    rng = np.random.default_rng(np.random.SeedSequence(20261112, spawn_key=(case,)))
    calendar = pd.bdate_range("2020-01-01", periods=30)
    instruments, dates = [], []
    for stock in range(int(rng.integers(1, 5))):
        positions = np.sort(rng.choice(len(calendar), size=int(rng.integers(1, 31)), replace=False))
        selected = calendar[positions]
        instruments.extend([str(stock)] * len(selected))
        dates.extend(selected)
    index = pd.MultiIndex.from_arrays([instruments, pd.DatetimeIndex(dates)], names=["instrument", "datetime"])
    expected = _scalar_reasons(index, calendar, length)
    _assert_reasons(classify_sequence_windows(index, calendar, length), index, expected)


@pytest.mark.parametrize("length", [1, np.int64(1), 2, 2**100])
def test_single_endpoint(length):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    calendar = pd.DatetimeIndex(["2020-01-03"])
    index = pd.MultiIndex.from_product([["A"], calendar], names=["instrument", "datetime"])
    expected = ["eligible"] if length == 1 else ["insufficient-own-history"]
    _assert_reasons(classify_sequence_windows(index, calendar, length), index, expected)
    _assert_mask(
        label_maturity_mask(index, calendar, horizon=length, cutoff=cast(pd.Timestamp, pd.Timestamp("2021-01-01"))),
        index,
        [False],
    )


@pytest.mark.parametrize("date,expected", [("1500-01-01", [False, False]), ("2500-01-01", [True, False])])
@pytest.mark.parametrize("calendar_unit", ["s", "ms", "us", "ns"])
def test_maturity_accepts_cutoffs_outside_nanosecond_range(date, expected, calendar_unit):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.date_range("2020-01-01", periods=2).as_unit(calendar_unit)
    index = pd.MultiIndex.from_arrays([["A", "A"], calendar], names=["instrument", "datetime"])
    cutoff = cast(pd.Timestamp, pd.Timestamp(date))
    _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, expected)


@pytest.mark.parametrize("date", ["1500-01-01", "2500-01-01"])
@pytest.mark.parametrize("empty_calendar", [False, True])
def test_maturity_empty_inputs_accept_wide_cutoffs(date, empty_calendar):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.DatetimeIndex([]) if empty_calendar else pd.date_range("2020-01-01", periods=2)
    index = pd.MultiIndex.from_arrays([[], pd.DatetimeIndex([])], names=["instrument", "datetime"])
    cutoff = cast(pd.Timestamp, pd.Timestamp(date))
    _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, [])


@pytest.mark.parametrize("year", ["0001", "1500", "1677", "2263", "9999"])
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_maturity_preserves_wide_calendar_resolution(year, unit):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    calendar = pd.DatetimeIndex(
        np.array([f"{year}-01-01", f"{year}-01-02", f"{year}-01-03"], dtype=f"datetime64[{unit}]")
    )
    index = pd.MultiIndex.from_arrays([["A"] * len(calendar), calendar], names=["instrument", "datetime"])
    cutoff = cast(pd.Timestamp, calendar[1])
    _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, [True, False, False])


@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("helper", ["classify", "maturity"])
def test_duration_cannot_be_a_session_count(unit, helper):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    calendar = pd.date_range("2020-01-01", periods=3)
    index = pd.MultiIndex.from_arrays([["A"] * 3, calendar], names=["instrument", "datetime"])
    value = cast(int, np.timedelta64(2, unit))
    with pytest.raises(ValueError):
        if helper == "classify":
            classify_sequence_windows(index, calendar, value)
        else:
            label_maturity_mask(index, calendar, horizon=value, cutoff=cast(pd.Timestamp, calendar[-1]))


@pytest.mark.parametrize("calendar_unit", ["s", "ms", "us"])
@pytest.mark.parametrize("endpoint_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("helper", ["classify", "maturity"])
def test_wide_calendar_preserves_narrow_endpoint_membership(calendar_unit, endpoint_unit, helper):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    calendar = pd.DatetimeIndex(
        np.array(["1500-01-01", "2020-01-01", "2020-01-02", "2500-01-01"], dtype=f"datetime64[{calendar_unit}]")
    )
    dates = calendar[1:3].as_unit(endpoint_unit)
    index = pd.MultiIndex.from_arrays([["A", "A"], dates], names=["instrument", "datetime"])
    if helper == "classify":
        _assert_reasons(classify_sequence_windows(index, calendar, 2), index, ["insufficient-own-history", "eligible"])
    else:
        cutoff = cast(pd.Timestamp, pd.Timestamp("2500-01-01"))
        _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, [True, True])


@pytest.mark.parametrize("unit,scale", [("s", 86400), ("ms", 86400000), ("us", 86400000000), ("ns", 86400000000000)])
@pytest.mark.parametrize("side", ["first", "last"])
@pytest.mark.parametrize("helper", ["classify", "maturity"])
def test_first_and_last_representable_midnight_sessions(unit, scale, side, helper):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    last_day = (2**63 - 1) // scale
    start_day = -last_day if side == "first" else last_day - 2
    ticks = np.array([(start_day + offset) * scale for offset in range(3)], dtype=np.int64)
    calendar = pd.DatetimeIndex(ticks.view(f"datetime64[{unit}]"))
    index = pd.MultiIndex.from_arrays([["A"] * 3, calendar], names=["instrument", "datetime"])
    if helper == "classify":
        _assert_reasons(
            classify_sequence_windows(index, calendar, 2), index, ["insufficient-own-history", "eligible", "eligible"]
        )
    else:
        cutoff = cast(pd.Timestamp, calendar[1])
        _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, [True, False, False])


@pytest.mark.parametrize("unit,scale", [("s", 86400), ("ms", 86400000), ("us", 86400000000), ("ns", 86400000000000)])
@pytest.mark.parametrize("side,expected", [("first", [False, False]), ("last", [True, False])])
def test_extreme_cutoff_keeps_its_chronological_position(unit, scale, side, expected):
    from ashare_lab.research.sequence_eligibility import label_maturity_mask

    day = (2**63 - 1) // scale
    tick = (-day if side == "first" else day) * scale
    stamp = cast(np.datetime64, np.array(tick, dtype=np.int64).view(f"datetime64[{unit}]")[()])
    cutoff = cast(pd.Timestamp, pd.Timestamp(stamp))
    calendar = pd.date_range("2020-01-01", periods=2)
    index = pd.MultiIndex.from_arrays([["A", "A"], calendar], names=["instrument", "datetime"])
    assert (cutoff < calendar[0]) if side == "first" else (cutoff > calendar[-1])
    _assert_mask(label_maturity_mask(index, calendar, horizon=1, cutoff=cutoff), index, expected)


@pytest.mark.parametrize("name", [pd.NA, None, np.nan, pd.NaT, 0, ("instrument",), b"datetime"])
@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("helper", ["classify", "maturity"])
def test_nonstring_level_names_use_value_error(name, position, helper):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    calendar = pd.date_range("2020-01-01", periods=3)
    names = ["instrument", "datetime"]
    names[position] = name
    index = pd.MultiIndex.from_product([["A"], calendar], names=names)
    with pytest.raises(ValueError):
        if helper == "classify":
            classify_sequence_windows(index, calendar, 2)
        else:
            label_maturity_mask(index, calendar, horizon=1, cutoff=cast(pd.Timestamp, calendar[-1]))


@pytest.mark.parametrize("names", [["instrument", "datetime"], [np.str_("instrument"), np.str_("datetime")]])
def test_valid_string_level_names_remain_accepted(names):
    from ashare_lab.research.sequence_eligibility import classify_sequence_windows, label_maturity_mask

    calendar = pd.date_range("2020-01-01", periods=3)
    index = pd.MultiIndex.from_product([["A"], calendar], names=names)
    _assert_reasons(
        classify_sequence_windows(index, calendar, 2), index, ["insufficient-own-history", "eligible", "eligible"]
    )
    result = label_maturity_mask(index, calendar, horizon=1, cutoff=cast(pd.Timestamp, calendar[-1]))
    _assert_mask(result, index, [True, True, False])
