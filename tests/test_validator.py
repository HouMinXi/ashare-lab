"""Tests for ashare_lab.data.validator -- three-layer data validation."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_stock_df(
    date: str = "2026-06-26",
    open_: float = 10.0,
    high: float = 10.5,
    low: float = 9.5,
    close: float = 10.2,
    volume: float = 100000.0,
    factor: float = 1.0,
    change: float = 0.02,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "factor": factor,
                "change": change,
            }
        ]
    )


def _make_data(n: int, date: str = "2026-06-26") -> dict[str, pd.DataFrame]:
    """Generate n fake stock entries with valid data."""
    data = {}
    for i in range(n):
        code = f"{i:06d}"
        if i % 3 == 0:
            sym = f"sh6{code[1:]}"
        elif i % 3 == 1:
            sym = f"sz0{code[1:]}"
        else:
            sym = f"sz3{code[1:]}"
        data[sym] = _make_stock_df(date=date, change=0.01)
    return data


# ---------------------------------------------------------------------------
# Layer 1: Completeness
# ---------------------------------------------------------------------------


def test_completeness_pass():
    """5500 stocks passes completeness check."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    result = validate_daily_data(data, "2026-06-26")
    completeness_warns = [w for w in result.warnings if "completeness" in w.lower()]
    assert completeness_warns == []
    assert result.passed is True


def test_completeness_warn():
    """4000 stocks logs completeness warning and result.passed is False."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(4000)
    result = validate_daily_data(data, "2026-06-26")
    completeness_warns = [w for w in result.warnings if "completeness" in w.lower()]
    assert len(completeness_warns) > 0
    assert result.passed is False


@pytest.mark.parametrize("col,kwargs", [
    ("open",   {"open_": float("nan")}),
    ("high",   {"high": float("nan")}),
    ("low",    {"low": float("nan")}),
    ("close",  {"close": float("nan")}),
    ("volume", {"volume": float("nan")}),
])
def test_nan_in_ohlcv_flagged(col, kwargs):
    """NaN in any of the five OHLCV columns is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000001"] = _make_stock_df(**kwargs)
    result = validate_daily_data(data, "2026-06-26")
    nan_warns = [w for w in result.warnings if "nan" in w.lower()]
    assert len(nan_warns) > 0, f"expected NaN warning for column '{col}'"


def test_nan_in_volume_skips_zero_check():
    """NaN volume: the volume==0 check is skipped silently (no 'volume' warning).

    The validator guards the zero-volume check with:
        if not (isinstance(vol, float) and math.isnan(vol)):
    so NaN volume produces a NaN-in-OHLCV warning (Layer 1) but NOT
    the 'volume is 0 (suspended)' warning (Layer 2).  This test documents
    that intentional gap so future readers do not mistake it for a bug.
    """
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000001"] = _make_stock_df(volume=float("nan"))
    result = validate_daily_data(data, "2026-06-26")
    # Layer 1 must fire: NaN detected in 'volume' column
    nan_warns = [w for w in result.warnings if "nan" in w.lower() and "sz000001" in w]
    assert len(nan_warns) > 0, "expected NaN-in-volume Layer-1 warning"
    # Layer 2 must NOT fire: 'volume is 0 (suspended)' check is bypassed for NaN
    zero_warns = [w for w in result.warnings if "sz000001" in w and "suspended" in w.lower()]
    assert len(zero_warns) == 0, f"unexpected zero-volume warning for NaN volume: {zero_warns}"


# ---------------------------------------------------------------------------
# Layer 2: Price sanity
# ---------------------------------------------------------------------------


def test_st_stock_over_5pct_flagged():
    """ST stock with change > 5% is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000999"] = _make_stock_df(change=0.06)
    names = {"sz000999": "*ST Example"}
    result = validate_daily_data(data, "2026-06-26", stock_names=names)
    st_warns = [w for w in result.warnings if "sz000999" in w]
    assert len(st_warns) > 0


def test_normal_mainboard_over_10pct_flagged():
    """Normal mainboard stock with change > 10% is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sh600999"] = _make_stock_df(change=0.11)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sh600999" in w]
    assert len(warns) > 0


def test_gem_star_over_20pct_flagged():
    """GEM (300xxx) with change > 20% is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz300999"] = _make_stock_df(change=0.21)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sz300999" in w]
    assert len(warns) > 0


def test_star_688_over_20pct_flagged():
    """STAR (688xxx) with change > 20% is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sh688999"] = _make_stock_df(change=0.21)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sh688999" in w]
    assert len(warns) > 0


def test_volume_zero_flagged():
    """volume == 0 is flagged (suspended stock)."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000888"] = _make_stock_df(volume=0.0)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sz000888" in w and "volume" in w.lower()]
    assert len(warns) > 0


def test_close_outside_range_flagged():
    """close outside [low, high] is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    # close > high
    data["sz000777"] = _make_stock_df(close=11.0, high=10.5, low=9.5)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sz000777" in w]
    assert len(warns) > 0


# ---------------------------------------------------------------------------
# Layer 3: Cross-validation
# ---------------------------------------------------------------------------


def test_cross_validation_diff_flagged():
    """Close diff > 0.01 between tushare and AKShare is logged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000002"] = _make_stock_df(close=10.20)
    cross = {"sz000002": _make_stock_df(close=10.25)}

    result = validate_daily_data(data, "2026-06-26", cross_val_data=cross)
    cv_warns = [w for w in result.warnings if "cross" in w.lower() and "sz000002" in w]
    assert len(cv_warns) > 0


def test_cross_validation_empty_no_warns():
    """Empty cross-validation dict produces no cross-validation warnings."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    result = validate_daily_data(data, "2026-06-26", cross_val_data={})
    cv_warns = [w for w in result.warnings if "cross" in w.lower()]
    assert cv_warns == []


def test_cross_validation_none_no_warns():
    """None cross_val_data produces no cross-validation warnings."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    result = validate_daily_data(data, "2026-06-26", cross_val_data=None)
    cv_warns = [w for w in result.warnings if "cross" in w.lower()]
    assert cv_warns == []


# ---------------------------------------------------------------------------
# Overall behavior: never raise
# ---------------------------------------------------------------------------


def test_validation_never_raises():
    """Validation with multiple problems returns result with passed=False, never raises."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(3000)  # low count
    data["sz000001"] = _make_stock_df(volume=0, close=float("nan"))
    result = validate_daily_data(data, "2026-06-26")
    assert len(result.warnings) > 0
    assert result.passed is False




def test_st_name_exact_match_flagged():
    """Exact 'ST' and '*ST' names trigger the 5% limit, not the mainboard 10%."""
    from ashare_lab.data.validator import _change_limit

    assert _change_limit("sz000999", {"sz000999": "ST"}) == 0.05
    assert _change_limit("sz000998", {"sz000998": "*ST"}) == 0.05


def test_st_name_with_space_flagged():
    """'ST SomeName' (trailing space after ST) triggers 5% limit."""
    from ashare_lab.data.validator import _change_limit

    assert _change_limit("sz000997", {"sz000997": "ST PingAn"}) == 0.05


def test_st_star_prefix_flagged():
    """'*ST SomeName' triggers 5% limit."""
    from ashare_lab.data.validator import _change_limit

    assert _change_limit("sz000996", {"sz000996": "*ST Example"}) == 0.05


def test_st_false_positive_avoided():
    """Names containing 'ST' but not as a marker are not treated as ST stocks."""
    from ashare_lab.data.validator import _change_limit

    # 'STKN' contains 'ST' but is not an ST stock -- must use mainboard 10% limit
    result = _change_limit("sz000995", {"sz000995": "STKN Holdings"})
    assert result != 0.05, "STKN Holdings should not be flagged as ST"


def test_nan_in_change_column_skipped_gracefully():
    """NaN in 'change' column: limit check is skipped silently, no crash, no spurious warning."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    df = data["sz000001"].copy()
    df["change"] = float("nan")
    data["sz000001"] = df
    # Must not raise
    result = validate_daily_data(data, "2026-06-26")
    # NaN change does NOT produce a change-limit warning for sz000001
    limit_warns = [
        w for w in result.warnings if "sz000001" in w and "exceeds" in w
    ]
    assert len(limit_warns) == 0
    # Overall result should pass (only that one stock has NaN change, OHLCV is valid)
    assert result.passed is True


def test_negative_change_exceeds_mainboard_limit_flagged():
    """Negative change exceeding mainboard 10% limit is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    # -11% on mainboard -- should trigger the same limit check as +11%
    data["sh600999"] = _make_stock_df(change=-0.11)
    result = validate_daily_data(data, "2026-06-26")
    warns = [w for w in result.warnings if "sh600999" in w]
    assert len(warns) > 0
