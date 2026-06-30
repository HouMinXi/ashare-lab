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
    """4000 stocks logs warning but does NOT raise."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(4000)
    result = validate_daily_data(data, "2026-06-26")
    completeness_warns = [w for w in result.warnings if "completeness" in w.lower()]
    assert len(completeness_warns) > 0
    # Validation never raises
    assert isinstance(result.passed, bool)


def test_nan_in_ohlcv_flagged():
    """NaN in OHLCV columns is flagged."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(5500)
    data["sz000001"] = _make_stock_df(close=float("nan"))
    result = validate_daily_data(data, "2026-06-26")
    nan_warns = [w for w in result.warnings if "nan" in w.lower()]
    assert len(nan_warns) > 0


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
    """Validation with multiple problems returns result, never raises."""
    from ashare_lab.data.validator import validate_daily_data

    data = _make_data(3000)  # low count
    data["sz000001"] = _make_stock_df(volume=0, close=float("nan"))
    result = validate_daily_data(data, "2026-06-26")
    assert len(result.warnings) > 0
    assert isinstance(result.passed, bool)
