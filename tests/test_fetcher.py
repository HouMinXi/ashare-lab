"""Tests for ashare_lab.data.fetcher -- tushare data fetch + column mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


def test_tushare_code_to_qlib_sz():
    from ashare_lab.data.fetcher import _tushare_code_to_qlib

    assert _tushare_code_to_qlib("000001.SZ") == "sz000001"


def test_tushare_code_to_qlib_sh():
    from ashare_lab.data.fetcher import _tushare_code_to_qlib

    assert _tushare_code_to_qlib("600000.SH") == "sh600000"


def test_tushare_code_to_qlib_bj():
    from ashare_lab.data.fetcher import _tushare_code_to_qlib

    assert _tushare_code_to_qlib("430047.BJ") == "bj430047"


# ---------------------------------------------------------------------------
# fetch_today_data column mapping
# ---------------------------------------------------------------------------


def test_fetch_today_data_column_mapping():
    """Tushare DataFrame maps to _CSV_COLUMNS with correct transforms."""
    from ashare_lab.data.fallback import _CSV_COLUMNS
    from ashare_lab.data.fetcher import fetch_today_data

    fake_df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260626",
                "open": 10.42,
                "high": 10.47,
                "low": 10.19,
                "close": 10.23,
                "pre_close": 10.42,
                "change": -0.19,
                "pct_chg": -1.8234,
                "vol": 1236481.64,
                "amount": 1270902.94786,
            },
            {
                "ts_code": "600000.SH",
                "trade_date": "20260626",
                "open": 8.50,
                "high": 8.60,
                "low": 8.40,
                "close": 8.55,
                "pre_close": 8.50,
                "change": 0.05,
                "pct_chg": 0.5882,
                "vol": 500000.0,
                "amount": 425000.0,
            },
        ]
    )

    mock_pro = MagicMock()
    mock_pro.daily.return_value = fake_df

    with (
        patch("ashare_lab.data.fetcher._get_tushare_token", return_value="fake"),
        patch("ashare_lab.data.fetcher.ts") as mock_ts,
    ):
        mock_ts.pro_api.return_value = mock_pro
        result = fetch_today_data("2026-06-26")

    assert "sz000001" in result
    assert "sh600000" in result

    df = result["sz000001"]
    assert list(df.columns) == _CSV_COLUMNS
    assert df.iloc[0]["date"] == "2026-06-26"
    assert df.iloc[0]["volume"] == 1236481.64
    assert df.iloc[0]["factor"] == 1.0
    assert abs(df.iloc[0]["change"] - (-0.018234)) < 1e-6


def test_fetch_today_data_compact_date_format():
    """Compact YYYYMMDD format is accepted and normalized to YYYY-MM-DD in output."""
    from ashare_lab.data.fetcher import fetch_today_data

    fake_df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260626",
                "open": 10.42,
                "high": 10.47,
                "low": 10.19,
                "close": 10.23,
                "pre_close": 10.42,
                "change": -0.19,
                "pct_chg": -1.8234,
                "vol": 1236481.64,
                "amount": 1270902.94,
            }
        ]
    )
    mock_pro = MagicMock()
    mock_pro.daily.return_value = fake_df

    with (
        patch("ashare_lab.data.fetcher._get_tushare_token", return_value="fake"),
        patch("ashare_lab.data.fetcher.ts") as mock_ts,
    ):
        mock_ts.pro_api.return_value = mock_pro
        result = fetch_today_data("20260626")

    assert "sz000001" in result
    assert result["sz000001"].iloc[0]["date"] == "2026-06-26"


def test_fetch_today_data_empty_raises():
    """Empty tushare response raises RuntimeError."""
    from ashare_lab.data.fetcher import fetch_today_data

    mock_pro = MagicMock()
    mock_pro.daily.return_value = pd.DataFrame()

    with (
        patch("ashare_lab.data.fetcher._get_tushare_token", return_value="fake"),
        patch("ashare_lab.data.fetcher.ts") as mock_ts,
    ):
        mock_ts.pro_api.return_value = mock_pro
        with pytest.raises(RuntimeError, match="empty"):
            fetch_today_data("2026-06-26")


def test_fetch_today_data_none_raises():
    """None tushare response raises RuntimeError."""
    from ashare_lab.data.fetcher import fetch_today_data

    mock_pro = MagicMock()
    mock_pro.daily.return_value = None

    with (
        patch("ashare_lab.data.fetcher._get_tushare_token", return_value="fake"),
        patch("ashare_lab.data.fetcher.ts") as mock_ts,
    ):
        mock_ts.pro_api.return_value = mock_pro
        with pytest.raises(RuntimeError, match="empty"):
            fetch_today_data("2026-06-26")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def test_cross_validation_mocked():
    """Cross-validation returns mapped data from mocked akshare."""
    from ashare_lab.data.fetcher import fetch_cross_validation_sample

    fake_ak_df = pd.DataFrame(
        [
            {
                "日期": "2026-06-26",
                "开盘": 10.42,
                "收盘": 10.23,
                "最高": 10.47,
                "最低": 10.19,
                "成交量": 1236481,
                "成交额": 1270902947.86,
                "振幅": 2.69,
                "涨跌幅": -1.8234,
                "涨跌额": -0.19,
                "换手率": 0.64,
            }
        ]
    )

    mock_ak = MagicMock()
    mock_ak.stock_zh_a_hist.return_value = fake_ak_df

    with patch.dict("sys.modules", {"akshare": mock_ak}):
        result = fetch_cross_validation_sample("2026-06-26", ["sz000001"])

    assert "sz000001" in result
    df = result["sz000001"]
    assert df.iloc[0]["close"] == 10.23
    assert df.iloc[0]["factor"] == 1.0


def test_cross_validation_import_error():
    """AKShare ImportError is caught, returns empty dict.

    Uses builtins.__import__ side_effect so the ImportError is explicit and
    deterministic, not reliant on CPython's None-in-sys.modules behaviour.
    """
    from ashare_lab.data.fetcher import fetch_cross_validation_sample

    _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import_raiser(name, *args, **kwargs):
        if name == "akshare":
            raise ImportError("No module named 'akshare'")
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_raiser):
        result = fetch_cross_validation_sample("2026-06-26", ["sz000001"])

    assert result == {}


def test_cross_validation_per_symbol_error():
    """Per-symbol failure is caught, other symbols still returned."""
    from ashare_lab.data.fetcher import fetch_cross_validation_sample

    fake_ak_df = pd.DataFrame(
        [
            {
                "日期": "2026-06-26",
                "开盘": 8.50,
                "收盘": 8.55,
                "最高": 8.60,
                "最低": 8.40,
                "成交量": 500000,
                "成交额": 425000000.0,
                "振庅": 2.35,
                "涨跌幅": 0.5882,
                "涨跌额": 0.05,
                "换手率": 0.30,
            }
        ]
    )

    mock_ak = MagicMock()

    def side_effect(symbol, **kwargs):
        if symbol == "000001":
            raise ConnectionError("network error")
        return fake_ak_df

    mock_ak.stock_zh_a_hist.side_effect = side_effect

    with patch.dict("sys.modules", {"akshare": mock_ak}):
        result = fetch_cross_validation_sample(
            "2026-06-26", ["sz000001", "sh600000"]
        )

    assert "sz000001" not in result
    assert "sh600000" in result


# ---------------------------------------------------------------------------
# Stock name cache
# ---------------------------------------------------------------------------


def test_stock_names_cache(tmp_path: Path):
    """refresh_stock_names_cache writes CSV and returns dict."""
    from ashare_lab.data.fetcher import refresh_stock_names_cache

    fake_basic_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "平安银行"},
            {"ts_code": "600000.SH", "name": "浦发银行"},
        ]
    )

    mock_pro = MagicMock()
    mock_pro.stock_basic.return_value = fake_basic_df

    cache_path = tmp_path / "stock_names_cache.csv"

    with (
        patch("ashare_lab.data.fetcher._get_tushare_token", return_value="fake"),
        patch("ashare_lab.data.fetcher.ts") as mock_ts,
    ):
        mock_ts.pro_api.return_value = mock_pro
        names = refresh_stock_names_cache(cache_path)

    assert names["sz000001"] == "平安银行"
    assert names["sh600000"] == "浦发银行"
    assert cache_path.exists()

    cached = pd.read_csv(cache_path)
    assert list(cached.columns) == ["symbol", "name"]
    assert len(cached) == 2


def _stub_fetch_today(monkeypatch, trade_date=None):
    """Stub everything cmd_fetch_today needs except the backfill under test."""
    import datetime as dt

    day = trade_date or dt.date(2024, 3, 5)
    monkeypatch.setattr('ashare_lab.data.calendar.is_trading_day', lambda d: True)
    monkeypatch.setattr(
        'ashare_lab.data.calendar.latest_trading_day', lambda d: day)
    monkeypatch.setattr(
        'ashare_lab.data.fetcher.fetch_today_data', lambda d: {'sz000001': object()})
    monkeypatch.setattr(
        'ashare_lab.data.fetcher.fetch_cross_validation_sample',
        lambda *a, **k: {})
    monkeypatch.setattr(
        'ashare_lab.data.fetcher.refresh_stock_names_cache', lambda *a, **k: {})
    monkeypatch.setattr(
        'ashare_lab.data.validator.validate_daily_data',
        lambda *a, **k: type('R', (), {'passed': True, 'warnings': []})())
    monkeypatch.setattr('ashare_lab.cli._write_csvs', lambda *a, **k: 1)
    monkeypatch.setattr('ashare_lab.cli._dump_bin_update', lambda *a, **k: None)
    return day


def test_fetch_today_backfills_gaps_after_writing_today(monkeypatch):
    """A gap behind the day just written gets filled, and a backfill
    failure does not turn the day's success into a failure."""
    import datetime as dt

    from ashare_lab.cli import cmd_fetch_today

    seen = {}

    def fake_backfill(start, end, provider_uri=None):
        seen['span'] = (start, end)
        raise RuntimeError('tushare down')

    _stub_fetch_today(monkeypatch)
    monkeypatch.setattr('ashare_lab.data.fallback.backfill_missing_dates', fake_backfill)
    monkeypatch.setattr(
        'ashare_lab.data.fallback._calendar_dates',
        lambda provider: {'2024-02-01', '2024-02-28'})

    rc = cmd_fetch_today(type('A', (), {})())

    assert rc == 0
    assert seen['span'] == (dt.date(2024, 2, 27), dt.date(2024, 3, 5))


def test_fetch_today_refreshes_stale_indexes(monkeypatch):
    """The daily fetch covers stocks only, so an index keeps an older tail
    until it is refreshed here. A refresh failure must not fail the day."""
    from ashare_lab.cli import cmd_fetch_today

    seen = {}

    def fake_index_backfill(trade_date, provider_uri=None):
        seen['date'] = trade_date
        raise RuntimeError('baostock down')

    _stub_fetch_today(monkeypatch)
    monkeypatch.setattr(
        'ashare_lab.data.fallback.backfill_missing_dates', lambda *a, **k: [])
    monkeypatch.setattr(
        'ashare_lab.data.fallback._calendar_dates', lambda provider: set())
    monkeypatch.setattr(
        'ashare_lab.data.fallback.backfill_stale_indexes', fake_index_backfill)

    rc = cmd_fetch_today(type('A', (), {})())

    assert rc == 0
    assert seen['date'] == '2024-03-05', 'the day just written is the target'
