"""Integration tests for realtime data pipeline wiring.

Verifies that fetcher, validator, and report components work together
through pipeline.py: stock name cache preference, Chinese report
format, and data flow correctness.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ashare_lab.paper.report import ReportData, format_chinese_report


# ---------------------------------------------------------------------------
# Stock name cache preference (pipeline.py _load_stock_names_cache)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStockNameCachePreference:
    """Verify tushare cache is preferred over baostock, with fallback."""

    def _patch_roots(self, tmp_path):
        """Patch both PROJECT_ROOT and _BS_CACHE_DIR (computed at import)."""
        return (
            patch("ashare_lab.paper.pipeline.PROJECT_ROOT", tmp_path),
            patch(
                "ashare_lab.paper.pipeline._BS_CACHE_DIR",
                tmp_path / "data" / "baostock_cache",
            ),
        )

    def test_prefers_tushare_cache(self, tmp_path: Path):
        ts_path = tmp_path / "data" / "stock_names_cache.csv"
        ts_path.parent.mkdir(parents=True)
        with open(ts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "name"])
            w.writerow(["sz000001", "PingAnBank"])
            w.writerow(["sh600000", "PuFaBank"])

        bs_dir = tmp_path / "data" / "baostock_cache"
        bs_dir.mkdir(parents=True)
        bs_path = bs_dir / "stock_names.csv"
        with open(bs_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "code_name"])
            w.writerow(["sz.000001", "OldPingAn"])

        symbols = {"SZ000001", "SH600000"}

        p1, p2 = self._patch_roots(tmp_path)
        with p1, p2:
            from ashare_lab.paper.pipeline import _load_stock_names_cache

            result = _load_stock_names_cache(symbols)

        assert result is not None
        assert result["SZ000001"] == "PingAnBank"
        assert result["SH600000"] == "PuFaBank"

    def test_falls_back_to_baostock(self, tmp_path: Path):
        bs_dir = tmp_path / "data" / "baostock_cache"
        bs_dir.mkdir(parents=True)
        bs_path = bs_dir / "stock_names.csv"
        with open(bs_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "code_name"])
            w.writerow(["sz.000001", "OldPingAn"])

        symbols = {"SZ000001"}

        p1, p2 = self._patch_roots(tmp_path)
        with p1, p2:
            from ashare_lab.paper.pipeline import _load_stock_names_cache

            result = _load_stock_names_cache(symbols)

        assert result is not None
        assert result["SZ000001"] == "OldPingAn"

    def test_returns_none_when_no_cache(self, tmp_path: Path):
        (tmp_path / "data").mkdir(parents=True)

        symbols = {"sz000001"}

        p1, p2 = self._patch_roots(tmp_path)
        with p1, p2:
            from ashare_lab.paper.pipeline import _load_stock_names_cache

            result = _load_stock_names_cache(symbols)

        assert result is None

    def test_empty_tushare_cache_falls_back(self, tmp_path: Path):
        """Tushare cache exists but has no matching symbols -> baostock."""
        ts_path = tmp_path / "data" / "stock_names_cache.csv"
        ts_path.parent.mkdir(parents=True)
        with open(ts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "name"])

        bs_dir = tmp_path / "data" / "baostock_cache"
        bs_dir.mkdir(parents=True)
        bs_path = bs_dir / "stock_names.csv"
        with open(bs_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "code_name"])
            w.writerow(["sz.000001", "FallbackName"])

        symbols = {"sz000001"}

        p1, p2 = self._patch_roots(tmp_path)
        with p1, p2:
            from ashare_lab.paper.pipeline import _load_stock_names_cache

            result = _load_stock_names_cache(symbols)

        assert result is not None
        assert result["sz000001"] == "FallbackName"


# ---------------------------------------------------------------------------
# Chinese report format integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReportChineseFormat:
    """Verify format_chinese_report produces correct Chinese output."""

    @pytest.fixture
    def sample_report(self):
        return ReportData(
            trade_date="2026-06-30",
            total_nav=250000.0,
            cash=80000.0,
            daily_return_pct=0.75,
            daily_pnl=1875.0,
            cumulative_return_pct=5.0,
            max_drawdown_pct=3.2,
            cash_ratio=32.0,
            benchmark_csi1000=6500.0,
            benchmark_return_pct=0.3,
            trades=[
                {
                    "symbol": "sz000001",
                    "name": "PingAn",
                    "side": "buy",
                    "qty": 200,
                    "price": 12.50,
                    "total_fee": 3.75,
                },
            ],
            positions=[
                {
                    "symbol": "sz000001",
                    "name": "PingAn",
                    "qty": 200,
                    "market_value": 2500.0,
                    "unrealized_pnl": 0.0,
                    "weight": 1.0,
                    "daily_change_pct": 0.02,
                },
            ],
            pending_orders=[
                {
                    "symbol": "sh600000",
                    "name": "PuFa",
                    "side": "sell",
                    "qty": 100,
                },
            ],
            trade_count=1,
            risk_status={
                "buying_halted": False,
                "is_soft_reduced": False,
                "sell_order_count": 0,
                "cooldown_count": 0,
                "drawdown_halted": False,
                "regime_halted": False,
            },
            industry_distribution={"Bank": 1},
        )

    def test_contains_chinese_verbs(self, sample_report):
        text = format_chinese_report(sample_report)
        assert "买" in text
        assert "拟卖" in text

    def test_contains_section_headers(self, sample_report):
        text = format_chinese_report(sample_report)
        assert "今日交易" in text
        assert "持仓分布" in text
        assert "明日计划" in text

    def test_stock_names_in_output(self, sample_report):
        text = format_chinese_report(sample_report)
        assert "PingAn" in text
        assert "PuFa" in text

    def test_risk_normal_indicator(self, sample_report):
        text = format_chinese_report(sample_report)
        assert "风控正常" in text


# ---------------------------------------------------------------------------
# Fetcher exports exist (import smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetcher_exports():
    """Verify Plan 01 fetcher exports are importable."""
    from ashare_lab.data.fetcher import (
        fetch_cross_validation_sample,
        fetch_today_data,
        refresh_stock_names_cache,
    )

    assert callable(fetch_today_data)
    assert callable(fetch_cross_validation_sample)
    assert callable(refresh_stock_names_cache)


@pytest.mark.integration
def test_validator_exports():
    """Verify Plan 01 validator exports are importable."""
    from ashare_lab.data.validator import validate_daily_data

    assert callable(validate_daily_data)


@pytest.mark.integration
def test_report_format_export():
    """Verify Plan 02 report export is importable."""
    from ashare_lab.paper.report import format_chinese_report

    assert callable(format_chinese_report)
