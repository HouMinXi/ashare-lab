"""Tests for _step5_fetch_prices_and_universe with mocked qlib ecosystem."""
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.paper.pipeline import _step5_fetch_prices_and_universe, DailyRunContext


@pytest.fixture
def ctx():
    """Create a minimal DailyRunContext for testing."""
    conn = MagicMock()
    conn.execute.return_value = []  # No pending orders
    return DailyRunContext(
        conn=conn,
        trade_date="2025-01-15",
        current_positions={},
        steps={"full"},
        force=False,
        pred_path=None,
        start_time=0.0,
        predictions_date_str="2025-01-15",
    )


@pytest.fixture
def mock_qlib():
    """Mock qlib module and its D object."""
    mock_D = MagicMock()
    mock_D.list_instruments.return_value = ["SH600000", "SZ000001"]
    mock_D.instruments.return_value = "csi1000"

    mock_qlib_data = MagicMock()
    mock_qlib_data.D = mock_D

    with patch.dict(sys.modules, {
        "qlib": MagicMock(),
        "qlib.data": mock_qlib_data,
    }):
        yield mock_D


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_returns_2_on_qlib_import_failure(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx,
):
    """Test that step5 returns 2 when qlib import fails."""
    # Remove qlib from sys.modules to simulate import failure
    with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}):
        result = _step5_fetch_prices_and_universe(ctx)

    assert result == 2
    ctx.conn.execute.assert_called()  # record_run was called


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_fetches_universe_and_prices(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx, mock_qlib,
):
    """Test that step5 fetches universe, prices, and populates ctx."""
    # Mock IPO calendar (no IPOs)
    mock_ipo.return_value = []

    # Mock ST cache (no ST stocks)
    mock_st_cache.return_value = set()

    # Mock subprocess for price fetching
    price_data = {
        "SH600000": {"close": 10.0, "change": 0.01, "volume": 1e6, "factor": 1.0},
        "SZ000001": {"close": 15.0, "change": -0.02, "volume": 2e6, "factor": 1.0},
    }
    mock_subprocess.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(price_data),
    )

    # Mock benchmarks
    mock_benchmark.return_value = [100.0, 101.0]

    # Mock regime/industry caches
    mock_regime.return_value = [1000.0] * 11
    mock_industry.return_value = {"SH600000": "银行", "SZ000001": "房地产"}

    result = _step5_fetch_prices_and_universe(ctx)

    assert result != 2  # Should not fail
    assert len(ctx.universe_symbols) == 2
    assert len(ctx.prices) == 2
    assert "SH600000" in ctx.prices
    assert ctx.prices["SH600000"]["close"] == 10.0


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_handles_subprocess_timeout(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx, mock_qlib,
):
    """Test that step5 handles subprocess timeout gracefully."""
    mock_ipo.return_value = []
    mock_st_cache.return_value = set()
    mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="", timeout=90)
    mock_benchmark.return_value = []
    mock_regime.return_value = []
    mock_industry.return_value = {}

    result = _step5_fetch_prices_and_universe(ctx)

    assert result != 2  # Should not fail
    assert ctx.prices == {}  # Prices should be empty on timeout


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_populates_fetch_symbols_from_positions_and_orders(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx, mock_qlib,
):
    """Test that fetch_symbols includes universe + positions + orders."""
    # Add position
    ctx.current_positions = {"SZ000001": {"qty": 100, "avg_cost": 10.0}}

    # Add pending order
    ctx.conn.execute.return_value = [("SH600036",)]

    mock_ipo.return_value = []
    mock_st_cache.return_value = set()
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="{}")
    mock_benchmark.return_value = []
    mock_regime.return_value = []
    mock_industry.return_value = {}

    _step5_fetch_prices_and_universe(ctx)

    # Should include all three sources
    assert "SH600000" in ctx.fetch_symbols  # from universe
    assert "SZ000001" in ctx.fetch_symbols  # from positions
    assert "SH600036" in ctx.fetch_symbols  # from orders


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_handles_price_factor_normalization(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx, mock_qlib,
):
    """Test that prices are denormalized by factor."""
    mock_ipo.return_value = []
    mock_st_cache.return_value = set()

    # Price with factor 2.0 means actual price is close/factor
    price_data = {
        "SH600000": {"close": 20.0, "change": 0.01, "volume": 1e6, "factor": 2.0},
    }
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout=json.dumps(price_data),
    )
    mock_benchmark.return_value = []
    mock_regime.return_value = []
    mock_industry.return_value = {}

    _step5_fetch_prices_and_universe(ctx)

    # close should be 20.0 / 2.0 = 10.0
    assert ctx.prices["SH600000"]["close"] == 10.0


@patch("ashare_lab.paper.pipeline._load_industry_cache")
@patch("ashare_lab.paper.pipeline._load_regime_cache")
@patch("ashare_lab.paper.pipeline._fetch_benchmark_closes")
@patch("ashare_lab.paper.pipeline.subprocess.run")
@patch("ashare_lab.paper.pipeline._load_stock_names_cache")
@patch("ashare_lab.paper.pipeline._load_st_cache")
@patch("ashare_lab.paper.pipeline._fetch_ipo_calendar")
def test_step5_missing_factor_not_marked_adjusted(
    mock_ipo, mock_st_cache, mock_names_cache,
    mock_subprocess, mock_benchmark, mock_regime, mock_industry, ctx, mock_qlib,
):
    """When factor is missing from qlib data, price must NOT be marked
    adjusted so the sanity gate can flag it."""
    mock_ipo.return_value = []
    mock_st_cache.return_value = set()

    # Factor key absent -- qlib returned no factor data
    price_data = {
        "SH600000": {"close": 1.2, "change": 0.01, "volume": 1e6},
    }
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout=json.dumps(price_data),
    )
    mock_benchmark.return_value = []
    mock_regime.return_value = []
    mock_industry.return_value = {}

    _step5_fetch_prices_and_universe(ctx)

    # adjusted must be False so _is_normalized_price can detect the issue
    assert ctx.prices["SH600000"]["adjusted"] is False
    # factor stored as 1.0 (fallback) but NOT used for division
    assert ctx.prices["SH600000"]["factor"] == 1.0
