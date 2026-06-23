"""Tests for ashare_lab.paper.engine -- pure helpers and settle_day."""

from __future__ import annotations

import pytest

from ashare_lab.paper.engine import (
    SettleResult,
    apply_slippage,
    cap_fill_by_volume,
    compute_nav,
    get_limit_threshold,
    is_limit_down,
    is_limit_up,
    is_suspended,
    round_lots,
    settle_day,
)
from ashare_lab.paper.ledger import get_connection, init_schema


# ===================================================================
# Pure helper tests
# ===================================================================


class TestRoundLots:
    """round_lots: buy rounds down to lot multiples, sell keeps any qty."""

    def test_buy_rounds_down(self):
        assert round_lots(350, "buy") == 300

    def test_sell_keeps_odd_lots(self):
        assert round_lots(350, "sell") == 350

    def test_buy_below_lot_returns_zero(self):
        assert round_lots(99, "buy") == 0

    def test_buy_zero(self):
        assert round_lots(0, "buy") == 0

    def test_buy_exact_lot(self):
        assert round_lots(200, "buy") == 200

    def test_buy_float_qty(self):
        assert round_lots(350.9, "buy") == 300

    def test_sell_float_qty(self):
        assert round_lots(350.9, "sell") == 350


class TestGetLimitThreshold:
    """get_limit_threshold: board-specific thresholds."""

    def test_default_main_board(self):
        assert get_limit_threshold("000001") == 0.099

    def test_star_688(self):
        assert get_limit_threshold("688001", set()) == 0.199

    def test_star_689(self):
        assert get_limit_threshold("689009", set()) == 0.199

    def test_chinext_300(self):
        assert get_limit_threshold("300001", set()) == 0.199

    def test_st_stock(self):
        assert get_limit_threshold("000001", {"000001"}) == 0.049

    def test_st_none_names(self):
        assert get_limit_threshold("000001", None) == 0.099


class TestIsLimitUp:
    """is_limit_up: threshold is >= (inclusive)."""

    def test_at_threshold(self):
        assert is_limit_up(0.099, 0.099) is True

    def test_above_threshold(self):
        assert is_limit_up(0.10, 0.099) is True

    def test_below_threshold(self):
        assert is_limit_up(0.098, 0.099) is False

    def test_nan_returns_false(self):
        assert is_limit_up(float("nan"), 0.099) is False


class TestIsLimitDown:
    """is_limit_down: threshold is <= -threshold (inclusive)."""

    def test_at_threshold(self):
        assert is_limit_down(-0.099, 0.099) is True

    def test_beyond_threshold(self):
        assert is_limit_down(-0.10, 0.099) is True

    def test_above_threshold(self):
        assert is_limit_down(-0.098, 0.099) is False

    def test_nan_returns_false(self):
        assert is_limit_down(float("nan"), 0.099) is False


class TestIsSuspended:
    """is_suspended: zero or NaN volume means halted."""

    def test_zero_volume(self):
        assert is_suspended(0.0) is True

    def test_nan_volume(self):
        assert is_suspended(float("nan")) is True

    def test_positive_volume(self):
        assert is_suspended(1_000_000) is False


class TestApplySlippage:
    """apply_slippage: always unfavourable to investor."""

    def test_buy_slippage_increases_price(self):
        result = apply_slippage(10.0, "buy", 0.001)
        assert result == pytest.approx(10.01)

    def test_sell_slippage_decreases_price(self):
        result = apply_slippage(10.0, "sell", 0.001)
        assert result == pytest.approx(9.99)

    def test_buy_higher_than_close(self):
        assert apply_slippage(100.0, "buy") > 100.0

    def test_sell_lower_than_close(self):
        assert apply_slippage(100.0, "sell") < 100.0


class TestCapFillByVolume:
    """cap_fill_by_volume: caps at volume * participation_pct."""

    def test_not_binding(self):
        result = cap_fill_by_volume(500, 1_000_000, "buy", 0.05)
        assert result == 500

    def test_binding_cap(self):
        result = cap_fill_by_volume(100_000, 1_000_000, "buy", 0.05)
        assert result == 50_000

    def test_nan_volume_returns_zero(self):
        assert cap_fill_by_volume(500, float("nan"), "buy", 0.05) == 0

    def test_zero_volume_returns_zero(self):
        assert cap_fill_by_volume(500, 0.0, "sell", 0.05) == 0

    def test_sell_no_lot_rounding(self):
        # 1234 * 0.05 = 61.7 -> int = 61, sell keeps 61
        result = cap_fill_by_volume(100, 1234, "sell", 0.05)
        assert result == 61

    def test_buy_lot_rounding(self):
        # 1234 * 0.05 = 61.7 -> int = 61, buy rounds to 0
        result = cap_fill_by_volume(100, 1234, "buy", 0.05)
        assert result == 0


# ===================================================================
# settle_day tests -- use real SQLite via tmp_path
# ===================================================================


def _make_config(**overrides):
    """Build a minimal paper config dict."""
    cfg = {
        "carry_days": 3,
        "slippage": 0.001,
        "volume_participation_pct": 0.05,
    }
    cfg.update(overrides)
    return cfg


def _setup_db(tmp_path):
    """Create a fresh SQLite DB and return the connection."""
    db = tmp_path / "test.db"
    conn = get_connection(db)
    init_schema(conn)
    return conn


def _make_prices(symbol, close, change=0.0, volume=1e7, factor=1.0,
                 threshold=0.099):
    """Build a prices dict entry for one symbol."""
    return {
        symbol: {
            "close": close,
            "change": change,
            "volume": volume,
            "factor": factor,
            "threshold": threshold,
        }
    }


class TestSettleDaySellBeforeBuy:
    """Sell proceeds must fund subsequent buy within the same day."""

    def test_sell_releases_cash_for_buy(self, tmp_path):
        conn = _setup_db(tmp_path)
        # Position: 200 shares of A at 10.0
        positions = {
            "A": {
                "qty": 200,
                "avg_cost": 10.0,
                "market_value": 2000.0,
                "buy_date": "2024-01-01",
                "holding_high": 10.0,
                "factor": 1.0,
            }
        }
        # Cash is NOT enough to buy B without selling A first
        cash = 50.0
        prices = {
            "A": {
                "close": 10.0, "change": 0.0, "volume": 1e7,
                "factor": 1.0, "threshold": 0.099,
            },
            "B": {
                "close": 5.0, "change": 0.0, "volume": 1e7,
                "factor": 1.0, "threshold": 0.099,
            },
        }
        from ashare_lab.paper.ledger import insert_order as lio
        sell_id = lio(
            conn, "2024-01-02", "A", "sell", 200, None, "pending", 0,
            "2024-01-02",
        )
        buy_id = lio(
            conn, "2024-01-02", "B", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": buy_id, "symbol": "B", "side": "buy",
             "target_qty": 100, "carry_day": 0},
            {"id": sell_id, "symbol": "A", "side": "sell",
             "target_qty": 200, "carry_day": 0},
        ]
        result = settle_day(
            conn, "2024-01-02", orders, prices, positions,
            cash, {"A", "B"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        sides = [f["side"] for f in result.fills]
        assert sides == ["sell", "buy"]


class TestSettleDayLimitUpBlock:
    """Buy order blocked when stock is limit-up."""

    def test_limit_up_buy_carries(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "X", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "X", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("X", 10.0, change=0.10, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"X"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.fills) == 0
        assert len(result.carries_to_bump) == 1
        assert result.carries_to_bump[0]["symbol"] == "X"


class TestSettleDayLimitDownBlock:
    """Sell order blocked when stock is limit-down."""

    def test_limit_down_sell_carries(self, tmp_path):
        conn = _setup_db(tmp_path)
        positions = {
            "Y": {
                "qty": 100, "avg_cost": 10.0, "market_value": 1000.0,
                "buy_date": "2024-01-01", "holding_high": 10.0,
                "factor": 1.0,
            }
        }
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "Y", "sell", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "Y", "side": "sell",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("Y", 9.0, change=-0.10, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, positions,
            100_000.0, set(), {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.fills) == 0
        assert len(result.carries_to_bump) == 1


class TestSettleDayCarryOverCancel:
    """Orders exceeding carry_days limit are cancelled."""

    def test_carry_day_3_cancelled(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "Z", "buy", 100, None, "carry", 3,
            "2024-01-01",
        )
        orders = [
            {"id": oid, "symbol": "Z", "side": "buy",
             "target_qty": 100, "carry_day": 3},
        ]
        # Limit-up so it would carry, but carry_day=3 means cancel
        prices = _make_prices("Z", 10.0, change=0.10, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"Z"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert "Z" in result.cancels
        assert len(result.carries_to_bump) == 0


class TestSettleDayTopKRecheck:
    """Carried buy order for non-TopK stock is cancelled."""

    def test_non_topk_carry_cancelled(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "Q", "buy", 100, None, "carry", 1,
            "2024-01-01",
        )
        orders = [
            {"id": oid, "symbol": "Q", "side": "buy",
             "target_qty": 100, "carry_day": 1},
        ]
        prices = _make_prices("Q", 10.0, volume=1e7)
        # Q is NOT in topk
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"A", "B"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert "Q" in result.cancels

    def test_sell_carry_not_cancelled_by_topk(self, tmp_path):
        """Sell carry must NOT be cancelled just because stock left TopK."""
        conn = _setup_db(tmp_path)
        positions = {
            "Q": {
                "qty": 100, "avg_cost": 10.0, "market_value": 1000.0,
                "buy_date": "2024-01-01", "holding_high": 10.0,
                "factor": 1.0,
            }
        }
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "Q", "sell", 100, None, "carry", 1,
            "2024-01-01",
        )
        orders = [
            {"id": oid, "symbol": "Q", "side": "sell",
             "target_qty": 100, "carry_day": 1},
        ]
        prices = _make_prices("Q", 10.0, volume=1e7)
        # Q is NOT in topk -- sell should still fill
        result = settle_day(
            conn, "2024-01-02", orders, prices, positions, 100_000.0,
            {"A", "B"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert "Q" not in result.cancels
        assert len(result.fills) == 1
        assert result.fills[0]["side"] == "sell"


class TestSettleDaySuspension:
    """Suspended stock carries without incrementing carry_day."""

    def test_suspended_buy_carries_to_suspended(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "S", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "S", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        # volume=0 means suspended
        prices = _make_prices("S", 10.0, volume=0.0)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"S"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.carries_suspended) == 1
        assert result.carries_suspended[0]["symbol"] == "S"
        # Not in carries_to_bump (carry_day not bumped)
        assert len(result.carries_to_bump) == 0


class TestSettleDayCashProtection:
    """Buy skipped when insufficient cash (D-25)."""

    def test_insufficient_cash_carries(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "E", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "E", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("E", 100.0, volume=1e7)
        # Cash = 1.0, way too low for 100 shares at 100.0
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 1.0,
            {"E"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.fills) == 0
        assert len(result.carries_to_bump) == 1


class TestSettleDayLotSkip:
    """Lot-size skip: target_qty=0 gets lot_skip status."""

    def test_zero_target_lot_skip(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "L", "buy", 0, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "L", "side": "buy",
             "target_qty": 0, "carry_day": 0},
        ]
        prices = _make_prices("L", 10.0, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"L"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert "L" in result.lot_skips


class TestSettleDayNAVReconciliation:
    """Cash >= 0 and all positions qty >= 0 after settlement."""

    def test_nav_positive_after_normal_settlement(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "N", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "N", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("N", 10.0, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"N"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert result.cash >= 0


class TestSettleDayPartialFill:
    """Partial fill: fill_qty capped by volume participation."""

    def test_partial_fill_remainder(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        # target=1000, volume=10000, pct=0.05 -> cap=500
        oid = lio(
            conn, "2024-01-02", "P", "buy", 1000, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "P", "side": "buy",
             "target_qty": 1000, "carry_day": 0},
        ]
        prices = _make_prices("P", 5.0, volume=10000)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"P"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.fills) == 1
        assert result.fills[0]["fill_qty"] == 500
        # Remainder carry
        assert len(result.carries_to_bump) == 1
        assert result.carries_to_bump[0]["symbol"] == "P"

    def test_partial_fill_remainder_qty_is_reduced(self, tmp_path):
        """Remainder target_qty = target_qty - fill_qty, not original."""
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "P", "buy", 1000, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "P", "side": "buy",
             "target_qty": 1000, "carry_day": 0},
        ]
        prices = _make_prices("P", 5.0, volume=10000)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"P"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        # Check the remainder order in DB
        remainder_id = result.carries_to_bump[0]["order_id"]
        row = conn.execute(
            "SELECT target_qty FROM orders WHERE id = ?",
            (remainder_id,),
        ).fetchone()
        # 1000 - 500 = 500
        assert row["target_qty"] == 500


class TestSettleDayAvgCostUpdate:
    """Avg cost correctly recalculated on additional buy."""

    def test_avg_cost_weighted(self, tmp_path):
        conn = _setup_db(tmp_path)
        positions = {
            "M": {
                "qty": 100,
                "avg_cost": 10.0,
                "market_value": 1000.0,
                "buy_date": "2024-01-01",
                "holding_high": 10.0,
                "factor": 1.0,
            }
        }
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "M", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "M", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("M", 20.0, volume=1e7)
        result = settle_day(
            conn, "2024-01-02", orders, prices, positions,
            100_000.0, {"M"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.fills) == 1
        # New avg = (100*10 + 100*fill_price) / 200
        fill_price = result.fills[0]["fill_price"]
        expected_avg = (100 * 10.0 + 100 * fill_price) / 200
        assert positions["M"]["avg_cost"] == pytest.approx(
            expected_avg, rel=1e-6
        )


class TestSettleDayCapZeroCarry:
    """cap=0 from low liquidity goes to carries_to_bump, not suspended."""

    def test_low_liquidity_carries_to_bump(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "W", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "W", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        # Very low volume: 100 * 0.05 = 5, round_lots(5, buy) = 0
        prices = _make_prices("W", 10.0, volume=100)
        result = settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"W"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert len(result.carries_to_bump) == 1
        assert len(result.carries_suspended) == 0


class TestSettleDayHoldingHighUpdate:
    """Holding high updated even for positions with no buy today."""

    def test_holding_high_updated_on_price_rise(self, tmp_path):
        conn = _setup_db(tmp_path)
        positions = {
            "H": {
                "qty": 100,
                "avg_cost": 10.0,
                "market_value": 1000.0,
                "buy_date": "2024-01-01",
                "holding_high": 10.0,
                "factor": 1.0,
            }
        }
        prices = _make_prices("H", 12.0, volume=1e7)
        # No orders -- just settle to trigger holding_high update
        settle_day(
            conn, "2024-01-02", [], prices, positions,
            100_000.0, set(), {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert positions["H"]["holding_high"] == 12.0


class TestSettleDayBeforeImageJournal:
    """force_reset_day uses before-image journal entries."""

    def test_log_settle_change_called(self, tmp_path):
        conn = _setup_db(tmp_path)
        from ashare_lab.paper.ledger import insert_order as lio
        oid = lio(
            conn, "2024-01-02", "J", "buy", 100, None, "pending", 0,
            "2024-01-02",
        )
        orders = [
            {"id": oid, "symbol": "J", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices = _make_prices("J", 10.0, volume=1e7)
        settle_day(
            conn, "2024-01-02", orders, prices, {}, 100_000.0,
            {"J"}, {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        # Verify journal entry exists
        row = conn.execute(
            "SELECT * FROM order_settle_log WHERE run_date = ?",
            ("2024-01-02",),
        ).fetchone()
        assert row is not None
        assert row["order_id"] == oid


class TestSettleDaySettleResult:
    """SettleResult dataclass returned with correct fields."""

    def test_result_type(self, tmp_path):
        conn = _setup_db(tmp_path)
        result = settle_day(
            conn, "2024-01-02", [], {}, {}, 100_000.0,
            set(), {"csi300": 100.0, "csi1000": 200.0},
            _make_config(),
        )
        assert isinstance(result, SettleResult)
        assert result.cash == 100_000.0
        assert result.fills == []
        assert result.carries_to_bump == []
        assert result.carries_suspended == []


class TestComputeNavReexport:
    """compute_nav re-exported from engine for pipeline use."""

    def test_compute_nav_basic(self):
        positions = {
            "A": {"qty": 100, "avg_cost": 10.0, "market_value": 1000.0}
        }
        prices = {
            "A": {"close": 12.0}
        }
        nav = compute_nav(positions, prices, 5000.0)
        assert nav == pytest.approx(5000.0 + 100 * 12.0)

    def test_compute_nav_missing_price(self):
        positions = {
            "A": {"qty": 100, "avg_cost": 10.0, "market_value": 1000.0}
        }
        nav = compute_nav(positions, {}, 5000.0)
        # Falls back to market_value / qty = 10.0
        assert nav == pytest.approx(5000.0 + 100 * 10.0)
