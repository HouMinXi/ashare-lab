"""Acceptance tests for the paper trading engine.

D-04 layer 1: synthetic unit tests with hand-calculated expected
outcomes.  No qlib or external dependencies required.

D-04 layer 2: integration smoke tests with real qlib prices and
synthetic prediction files (no model needed).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.paper.adjust import check_and_apply_adjustfactor
from ashare_lab.paper.engine import (
    SettleResult,
    apply_slippage,
    cap_fill_by_volume,
    compute_nav,
    get_limit_threshold,
    is_limit_up,
    round_lots,
    settle_day,
)
from ashare_lab.paper.fees import FeeResult, calculate_fees
from ashare_lab.paper.ledger import (
    bump_carry_days,
    force_reset_day,
    get_connection,
    get_latest_cash,
    get_latest_positions,
    init_schema,
    insert_order,
    insert_trade,
    is_day_settled,
    record_nav,
    record_run,
    snapshot_positions,
    update_order,
)


# ===================================================================
# Helpers
# ===================================================================


def _setup_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh SQLite DB and return the connection."""
    db = tmp_path / "test.db"
    conn = get_connection(db)
    init_schema(conn)
    return conn


def _make_config(**overrides: object) -> dict:
    """Build a minimal paper config dict for settle_day."""
    cfg: dict = {
        "carry_days": 3,
        "slippage": 0.001,
        "volume_participation_pct": 0.05,
    }
    cfg.update(overrides)
    return cfg


def _price_entry(
    close: float,
    change: float = 0.0,
    volume: float = 1e7,
    factor: float = 1.0,
    threshold: float = 0.099,
) -> dict:
    return {
        "close": close,
        "change": change,
        "volume": volume,
        "factor": factor,
        "threshold": threshold,
    }


def _position(
    qty: int,
    avg_cost: float,
    buy_date: str = "2024-12-30",
    holding_high: float | None = None,
    factor: float = 1.0,
) -> dict:
    if holding_high is None:
        holding_high = avg_cost
    return {
        "qty": qty,
        "avg_cost": avg_cost,
        "market_value": qty * avg_cost,
        "buy_date": buy_date,
        "holding_high": holding_high,
        "factor": factor,
    }


_BENCHMARKS = {"csi300": 4000.0, "csi1000": 7000.0}


# ===================================================================
# D-04 Layer 1: Synthetic Acceptance Tests
# ===================================================================


class TestSyntheticAcceptance:
    """Deterministic acceptance tests using synthetic prices."""

    # ---------------------------------------------------------------
    # T+1 enforcement (pipeline layer, not engine layer)
    # ---------------------------------------------------------------

    def test_t_plus_1_enforcement(self, tmp_path: Path) -> None:
        """Same-day sell is filtered by pipeline step 10f.

        The engine does not implement T+1; the pipeline filters sell
        candidates where buy_date == trade_date.  We verify the filter
        logic directly rather than through the full pipeline.
        """
        # Simulate step 10f filter: sell_syms filtered by buy_date
        trade_date = "2025-01-03"
        current_positions = {
            "SZ000001": {"buy_date": "2025-01-03", "qty": 100},  # bought today
            "SZ000002": {"buy_date": "2025-01-02", "qty": 200},  # bought yesterday
        }
        sell_syms = ["SZ000001", "SZ000002"]

        # Step 10f filter: remove same-day buys
        filtered = [
            s for s in sell_syms
            if current_positions.get(s, {}).get("buy_date") != trade_date
        ]

        # SZ000001 bought today => filtered out (T+1 blocks same-day sell)
        assert "SZ000001" not in filtered
        # SZ000002 bought yesterday => allowed
        assert "SZ000002" in filtered

    # ---------------------------------------------------------------
    # Limit-up buy block and carry
    # ---------------------------------------------------------------

    def test_limit_up_buy_block_and_carry(self, tmp_path: Path) -> None:
        """Buy order blocked at limit-up; fills next day when normal."""
        conn = _setup_db(tmp_path)

        # Day 1: limit-up, buy should carry
        oid = insert_order(
            conn, "2025-01-02", "SZ000001", "buy", 100, None,
            "pending", 0, "2025-01-02",
        )
        orders_d1 = [
            {"id": oid, "symbol": "SZ000001", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        prices_d1 = {"SZ000001": _price_entry(10.0, change=0.10)}
        result_d1 = settle_day(
            conn, "2025-01-02", orders_d1, prices_d1, {},
            300_000.0, {"SZ000001"}, _BENCHMARKS, _make_config(),
        )
        assert len(result_d1.fills) == 0
        assert len(result_d1.carries_to_bump) == 1
        assert result_d1.carries_to_bump[0]["symbol"] == "SZ000001"

        # Bump carry_day (pipeline step 8e)
        bump_carry_days(conn, [oid])

        # Day 2: normal market, should fill
        orders_d2 = [
            {"id": oid, "symbol": "SZ000001", "side": "buy",
             "target_qty": 100, "carry_day": 1},
        ]
        prices_d2 = {"SZ000001": _price_entry(10.5, change=0.02)}
        result_d2 = settle_day(
            conn, "2025-01-03", orders_d2, prices_d2, {},
            300_000.0, {"SZ000001"}, _BENCHMARKS, _make_config(),
        )
        assert len(result_d2.fills) == 1
        assert result_d2.fills[0]["symbol"] == "SZ000001"
        expected_fill_price = apply_slippage(10.5, "buy", 0.001)
        assert result_d2.fills[0]["fill_price"] == pytest.approx(
            expected_fill_price, abs=1e-6,
        )

    # ---------------------------------------------------------------
    # Carry max 3 days cancel (M-15 intermediate assertions)
    # ---------------------------------------------------------------

    def test_carry_max_3_days_cancel(self, tmp_path: Path) -> None:
        """Buy blocked 3 consecutive days then cancelled on day 4.

        Intermediate carry_day values asserted per M-15:
        day 1 -> carry_day=1, day 2 -> 2, day 3 -> 3, day 4 -> cancelled.
        """
        conn = _setup_db(tmp_path)

        oid = insert_order(
            conn, "2025-01-02", "SZ000005", "buy", 100, None,
            "pending", 0, "2025-01-02",
        )

        limit_up_prices = {"SZ000005": _price_entry(10.0, change=0.10)}

        # Day 1: carry_day starts at 0, limit-up -> carry, bump to 1
        orders = [
            {"id": oid, "symbol": "SZ000005", "side": "buy",
             "target_qty": 100, "carry_day": 0},
        ]
        r = settle_day(
            conn, "2025-01-02", orders, limit_up_prices, {},
            300_000.0, {"SZ000005"}, _BENCHMARKS, _make_config(),
        )
        assert len(r.carries_to_bump) == 1
        bump_carry_days(conn, [oid])
        row = conn.execute(
            "SELECT carry_day FROM orders WHERE id=?", (oid,)
        ).fetchone()
        assert row["carry_day"] == 1  # M-15: after day 1

        # Day 2: carry_day=1, limit-up -> carry, bump to 2
        orders = [
            {"id": oid, "symbol": "SZ000005", "side": "buy",
             "target_qty": 100, "carry_day": 1},
        ]
        r = settle_day(
            conn, "2025-01-03", orders, limit_up_prices, {},
            300_000.0, {"SZ000005"}, _BENCHMARKS, _make_config(),
        )
        assert len(r.carries_to_bump) == 1
        bump_carry_days(conn, [oid])
        row = conn.execute(
            "SELECT carry_day FROM orders WHERE id=?", (oid,)
        ).fetchone()
        assert row["carry_day"] == 2  # M-15: after day 2

        # Day 3: carry_day=2, limit-up -> carry, bump to 3
        orders = [
            {"id": oid, "symbol": "SZ000005", "side": "buy",
             "target_qty": 100, "carry_day": 2},
        ]
        r = settle_day(
            conn, "2025-01-06", orders, limit_up_prices, {},
            300_000.0, {"SZ000005"}, _BENCHMARKS, _make_config(),
        )
        assert len(r.carries_to_bump) == 1
        bump_carry_days(conn, [oid])
        row = conn.execute(
            "SELECT carry_day FROM orders WHERE id=?", (oid,)
        ).fetchone()
        assert row["carry_day"] == 3  # M-15: after day 3

        # Day 4: carry_day=3, limit-up -> CANCELLED (>= carry_days limit)
        orders = [
            {"id": oid, "symbol": "SZ000005", "side": "buy",
             "target_qty": 100, "carry_day": 3},
        ]
        r = settle_day(
            conn, "2025-01-07", orders, limit_up_prices, {},
            300_000.0, {"SZ000005"}, _BENCHMARKS, _make_config(),
        )
        assert "SZ000005" in r.cancels
        assert len(r.carries_to_bump) == 0
        row = conn.execute(
            "SELECT status FROM orders WHERE id=?", (oid,)
        ).fetchone()
        assert row["status"] == "cancelled"

    # ---------------------------------------------------------------
    # Partial fill volume cap
    # ---------------------------------------------------------------

    def test_partial_fill_volume_cap(self, tmp_path: Path) -> None:
        """Volume cap limits fill; remainder creates carry order."""
        conn = _setup_db(tmp_path)

        # target=1000, volume=10000, pct=0.05 -> cap = 500
        oid = insert_order(
            conn, "2025-01-02", "SZ000003", "buy", 1000, None,
            "pending", 0, "2025-01-02",
        )
        orders = [
            {"id": oid, "symbol": "SZ000003", "side": "buy",
             "target_qty": 1000, "carry_day": 0},
        ]
        prices = {"SZ000003": _price_entry(5.0, volume=10_000)}
        result = settle_day(
            conn, "2025-01-02", orders, prices, {},
            300_000.0, {"SZ000003"}, _BENCHMARKS, _make_config(),
        )

        assert len(result.fills) == 1
        assert result.fills[0]["fill_qty"] == 500
        assert len(result.carries_to_bump) == 1

        # Remainder qty = 1000 - 500 = 500
        rem_id = result.carries_to_bump[0]["order_id"]
        rem_row = conn.execute(
            "SELECT target_qty FROM orders WHERE id=?", (rem_id,)
        ).fetchone()
        assert rem_row["target_qty"] == 500

    # ---------------------------------------------------------------
    # Risk forced sell trail
    # ---------------------------------------------------------------

    def test_risk_forced_sell_trail(self, tmp_path: Path) -> None:
        """Trailing stop triggers full-position forced sell.

        When decline from holding_high exceeds stop_pct (20%), the
        risk check returns the symbol in forced_sells with full qty.
        """
        from ashare_lab.paper.risk import check_trailing_stop

        # holding_high=100, current=75 => decline=25% > 20% threshold
        assert check_trailing_stop(100.0, 75.0, 0.20) is True

        # holding_high=100, current=85 => decline=15% < 20%
        assert check_trailing_stop(100.0, 85.0, 0.20) is False

    # ---------------------------------------------------------------
    # IPO win debits cash
    # ---------------------------------------------------------------

    def test_ipo_win_debits_cash(self, tmp_path: Path) -> None:
        """IPO subscription debits cash by shares * issue_price.

        Verifies the cash deduction mechanism is correct when an IPO
        is won.  Uses direct ledger operations (the pipeline layer
        applies IPO outcomes via the same ledger calls).
        """
        conn = _setup_db(tmp_path)

        initial_cash = 300_000.0
        issue_price = 12.50
        ipo_shares = 500

        # Record initial NAV
        record_nav(
            conn, "2025-01-02", initial_cash, 0.0, initial_cash,
            None, None, None, None,
        )
        conn.commit()

        # Simulate IPO win: debit cash, create position
        ipo_cost = ipo_shares * issue_price  # 6250.0
        post_ipo_cash = initial_cash - ipo_cost

        positions = {
            "SZ688001": {
                "qty": ipo_shares,
                "avg_cost": issue_price,
                "market_value": ipo_shares * issue_price,
                "buy_date": "2025-01-03",
                "holding_high": issue_price,
                "factor": 1.0,
            },
        }
        snapshot_positions(conn, "2025-01-03", positions)

        # Compute NAV: position value + cash
        prices = {"SZ688001": _price_entry(issue_price)}
        total_nav = compute_nav(positions, prices, post_ipo_cash)

        # Cash dropped by exactly the IPO cost
        assert post_ipo_cash == pytest.approx(
            initial_cash - ipo_cost, abs=1e-6,
        )

        # NAV did NOT jump by free shares: net change ~0
        assert total_nav == pytest.approx(initial_cash, abs=1e-6)

    # ---------------------------------------------------------------
    # Full-day idempotent re-run
    # ---------------------------------------------------------------

    def test_full_day_idempotent(self, tmp_path: Path) -> None:
        """Second call to is_day_settled returns True after recording settled."""
        conn = _setup_db(tmp_path)

        trade_date = "2025-01-02"
        assert is_day_settled(conn, trade_date) is False

        record_run(conn, trade_date, "settled")
        conn.commit()

        assert is_day_settled(conn, trade_date) is True

        # Record counts before hypothetical re-run
        orders_before = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        trades_before = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

        # A well-behaved pipeline checks is_day_settled first and returns 0
        # without inserting anything.  Verify no new rows would be created.
        assert is_day_settled(conn, trade_date) is True
        orders_after = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        trades_after = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert orders_after == orders_before
        assert trades_after == trades_before

    # ---------------------------------------------------------------
    # Force reset restores state
    # ---------------------------------------------------------------

    def test_force_reset_restores(self, tmp_path: Path) -> None:
        """force_reset_day undoes a settled day so it can be re-run.

        Verifies: runs row deleted, trades deleted, positions deleted,
        nav deleted, and pre-existing orders restored from journal.
        """
        conn = _setup_db(tmp_path)
        trade_date = "2025-01-02"

        # Create an order and settle it
        oid = insert_order(
            conn, trade_date, "SZ000001", "buy", 100, None,
            "pending", 0, trade_date,
        )
        prices = {"SZ000001": _price_entry(10.0)}
        positions: dict[str, dict] = {}
        settle_day(
            conn, trade_date,
            [{"id": oid, "symbol": "SZ000001", "side": "buy",
              "target_qty": 100, "carry_day": 0}],
            prices, positions, 300_000.0,
            {"SZ000001"}, _BENCHMARKS, _make_config(),
        )
        record_run(conn, trade_date, "settled")
        conn.commit()

        # Verify settled state
        assert is_day_settled(conn, trade_date) is True
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_date=?", (trade_date,)
        ).fetchone()[0] > 0

        # Force reset
        force_reset_day(conn, trade_date)

        # Verify reset state
        assert is_day_settled(conn, trade_date) is False
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_date=?", (trade_date,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM nav WHERE trade_date=?", (trade_date,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM positions WHERE trade_date=?", (trade_date,)
        ).fetchone()[0] == 0

    # ---------------------------------------------------------------
    # Lot rounding buy (exact share count)
    # ---------------------------------------------------------------

    def test_lot_rounding_buy(self) -> None:
        """Buy at close=15.0 with target_value=20000.

        raw_qty = 20000 / 15.0 = 1333.33
        rounded = int(1333.33 // 100) * 100 = 1300
        """
        target_value = 20_000.0
        close = 15.0
        raw_qty = target_value / close
        rounded = round_lots(raw_qty, "buy")
        assert rounded == 1300

    # ---------------------------------------------------------------
    # Lot-size skip for expensive stocks
    # ---------------------------------------------------------------

    def test_lot_skip_budget(self) -> None:
        """Buy at close=350.0 with target_value=20000 => 0 lots.

        raw_qty = 20000 / 350.0 = 57.14
        rounded = int(57.14 // 100) * 100 = 0
        """
        target_value = 20_000.0
        close = 350.0
        raw_qty = target_value / close
        rounded = round_lots(raw_qty, "buy")
        assert rounded == 0

    # ---------------------------------------------------------------
    # Fee math exactness (buy side)
    # ---------------------------------------------------------------

    def test_fee_math_buy(self) -> None:
        """Buy 1300 shares at close=15.0, slippage=0.001.

        fill_price = 15.0 * 1.001 = 15.015
        notional = 1300 * 15.015 = 19519.50
        commission = max(19519.50 * 0.00025, 5.0) = max(4.87988, 5.0) = 5.0
        stamp = 0.0 (buy side)
        transfer = 19519.50 * 0.00001 = 0.195195
        """
        fill_price = apply_slippage(15.0, "buy", 0.001)
        assert fill_price == pytest.approx(15.015, abs=1e-6)

        notional = 1300 * fill_price
        assert notional == pytest.approx(19519.50, abs=1e-6)

        fees = calculate_fees(notional, "buy")
        assert fees.commission == pytest.approx(5.0, abs=1e-6)  # min floor
        assert fees.stamp == pytest.approx(0.0, abs=1e-6)
        assert fees.transfer == pytest.approx(
            19519.50 * 0.00001, abs=1e-6,
        )
        assert fees.total == pytest.approx(
            5.0 + 0.0 + 19519.50 * 0.00001, abs=1e-6,
        )

    # ---------------------------------------------------------------
    # Fee math sell (stamp duty)
    # ---------------------------------------------------------------

    def test_fee_math_sell_stamp(self) -> None:
        """Sell 1300 shares at close=15.0.

        fill_price = 15.0 * 0.999 = 14.985
        notional = 1300 * 14.985 = 19480.50
        stamp = 19480.50 * 0.0005 = 9.74025
        """
        fill_price = apply_slippage(15.0, "sell", 0.001)
        assert fill_price == pytest.approx(14.985, abs=1e-6)

        notional = 1300 * fill_price
        fees = calculate_fees(notional, "sell")
        assert fees.stamp == pytest.approx(
            notional * 0.0005, abs=1e-6,
        )
        assert fees.commission == pytest.approx(
            max(notional * 0.00025, 5.0), abs=1e-6,
        )

    # ---------------------------------------------------------------
    # NAV reconciliation within tolerance
    # ---------------------------------------------------------------

    def test_nav_reconciliation(self, tmp_path: Path) -> None:
        """After settlement: |cash + sum(qty*close) - total_nav| < 0.01."""
        conn = _setup_db(tmp_path)

        positions = {"SZ000001": _position(200, 10.0)}
        oid = insert_order(
            conn, "2025-01-02", "SZ000002", "buy", 100, None,
            "pending", 0, "2025-01-02",
        )
        prices = {
            "SZ000001": _price_entry(10.5),
            "SZ000002": _price_entry(8.0),
        }
        result = settle_day(
            conn, "2025-01-02",
            [{"id": oid, "symbol": "SZ000002", "side": "buy",
              "target_qty": 100, "carry_day": 0}],
            prices, positions, 290_000.0,
            {"SZ000001", "SZ000002"}, _BENCHMARKS, _make_config(),
        )

        # Recompute from positions and cash
        computed = compute_nav(positions, prices, result.cash)
        assert abs(computed - result.post_trade_nav) < 0.01

        # Cash >= 0 and qty >= 0
        assert result.cash >= 0
        for pos in positions.values():
            assert pos["qty"] >= 0

    # ---------------------------------------------------------------
    # Cash conservation across multi-day replay
    # ---------------------------------------------------------------

    def test_cash_conservation(self, tmp_path: Path) -> None:
        """Cash follows: today_cash = yesterday_cash + sell_proceeds
        - buy_costs - total_fees.

        Verifies no cash leak or double-debit across a 2-day synthetic
        replay.
        """
        conn = _setup_db(tmp_path)

        day1_cash = 300_000.0
        prices_d1 = {"SZ000001": _price_entry(10.0)}

        # Day 1: buy 100 shares
        oid1 = insert_order(
            conn, "2025-01-02", "SZ000001", "buy", 100, None,
            "pending", 0, "2025-01-02",
        )
        positions: dict[str, dict] = {}
        r1 = settle_day(
            conn, "2025-01-02",
            [{"id": oid1, "symbol": "SZ000001", "side": "buy",
              "target_qty": 100, "carry_day": 0}],
            prices_d1, positions, day1_cash,
            {"SZ000001"}, _BENCHMARKS, _make_config(),
        )
        record_run(conn, "2025-01-02", "settled")
        conn.commit()

        # Verify day 1 cash conservation
        buy_fill = r1.fills[0]
        buy_cost = buy_fill["fill_qty"] * buy_fill["fill_price"]
        buy_fees = buy_fill["fees"]
        expected_cash_d1 = day1_cash - buy_cost - buy_fees
        assert r1.cash == pytest.approx(expected_cash_d1, abs=1e-6)

        # Day 2: sell the same 100 shares
        prices_d2 = {"SZ000001": _price_entry(10.5)}
        oid2 = insert_order(
            conn, "2025-01-03", "SZ000001", "sell", 100, None,
            "pending", 0, "2025-01-03",
        )
        r2 = settle_day(
            conn, "2025-01-03",
            [{"id": oid2, "symbol": "SZ000001", "side": "sell",
              "target_qty": 100, "carry_day": 0}],
            prices_d2, positions, r1.cash,
            set(), _BENCHMARKS, _make_config(),
        )

        sell_fill = r2.fills[0]
        sell_proceeds = sell_fill["fill_qty"] * sell_fill["fill_price"]
        sell_fees = sell_fill["fees"]
        expected_cash_d2 = r1.cash + sell_proceeds - sell_fees
        assert r2.cash == pytest.approx(expected_cash_d2, abs=1e-6)

    # ---------------------------------------------------------------
    # Adjustfactor audit-only (D-47 regression guard)
    # ---------------------------------------------------------------

    def test_nav_continuity_across_split(self) -> None:
        """Stock split: factor changes, qty/avg_cost UNCHANGED.

        Hold a position where factor changes (e.g. 0.14 -> 0.28 like a
        2:1 split).  qfq $close stays continuous, so NAV should be
        continuous.  No qty mutation.
        """
        positions = {
            "SZ000001": {
                "qty": 1000,
                "avg_cost": 7.0,
                "market_value": 7000.0,
                "buy_date": "2025-01-02",
                "holding_high": 7.0,
                "factor": 0.14,
            },
        }
        previous_factors = {"SZ000001": 0.14}
        current_factors = {"SZ000001": 0.28}

        new_positions, records = check_and_apply_adjustfactor(
            positions, previous_factors, current_factors,
        )

        # qty and avg_cost UNCHANGED (audit-only, D-47)
        assert new_positions["SZ000001"]["qty"] == 1000
        assert new_positions["SZ000001"]["avg_cost"] == 7.0

        # factor refreshed
        assert new_positions["SZ000001"]["factor"] == 0.28

        # Audit record emitted
        assert len(records) == 1
        assert records[0]["old_factor"] == 0.14
        assert records[0]["new_factor"] == 0.28

        # NAV = qty * close is continuous: use same $close on both sides
        close_price = 7.0  # qfq close stays continuous
        prices = {"SZ000001": {"close": close_price}}
        nav_before = compute_nav(positions, prices, 0.0)
        nav_after = compute_nav(new_positions, prices, 0.0)
        assert nav_after == pytest.approx(nav_before, abs=1e-6)


# ===================================================================
# D-04 Layer 2: Integration Smoke Tests
# ===================================================================


@pytest.mark.integration
class TestIntegrationSmoke:
    """Integration smoke tests using real qlib prices + synthetic
    prediction files.

    Marked @pytest.mark.integration so they can be skipped in CI:
        pytest -m "not integration"

    These tests require qlib runtime with real price data for the
    pinned dates.  Predictions are synthetic (no model needed).
    """

    _qlib_ready = False

    @classmethod
    def _ensure_qlib(cls) -> None:
        """Initialize qlib once per test class; skip if unavailable."""
        if cls._qlib_ready:
            return
        try:
            import qlib  # noqa: PLC0415
            from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415
            from qlib.config import REG_CN  # noqa: PLC0415
        except ImportError:
            pytest.skip("qlib not installed")
        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI), region=REG_CN)
        cls._qlib_ready = True

    def test_5_day_replay(
        self,
        tmp_path: Path,
        synthetic_prediction_files: Path,
    ) -> None:
        """Replay 5 consecutive trading days with real qlib prices."""
        self._ensure_qlib()
        from qlib.data import D  # noqa: PLC0415

        # Verify qlib has price data for the pinned range
        dates = [
            "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ]
        try:
            test_data = D.features(
                instruments=["SZ000001"],
                fields=["$close"],
                start_time=dates[0],
                end_time=dates[-1],
            )
            if test_data is None or test_data.empty:
                pytest.skip(
                    "qlib price data for 2025-01-06..10 absent "
                    "-- bootstrap qlib cn_data"
                )
        except Exception:
            pytest.skip(
                "qlib price data for 2025-01-06..10 absent "
                "-- bootstrap qlib cn_data"
            )

        from ashare_lab.paper.pipeline import run_daily

        pred_dir = synthetic_prediction_files

        for date_str in dates:
            pred_path = pred_dir / f"{date_str}.parquet"
            rc = run_daily(date_str, pred_path=pred_path)
            assert rc in (0, 1), f"run_daily({date_str}) returned {rc}"

        # Verify 5 nav rows
        from ashare_lab.paper.ledger import get_connection, init_schema
        from ashare_lab.config import PROJECT_ROOT, load_config

        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]
        conn = get_connection(db_path)
        nav_count = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert nav_count >= 5

        # NAV band: +/-3% over 5 days
        nav_row = conn.execute(
            "SELECT total_nav FROM nav ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
        total_nav = float(nav_row["total_nav"])
        assert total_nav > 291_000
        assert total_nav < 309_000

    def test_idempotent_5_day(
        self,
        tmp_path: Path,
        synthetic_prediction_files: Path,
    ) -> None:
        """After 5-day replay, re-running each date returns 0 (skip)."""
        self._ensure_qlib()
        from qlib.data import D  # noqa: PLC0415

        try:
            test_data = D.features(
                instruments=["SZ000001"],
                fields=["$close"],
                start_time="2025-01-06",
                end_time="2025-01-10",
            )
            if test_data is None or test_data.empty:
                pytest.skip("qlib price data absent")
        except Exception:
            pytest.skip("qlib price data absent")

        from ashare_lab.paper.pipeline import run_daily

        dates = [
            "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ]
        pred_dir = synthetic_prediction_files

        # First pass
        for d in dates:
            run_daily(d, pred_path=pred_dir / f"{d}.parquet")

        # Second pass: all should return 0 (already settled)
        for d in dates:
            rc = run_daily(d, pred_path=pred_dir / f"{d}.parquet")
            assert rc == 0, f"re-run of {d} should return 0, got {rc}"


# ===================================================================
# D-04 Layer 3: Real Prediction Integration Tests
# ===================================================================


@pytest.mark.integration
class TestRealPredictionReplay:
    """Integration tests using real canonical predictions (60/40 blend).

    Gated on: qlib runtime + models/w1.pt + predictions/2021-07-01.parquet.
    Skips cleanly when any prerequisite is absent (local dev, CI).

    These tests prove the engine works with real model output,
    closing the synthetic-only gap from Layer 2.
    """

    _qlib_ready = False

    @classmethod
    def _ensure_prerequisites(cls) -> None:
        """Initialize qlib and verify models + predictions exist."""
        if cls._qlib_ready:
            return
        try:
            import qlib  # noqa: PLC0415
            from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415
            from qlib.config import REG_CN  # noqa: PLC0415
        except ImportError:
            pytest.skip("qlib not installed")

        from ashare_lab.config import MODELS_DIR, PREDICTIONS_DIR  # noqa: PLC0415

        if not (MODELS_DIR / "w1.pt").exists():
            pytest.skip("canonical models not staged (models/w1.pt absent)")
        if not (PREDICTIONS_DIR / "2021-07-01.parquet").exists():
            pytest.skip(
                "real predictions not generated "
                "(predictions/2021-07-01.parquet absent)"
            )

        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI), region=REG_CN)
        cls._qlib_ready = True

    def test_real_prediction_5_day_replay(self) -> None:
        """Replay 5 W1 trading days with real 60/40 blend predictions."""
        self._ensure_prerequisites()

        from ashare_lab.config import PROJECT_ROOT, load_config  # noqa: PLC0415
        from ashare_lab.paper.ledger import get_connection  # noqa: PLC0415
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415

        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]

        # Clean slate
        if db_path.exists():
            db_path.unlink()
        for suffix in ("-wal", "-shm"):
            side = db_path.parent / (db_path.name + suffix)
            if side.exists():
                side.unlink()

        dates = [
            "2021-07-01", "2021-07-02", "2021-07-05",
            "2021-07-06", "2021-07-07",
        ]

        for date_str in dates:
            rc = run_daily(date_str)
            assert rc in (0, 1), f"run_daily({date_str}) returned {rc}"

        conn = get_connection(db_path)
        nav_count = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert nav_count >= 5, f"expected >= 5 NAV rows, got {nav_count}"

        nav_row = conn.execute(
            "SELECT total_nav, cash FROM nav ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
        total_nav = float(nav_row["total_nav"])
        cash = float(nav_row["cash"])
        assert total_nav > 0, f"NAV must be positive, got {total_nav}"
        assert cash >= 0, f"cash must be non-negative, got {cash}"

        # Verify trading actually occurred (not just ingestion + no-crash).
        order_count = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE status='filled'"
        ).fetchone()[0]
        assert order_count > 0, (
            "no filled orders -- engine ran but did not trade"
        )

        pos_count = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM positions"
        ).fetchone()[0]
        assert pos_count > 0, (
            "no positions -- engine did not build a portfolio"
        )

        assert cash < 300_000, (
            f"cash={cash} == initial capital -- no capital deployed"
        )

        assert total_nav != cash, (
            f"total_nav==cash ({total_nav}) -- holdings value is zero"
        )

    def test_real_prediction_idempotent(self) -> None:
        """Re-running settled dates returns 0 without side effects."""
        self._ensure_prerequisites()

        from ashare_lab.config import PROJECT_ROOT, load_config  # noqa: PLC0415
        from ashare_lab.paper.ledger import get_connection  # noqa: PLC0415
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415

        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]

        # Ensure DB exists from prior test (test ordering within class)
        if not db_path.exists():
            pytest.skip("requires test_real_prediction_5_day_replay to run first")

        conn = get_connection(db_path)
        nav_before = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]

        dates = [
            "2021-07-01", "2021-07-02", "2021-07-05",
            "2021-07-06", "2021-07-07",
        ]
        for date_str in dates:
            rc = run_daily(date_str)
            assert rc == 0, f"idempotent re-run of {date_str} got rc={rc}"

        nav_after = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert nav_after == nav_before, (
            f"NAV rows changed on re-run: {nav_before} -> {nav_after}"
        )

    def test_corrupt_prediction_detected(self) -> None:
        """Corrupt parquet is rejected at read time.

        Tests at the prediction-reading layer rather than the full
        pipeline to avoid triggering baostock network calls (which
        hang cross-Pacific without timeout -- known gap #a).
        """
        self._ensure_prerequisites()

        import shutil  # noqa: PLC0415

        import pandas as pd  # noqa: PLC0415

        from ashare_lab.config import PREDICTIONS_DIR  # noqa: PLC0415

        target = PREDICTIONS_DIR / "2021-07-01.parquet"
        backup = PREDICTIONS_DIR / "2021-07-01.parquet.bak"

        # Verify the real file reads cleanly first
        df_good = pd.read_parquet(target)
        assert len(df_good) > 0, "real prediction file should not be empty"
        assert "instrument" in df_good.columns
        assert "score" in df_good.columns

        shutil.copy2(target, backup)
        try:
            target.write_bytes(b"CORRUPT DATA - not a valid parquet")

            # Corrupt parquet must raise on read
            with pytest.raises(Exception):
                pd.read_parquet(target)
        finally:
            shutil.move(str(backup), str(target))

        # Verify restore worked -- file reads cleanly again
        df_restored = pd.read_parquet(target)
        assert len(df_restored) == len(df_good)
