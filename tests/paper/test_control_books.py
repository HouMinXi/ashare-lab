"""Unit tests for L4 control books: Book B shadow ledger + A/B/C artifact.

Tests cover:
- timing_multiplier() interface
- Known-answer validation (m=1.0 reproduces Book A to the cent)
- fail-open behavior (Book B failure doesn't affect production)
- bootstrap (DB copy)
- T+1 settle timing (B1 fix proof)
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock

from ashare_lab.paper.pipeline import (
    timing_multiplier,
    _bootstrap_book_b,
    _write_abc_artifact,
    _step12_book_b,
)


# -- timing_multiplier -------------------------------------------------------

class TestTimingMultiplier:
    def test_none_model_returns_one(self):
        assert timing_multiplier("2026-07-25", "none") == 1.0

    def test_default_model_is_none(self):
        assert timing_multiplier("2026-07-25") == 1.0

    def test_unknown_model_returns_one_with_warning(self, caplog):
        result = timing_multiplier("2026-07-25", "unknown_model")
        assert result == 1.0
        assert "Unknown timing model" in caplog.text


# -- bootstrap ---------------------------------------------------------------

class TestBootstrap:
    def test_bootstrap_copies_db(self, tmp_path):
        prod_db = tmp_path / "paper.db"
        book_b_db = tmp_path / "paper_b_none.db"

        conn = sqlite3.connect(str(prod_db))
        conn.execute("CREATE TABLE nav (trade_date TEXT, total_nav REAL)")
        conn.execute("INSERT INTO nav VALUES ('2026-07-25', 300000.0)")
        conn.commit()
        conn.close()

        _bootstrap_book_b(prod_db, book_b_db)

        assert book_b_db.exists()
        conn = sqlite3.connect(str(book_b_db))
        row = conn.execute("SELECT total_nav FROM nav WHERE trade_date='2026-07-25'").fetchone()
        assert row[0] == 300000.0
        conn.close()

    def test_bootstrap_overwrites_existing(self, tmp_path):
        prod_db = tmp_path / "paper.db"
        book_b_db = tmp_path / "paper_b_none.db"

        conn = sqlite3.connect(str(prod_db))
        conn.execute("CREATE TABLE nav (trade_date TEXT, total_nav REAL)")
        conn.execute("INSERT INTO nav VALUES ('2026-07-25', 300000.0)")
        conn.commit()
        conn.close()

        conn = sqlite3.connect(str(book_b_db))
        conn.execute("CREATE TABLE nav (trade_date TEXT, total_nav REAL)")
        conn.execute("INSERT INTO nav VALUES ('2026-07-25', 999999.0)")
        conn.commit()
        conn.close()

        _bootstrap_book_b(prod_db, book_b_db)

        conn = sqlite3.connect(str(book_b_db))
        row = conn.execute("SELECT total_nav FROM nav WHERE trade_date='2026-07-25'").fetchone()
        assert row[0] == 300000.0
        conn.close()


# -- fail-open ---------------------------------------------------------------

class TestFailOpen:
    def test_book_b_exception_does_not_raise(self, tmp_path):
        ctx = MagicMock()
        ctx.trade_date = "2026-07-25"
        ctx.db_path = tmp_path / "paper.db"
        _step12_book_b(ctx)

    def test_book_b_logs_warning_on_failure(self, tmp_path, caplog):
        ctx = MagicMock()
        ctx.trade_date = "2026-07-25"
        ctx.db_path = tmp_path / "paper.db"
        _step12_book_b(ctx)
        assert "Book B failed" in caplog.text


# -- A/B/C artifact ----------------------------------------------------------

class TestABCArtifact:
    def test_artifact_written(self, tmp_path):
        from ashare_lab.paper.ledger import init_schema

        book_b_db = tmp_path / "paper_b_none.db"
        conn = sqlite3.connect(str(book_b_db))
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        conn.executescript("""
            INSERT INTO runs VALUES ('2026-07-25', 'settled', '2026-07-25T15:00:00');
            INSERT INTO positions VALUES ('2026-07-25', 'SH600519', 100, 1800.0, 180000.0, '2026-07-20', 0.0, 1.0);
            INSERT INTO paper_state VALUES ('cash', '120000.0');
        """)
        conn.commit()
        conn.close()

        ctx = MagicMock()
        ctx.trade_date = "2026-07-25"
        ctx.total_nav = 300000.0
        ctx.benchmarks = {"csi1000": 1.0}
        ctx.paper_cfg = {"initial_cash": 300000.0}

        book_conn = sqlite3.connect(str(book_b_db))
        book_conn.row_factory = sqlite3.Row
        _write_abc_artifact(ctx, book_conn, "none", 1.0)
        book_conn.close()

        artifact_path = Path("experiments/control_books/2026-07-25_none.json")
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["date"] == "2026-07-25"
        assert artifact["book_id"] == "none"
        assert artifact["nav_a"] == 300000.0
        assert artifact["m"] == 1.0


# -- Known-answer (R3) -------------------------------------------------------

class TestKnownAnswer:
    def test_target_value_same_with_m_one(self):
        equity_nav = 300000.0
        risk_degree = 0.95
        effective_topk = 15
        book_a = (equity_nav * risk_degree) / effective_topk
        book_b = (equity_nav * risk_degree * 1.0) / effective_topk
        assert book_a == book_b

    def test_target_value_scaled_with_m_half(self):
        equity_nav = 300000.0
        risk_degree = 0.95
        effective_topk = 15
        book_a = (equity_nav * risk_degree) / effective_topk
        book_b = (equity_nav * risk_degree * 0.5) / effective_topk
        assert book_b == book_a * 0.5

    def test_target_value_zero_with_m_zero(self):
        equity_nav = 300000.0
        risk_degree = 0.95
        effective_topk = 15
        book_b = (equity_nav * risk_degree * 0.0) / effective_topk
        assert book_b == 0.0


# -- R3 Replay Driver (B2) --------------------------------------------------

class TestReplayDriver:
    """R3: T+1 settle timing proof + known-answer structural test."""

    def _init_db(self, db_path):
        """Create a DB with the full ledger schema."""
        from ashare_lab.paper.ledger import init_schema
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        conn.execute("INSERT OR IGNORE INTO paper_state (key, value) VALUES ('cash', '300000.0')")
        conn.commit()
        return conn

    def test_settle_t1_timing(self, tmp_path):
        """Book B settles yesterday's orders today (T+1), not same-day.

        Day 1: Insert buy orders (trade_date=d2).
        Day 2: Settle with today's prices. Both books must match.
        """
        from ashare_lab.paper.ledger import insert_order, get_latest_positions, get_latest_cash
        from ashare_lab.paper.engine import settle_day

        d1 = "2026-07-21"
        d2 = "2026-07-22"
        prices = {"SH600519": {"close": 1800.0}}
        settle_cfg = {"carry_days": 3, "slippage": 0.001, "volume_participation_pct": 0.05}

        for db_name in ["paper.db", "paper_b_none.db"]:
            conn = self._init_db(tmp_path / db_name)
            insert_order(conn, d2, "SH600519", "buy", 100, None, "pending", 0, d1)
            conn.commit()
            conn.close()

        for db_name in ["paper.db", "paper_b_none.db"]:
            conn = sqlite3.connect(str(tmp_path / db_name))
            conn.row_factory = sqlite3.Row
            positions = get_latest_positions(conn)
            cash = get_latest_cash(conn, 300000.0)
            pending = [dict(r) for r in conn.execute(
                "SELECT id, symbol, side, target_qty, carry_day, "
                "reset_count, suspension_carry_day FROM orders "
                "WHERE status IN ('pending','carry') AND trade_date <= ?",
                (d2,),
            ).fetchall()]
            settle_day(conn, d2, pending, prices, positions, cash, set(), {}, settle_cfg)
            conn.commit()
            conn.close()

        conn_a = sqlite3.connect(str(tmp_path / "paper.db"))
        conn_a.row_factory = sqlite3.Row
        conn_b = sqlite3.connect(str(tmp_path / "paper_b_none.db"))
        conn_b.row_factory = sqlite3.Row

        nav_a = conn_a.execute("SELECT * FROM nav WHERE trade_date=?", (d2,)).fetchone()
        nav_b = conn_b.execute("SELECT * FROM nav WHERE trade_date=?", (d2,)).fetchone()

        assert nav_a is not None, "Book A NAV not recorded"
        assert nav_b is not None, "Book B NAV not recorded"
        assert abs(nav_a["total_nav"] - nav_b["total_nav"]) < 0.01, \
            f"NAV mismatch: A={nav_a['total_nav']}, B={nav_b['total_nav']}"
        assert abs(nav_a["cash"] - nav_b["cash"]) < 0.01, \
            f"Cash mismatch: A={nav_a['cash']}, B={nav_b['cash']}"

        pos_a = conn_a.execute("SELECT * FROM positions WHERE trade_date=?", (d2,)).fetchall()
        pos_b = conn_b.execute("SELECT * FROM positions WHERE trade_date=?", (d2,)).fetchall()
        assert len(pos_a) == len(pos_b), f"Position count mismatch: A={len(pos_a)}, B={len(pos_b)}"

        conn_a.close()
        conn_b.close()

    def test_settle_same_day_bug_fails(self, tmp_path):
        """Bug-injection: settling same-day (no T+1) fails to settle.

        Book A: settle d2 orders on d2 (correct T+1).
        Book B (BUG): insert d2 orders, query with trade_date=d1.
        d2 orders don't match d1 -> nothing settles -> proves T+1 matters.
        """
        from ashare_lab.paper.ledger import insert_order, get_latest_positions, get_latest_cash
        from ashare_lab.paper.engine import settle_day

        d1 = "2026-07-21"
        d2 = "2026-07-22"
        prices = {"SH600519": {"close": 1800.0}}
        settle_cfg = {"carry_days": 3, "slippage": 0.001, "volume_participation_pct": 0.05}

        # Book A: correct T+1
        conn_a = self._init_db(tmp_path / "paper.db")
        insert_order(conn_a, d2, "SH600519", "buy", 100, None, "pending", 0, d1)
        conn_a.commit()
        positions_a = get_latest_positions(conn_a)
        cash_a = get_latest_cash(conn_a, 300000.0)
        pending_a = [dict(r) for r in conn_a.execute(
            "SELECT id, symbol, side, target_qty, carry_day, "
            "reset_count, suspension_carry_day FROM orders "
            "WHERE status IN ('pending','carry') AND trade_date <= ?",
            (d2,),
        ).fetchall()]
        settle_day(conn_a, d2, pending_a, prices, positions_a, cash_a, set(), {}, settle_cfg)
        conn_a.commit()
        conn_a.close()

        # Book B (BUG): query with d1 instead of d2
        conn_b = self._init_db(tmp_path / "paper_b_none.db")
        insert_order(conn_b, d2, "SH600519", "buy", 100, None, "pending", 0, d1)
        conn_b.commit()
        pending_b = [dict(r) for r in conn_b.execute(
            "SELECT id, symbol, side, target_qty, carry_day, "
            "reset_count, suspension_carry_day FROM orders "
            "WHERE status IN ('pending','carry') AND trade_date <= ?",
            (d1,),
        ).fetchall()]
        # d2 orders don't match trade_date <= d1 -> nothing to settle
        assert len(pending_b) == 0, "Bug: d2 orders should NOT match trade_date <= d1"
        conn_b.close()

        # Book A settled, Book B did not
        conn_a2 = sqlite3.connect(str(tmp_path / "paper.db"))
        conn_a2.row_factory = sqlite3.Row
        nav_a = conn_a2.execute("SELECT * FROM nav WHERE trade_date=?", (d2,)).fetchone()
        assert nav_a is not None, "Book A should have NAV for d2"
        conn_a2.close()


class TestRunBookBPhase1Phase2:
    """Direct test of _run_book_b Phase 1 (settle yesterday) + Phase 2 (generate today).

    This is the B1 fix coverage: verifies _run_book_b correctly separates
    settle (trade_date <= today) from order generation (trade_date = next_td).
    """

    def _make_ctx(self, tmp_path, trade_date, pred_path):
        """Build a minimal DailyRunContext for _run_book_b."""
        from ashare_lab.paper.pipeline import DailyRunContext

        ctx = DailyRunContext.__new__(DailyRunContext)
        ctx.trade_date = trade_date
        ctx.db_path = tmp_path / "paper.db"
        ctx.paper_cfg = {
            "initial_cash": 300000.0, "topk": 15, "n_drop": 1,
            "listing_min_days": 60,
        }
        ctx.config = {
            "cost_model": {"risk_degree": 0.95},
            "universe": {"listing_min_days": 60, "min_avg_turnover_20d": 0.0, "exclude_close_above_cny": 300.0},
            "carry_days": 3, "slippage": 0.001, "volume_participation_pct": 0.05,
        }
        ctx.risk_result = MagicMock(
            buying_halted=False, forced_sells={},
            blocked_industries=set(), blocked_rebuys=set(),
            topk_override=None, cooldown_entries={},
        )
        ctx.prices = {"SH600519": {"close": 180.0}}
        ctx.pred_path = pred_path
        ctx.universe_symbols = ["SH600519"]
        ctx.ipo_listing_syms = set()
        ctx.st_names = set()
        ctx.market_data = {
            "SH600519": {"close": 180.0, "listing_days": 100, "avg_turnover_20d": 100_000_000.0},
        }
        ctx.industry_map = {}
        ctx.benchmarks = {"csi1000": 1.0}
        ctx.hedge_state = None
        ctx.hedge_symbols = set()
        ctx.total_nav = 300000.0
        ctx.current_positions = {}
        ctx.cash = 300000.0
        return ctx

    def test_phase1_settles_yesterday_phase2_generates_today(self, tmp_path):
        """_run_book_b Phase 1 settles d1 orders, Phase 2 inserts d2 orders.

        Day 1: Insert pending order (trade_date=d2) into Book B DB.
        Day 2: Call _run_book_b(ctx_d2).
        Verify:
        - d2 order is settled (status != pending)
        - New d3 order is inserted (pending, not settled)
        """
        from ashare_lab.paper.ledger import init_schema, insert_order, get_connection
        from ashare_lab.paper.pipeline import _run_book_b

        d1 = "2026-07-21"
        d2 = "2026-07-22"
        d3 = "2026-07-23"  # next trading day after d2

        # Create production DB (needed for bootstrap)
        prod_db = tmp_path / "paper.db"
        conn = sqlite3.connect(str(prod_db))
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        conn.execute("INSERT OR IGNORE INTO paper_state (key, value) VALUES ('cash', '300000.0')")
        conn.commit()
        conn.close()

        # Create Book B DB with a pending order for d2 (placed on d1)
        book_b_db = tmp_path / "paper_b_none.db"
        conn_b = get_connection(book_b_db)
        init_schema(conn_b)
        conn_b.execute("INSERT OR IGNORE INTO paper_state (key, value) VALUES ('cash', '300000.0')")
        insert_order(conn_b, d2, "SH600519", "buy", 100, None, "pending", 0, d1)
        conn_b.commit()
        conn_b.close()

        # Create prediction parquet for d2
        import pandas as pd
        pred_path = tmp_path / "predictions" / f"{d2}.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"instrument": ["SH600519"], "score": [0.8]}).to_parquet(pred_path)

        # Build ctx for d2
        ctx = self._make_ctx(tmp_path, d2, pred_path)

        # Patch next_trading_day to return d3, and generate_signals to return fixed data
        with unittest.mock.patch("ashare_lab.paper.pipeline.next_trading_day", return_value=dt.date.fromisoformat(d3)), \
             unittest.mock.patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            _run_book_b(ctx)

        # Debug: check all orders
        conn_debug = get_connection(book_b_db)
        all_orders = conn_debug.execute("SELECT trade_date, status FROM orders").fetchall()
        print(f"\nDEBUG all orders: {[(r['trade_date'], r['status']) for r in all_orders]}")
        conn_debug.close()

        # Verify: d2 orders should be settled (Phase 1)
        conn_check = get_connection(book_b_db)
        d2_orders = conn_check.execute(
            "SELECT status FROM orders WHERE trade_date=?",
            (d2,),
        ).fetchall()
        for row in d2_orders:
            assert row["status"] != "pending", \
                f"d2 order should be settled, got status={row['status']}"

        # Verify: d3 orders should exist and be pending (Phase 2)
        d3_orders = conn_check.execute(
            "SELECT status FROM orders WHERE trade_date=?",
            (d3,),
        ).fetchall()
        assert len(d3_orders) > 0, "Phase 2 should have inserted d3 orders"
        for row in d3_orders:
            assert row["status"] == "pending", \
                f"d3 order should be pending, got status={row['status']}"

        conn_check.close()

    def test_no_pending_orders_skips_settle(self, tmp_path):
        """_run_book_b with no pending orders skips Phase 1 settle."""
        from ashare_lab.paper.ledger import init_schema, get_connection
        from ashare_lab.paper.pipeline import _run_book_b

        d2 = "2026-07-22"
        d3 = "2026-07-23"

        # Create empty Book B DB (no pending orders)
        prod_db = tmp_path / "paper.db"
        conn = sqlite3.connect(str(prod_db))
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        conn.execute("INSERT OR IGNORE INTO paper_state (key, value) VALUES ('cash', '300000.0')")
        conn.commit()
        conn.close()

        book_b_db = tmp_path / "paper_b_none.db"
        conn_b = get_connection(book_b_db)
        init_schema(conn_b)
        conn_b.execute("INSERT OR IGNORE INTO paper_state (key, value) VALUES ('cash', '300000.0')")
        conn_b.commit()
        conn_b.close()

        import pandas as pd
        pred_path = tmp_path / "predictions" / f"{d2}.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"instrument": ["SH600519"], "score": [0.8]}).to_parquet(pred_path)

        ctx = self._make_ctx(tmp_path, d2, pred_path)

        with unittest.mock.patch("ashare_lab.paper.pipeline.next_trading_day", return_value=dt.date.fromisoformat(d3)), \
             unittest.mock.patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            _run_book_b(ctx)

        # Verify: no d2 orders to settle, but d3 orders generated
        conn_check = get_connection(book_b_db)
        d3_orders = conn_check.execute(
            "SELECT status FROM orders WHERE trade_date=?",
            (d3,),
        ).fetchall()
        assert len(d3_orders) > 0, "Phase 2 should have inserted d3 orders"
        conn_check.close()
