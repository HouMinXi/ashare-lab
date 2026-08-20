"""Book B settled-run recording regression tests.

Root cause: _run_book_b never called record_run, so the 'latest settled'
anchor in get_latest_positions was frozen at the bootstrap-copy day forever.
Every daily run restarted positions from the copied snapshot: sells of
copied positions did not stick, positions bought after bootstrap were
invisible next run.

Fix: settle_day called unconditionally + record_run on the success path.
These tests exercise the REAL _run_book_b path, not hand-seeded rows,
to catch regressions in the actual wiring.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import pandas as pd

from ashare_lab.paper.ledger import (
    get_connection,
    get_latest_positions,
    init_schema,
    insert_order,
)
from ashare_lab.paper.pipeline import _bootstrap_book_b, _run_book_b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path, trade_date, pred_path, book_a_nav=1_000_000.0):
    """Build a minimal DailyRunContext for _run_book_b tests."""
    from ashare_lab.paper.pipeline import DailyRunContext

    ctx = DailyRunContext.__new__(DailyRunContext)
    ctx.trade_date = trade_date
    ctx.db_path = tmp_path / "paper.db"
    ctx.paper_cfg = {
        "initial_cash": book_a_nav,
        "topk": 15,
        "n_drop": 1,
        "listing_min_days": 60,
        "turnover_cap": None,
    }
    ctx.config = {
        "cost_model": {"risk_degree": 0.95},
        "universe": {
            "listing_min_days": 60,
            "min_avg_turnover_20d": 0.0,
            "exclude_close_above_cny": 300.0,
        },
        "carry_days": 3,
        "slippage": 0.001,
        "volume_participation_pct": 0.05,
    }
    ctx.risk_result = MagicMock(
        buying_halted=False,
        forced_sells={},
        blocked_industries=set(),
        blocked_rebuys=set(),
        topk_override=None,
        cooldown_entries={},
    )
    ctx.prices = {
        "SH600519": {"close": 180.0, "volume": 10_000_000.0},
        "SH601318": {"close": 55.0, "volume": 8_000_000.0},
    }
    ctx.pred_path = pred_path
    ctx.universe_symbols = ["SH600519", "SH601318"]
    ctx.ipo_listing_syms = set()
    ctx.st_names = set()
    ctx.market_data = {
        "SH600519": {
            "close": 180.0,
            "listing_days": 100,
            "avg_turnover_20d": 100_000_000.0,
        },
        "SH601318": {
            "close": 55.0,
            "listing_days": 200,
            "avg_turnover_20d": 80_000_000.0,
        },
    }
    ctx.industry_map = {}
    ctx.benchmarks = {"csi1000": 1.0}
    ctx.hedge_state = None
    ctx.hedge_symbols = set()
    ctx.total_nav = book_a_nav
    ctx.current_positions = {}
    ctx.cash = book_a_nav
    return ctx


def _init_prod_db(tmp_path, cash=1_000_000.0):
    """Create a production DB with nav state (needed for bootstrap copy)."""
    prod_db = tmp_path / "paper.db"
    conn = sqlite3.connect(str(prod_db))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.execute(
        "INSERT OR REPLACE INTO nav (trade_date, cash, market_value, total_nav) "
        "VALUES (?, ?, ?, ?)",
        ("2026-07-01", cash, 0.0, cash),
    )
    conn.commit()
    conn.close()
    return prod_db


def _make_pred_path(tmp_path, trade_date):
    """Create a placeholder prediction parquet."""
    pred_path = tmp_path / "predictions" / f"{trade_date}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"instrument": ["SH600519", "SH601318"], "score": [0.8, 0.7]}).to_parquet(pred_path)
    return pred_path


def _seed_buy_order(book_conn, trade_date, symbol="SH600519", qty=100):
    """Insert a pending buy order into the Book B DB for settle_day to fill."""
    insert_order(book_conn, trade_date, symbol, "buy", qty, None, "pending", 0, trade_date)
    book_conn.commit()


def _seed_sell_order(book_conn, trade_date, symbol="SH600519", qty=100):
    """Insert a pending sell order into the Book B DB for settle_day to fill."""
    insert_order(book_conn, trade_date, symbol, "sell", qty, None, "pending", 0, trade_date)
    book_conn.commit()


def _seed_position(book_conn, trade_date, symbol="SH600519", qty=100,
                   avg_cost=180.0, market_value=18000.0):
    """Seed a position snapshot directly (for day-0 bootstrap simulation)."""
    market_val = qty * avg_cost
    total_nav = 1_000_000.0 + market_val
    book_conn.execute(
        "INSERT OR REPLACE INTO nav (trade_date, cash, market_value, total_nav) "
        "VALUES (?, ?, ?, ?)",
        (trade_date, 1_000_000.0, market_val, total_nav),
    )
    book_conn.execute(
        "INSERT INTO positions "
        "(trade_date, symbol, qty, avg_cost, market_value, buy_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (trade_date, symbol, qty, avg_cost, market_val, trade_date),
    )
    book_conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBookBRunRecord:
    """Verify _run_book_b records settled runs in the book DB."""

    def test_book_b_marks_run_settled(self, tmp_path):
        """After a simulated Book B run, runs table has (date, 'settled')."""
        trade_date = "2026-08-15"
        _init_prod_db(tmp_path)
        pred_path = _make_pred_path(tmp_path, trade_date)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx, "none")

        book_db = tmp_path / "paper_b_none.db"
        assert book_db.exists()
        conn = get_connection(book_db)
        row = conn.execute(
            "SELECT status FROM runs WHERE trade_date = ?", (trade_date,)
        ).fetchone()
        conn.close()
        assert row is not None, f"no runs row for {trade_date}"
        assert row["status"] == "settled"

    def test_book_b_positions_carry_across_days(self, tmp_path):
        """Day T: settle a buy via real code path; day T+1: zero pending orders.

        The code under test must:
        - record_run(T) so the anchor advances (injection A: anchor frozen,
          get_latest_positions returns bootstrap snapshot, SH600519 absent).
        - call settle_day unconditionally on T+1 so a positions snapshot is
          written for that day (injection B: no snapshot, anchor points at T,
          but T+1 has no rows -> fully-liquidated semantics, empty dict).
        """
        day_t = "2026-08-15"
        day_t1 = "2026-08-16"
        _init_prod_db(tmp_path)
        pred_t = _make_pred_path(tmp_path, day_t)
        pred_t1 = _make_pred_path(tmp_path, day_t1)
        ctx_t = _make_ctx(tmp_path, day_t, pred_t)
        ctx_t1 = _make_ctx(tmp_path, day_t1, pred_t1)

        book_db = tmp_path / "paper_b_none.db"

        # Bootstrap the Book B DB (creates paper_b_none.db from prod).
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx_t, "none")

        # Seed position + pending buy with trade_date=day_t.
        # Day T's SECOND run will settle this buy via the real settle_day.
        conn_b = get_connection(book_db)
        _seed_position(conn_b, day_t, "SH600519", qty=100, avg_cost=180.0)
        _seed_buy_order(conn_b, day_t, "SH600519", qty=100)
        conn_b.close()

        # Day T: run again -- settles the buy, record_run advances the anchor.
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx_t, "none")

        # Verify the code wrote exactly one settled row for day T.
        conn_b = get_connection(book_db)
        runs_t = conn_b.execute(
            "SELECT trade_date, status FROM runs WHERE trade_date = ?",
            (day_t,),
        ).fetchall()
        assert len(runs_t) == 1, f"expected 1 runs row for {day_t}, got {len(runs_t)}: {runs_t}"
        assert runs_t[0]["status"] == "settled"
        conn_b.close()

        # Day T+1: run with genuinely zero pending orders.
        # The buy was already settled on day T; no new seed.
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx_t1, "none")

        # Positions from day T must carry into T+1.
        conn_b = get_connection(book_db)
        positions = get_latest_positions(conn_b)
        runs = conn_b.execute(
            "SELECT trade_date, status FROM runs ORDER BY trade_date",
        ).fetchall()
        conn_b.close()

        assert "SH600519" in positions, (
            f"SH600519 vanished after no-order day; positions={positions}"
        )
        assert len(runs) == 2, f"expected 2 settled runs, got {len(runs)}: {runs}"

    def test_book_b_sold_position_stays_sold(self, tmp_path):
        """Hold a symbol, settle its sell on day T; day T+1 must NOT contain it.

        Regression: the frozen 08-14 snapshot resurrected SH600667 after it
        was sold on 08-18 because held_set was rebuilt from the bootstrap
        snapshot.
        """
        day_t = "2026-08-15"
        day_t1 = "2026-08-16"
        _init_prod_db(tmp_path)
        pred_t = _make_pred_path(tmp_path, day_t)
        pred_t1 = _make_pred_path(tmp_path, day_t1)
        ctx_t = _make_ctx(tmp_path, day_t, pred_t)
        ctx_t1 = _make_ctx(tmp_path, day_t1, pred_t1)

        book_db = tmp_path / "paper_b_none.db"

        # Day T: bootstrap + seed a position, then sell it
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx_t, "none")

        conn_b = get_connection(book_db)
        _seed_position(conn_b, day_t, "SH600519", qty=100, avg_cost=180.0)
        _seed_sell_order(conn_b, day_t, "SH600519", qty=100)
        conn_b.execute(
            "INSERT OR REPLACE INTO runs (trade_date, status, started_at) "
            "VALUES (?, 'settled', datetime('now'))",
            (day_t,),
        )
        conn_b.commit()
        conn_b.close()

        # Day T+1: run -- sell should have been settled, position gone
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            _run_book_b(ctx_t1, "none")

        conn_b = get_connection(book_db)
        positions = get_latest_positions(conn_b)
        conn_b.close()
        assert "SH600519" not in positions, (
            f"SH600519 resurrected after sell; positions={positions}"
        )

    def test_book_b_failed_record_run_leaves_anchor(self, tmp_path):
        """When record_run raises, artifact is already committed but no runs
        row exists for T -- anchor stays at T-1 and next run re-anchors cleanly.

        Injection proof: patching ashare_lab.paper.ledger.record_run (the
        lazy-imported target at pipeline.py:2028) to raise RuntimeError
        triggers the failure window.  If the patch target is wrong (a no-op
        location), this test would PASS instead of FAIL.
        """
        trade_date = "2026-08-15"
        _init_prod_db(tmp_path)
        pred_path = _make_pred_path(tmp_path, trade_date)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        book_db = tmp_path / "paper_b_none.db"
        _bootstrap_book_b(ctx.db_path, book_db)

        # Clean pre-existing artifact so the assertion tests *this* run's write
        artifact = Path("experiments/control_books") / f"{trade_date}_none.json"
        artifact.unlink(missing_ok=True)

        conn_b = get_connection(book_db)
        _seed_buy_order(conn_b, trade_date, "SH600519", qty=100)
        conn_b.close()

        # Patch the lazy-imported record_run to raise -- this is the real
        # call site: pipeline.py:2028 imports it, pipeline.py:2177 calls it.
        with patch("ashare_lab.paper.ledger.record_run", side_effect=RuntimeError("DB write failure")), \
             patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={}):
            with pytest.raises(RuntimeError, match="DB write failure"):
                _run_book_b(ctx, "none")

        # (b) A/B/C artifact already committed before the failure
        artifact = Path("experiments/control_books") / f"{trade_date}_none.json"
        assert artifact.exists(), (
            f"artifact {artifact} should exist -- it was committed before "
            f"record_run raised"
        )

        # (c) runs table has NO row for trade_date T
        conn_b = get_connection(book_db)
        row = conn_b.execute(
            "SELECT status FROM runs WHERE trade_date = ?", (trade_date,)
        ).fetchone()
        conn_b.close()
        assert row is None, (
            f"runs table should have no row for {trade_date} after "
            f"record_run failure, but found: {row}"
        )

    def test_book_b_settle_visible_same_day(self, tmp_path):
        """Settled positions visible to order generation on the SAME day.

        Production runs _run_book_b exactly ONCE per day.  At the post-settle
        reload (pipeline.py:2060), record_run has NOT yet been called for
        today (it runs at line 2177), so get_latest_positions still anchors
        on yesterday.  Using get_positions_for_date(T) instead is critical:
        injection C (swap to get_latest_positions) causes a duplicate buy
        because held_set misses the just-settled position.

        This test mirrors the production single-run shape:
        1. Bootstrap the Book B DB directly (no _run_book_b, so no
           record_run(T) -- this is what the old test masked by calling
           _run_book_b for bootstrap, which advanced the anchor to T).
        2. Seed position + pending buy for SH600519 on day T.
        3. Run _run_book_b ONCE on day T with real signals containing X.
        4. Assert: NO pending buy for SH600519 with trade_date > T
           (the duplicate-buy check).
        5. Assert: the day-T buy settled (filled) so the scenario is honest.
        """
        day_t = "2026-08-15"
        _init_prod_db(tmp_path)
        pred_t = _make_pred_path(tmp_path, day_t)
        ctx_t = _make_ctx(tmp_path, day_t, pred_t)

        book_db = tmp_path / "paper_b_none.db"

        # Bootstrap the Book B DB directly -- no _run_book_b call, so no
        # record_run(T).  This matches production before the first daily run.
        _bootstrap_book_b(ctx_t.db_path, book_db)

        # Seed position + pending buy for SH600519 on day T.
        conn_b = get_connection(book_db)
        _seed_position(conn_b, day_t, "SH600519", qty=100, avg_cost=180.0)
        _seed_buy_order(conn_b, day_t, "SH600519", qty=100)
        conn_b.close()

        # generate_signals returns both symbols with real scores.
        real_signals = {"SH600519": 0.9, "SH601318": 0.7}

        # Single day-T run: settle the buy + generate next-day orders.
        # Under injection C (get_latest_positions at 2060), held_set misses
        # SH600519 because the anchor still points at yesterday (no
        # record_run yet), so order generation emits a duplicate buy for T+1.
        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value=real_signals):
            _run_book_b(ctx_t, "none")

        conn_b = get_connection(book_db)

        # The day-T buy must have settled (filled).
        settled_buy = conn_b.execute(
            "SELECT trade_date, status, target_qty FROM orders "
            "WHERE symbol = 'SH600519' AND side = 'buy' AND trade_date = ?",
            (day_t,),
        ).fetchone()
        assert settled_buy is not None, (
            f"no buy order for SH600519 on {day_t} -- settle_day did not fill"
        )
        assert settled_buy["status"] == "filled", (
            f"day-T buy status={settled_buy['status']}, expected 'filled'"
        )

        # Duplicate-buy check: NO pending buy for SH600519 with trade_date > T.
        dup = conn_b.execute(
            "SELECT id, trade_date, status FROM orders "
            "WHERE symbol = 'SH600519' AND side = 'buy' "
            "AND trade_date > ? AND status = 'pending'",
            (day_t,),
        ).fetchall()
        assert len(dup) == 0, (
            f"duplicate pending buy for SH600519 after single day-T run: "
            f"{[dict(r) for r in dup]}"
        )

        conn_b.close()
