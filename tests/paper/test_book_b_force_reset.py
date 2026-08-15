"""Book B force-reset regression tests.

A --force rerun of day T resets the production ledger via
force_reset_day but used to leave the paper_b_<id>.db shadow ledgers
untouched, so Book B replayed T's settles into stale state and
double-booked T+1.  These tests seed real Book B databases (production
schema, as the bootstrap copy produces) and assert that both force entry
points -- run_daily's idempotency step and run_backfill's force loop --
roll Book B back together with production.

Dates are arbitrary ISO strings; the force path under test touches only
the ledger, never the trading calendar.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from unittest.mock import patch

from ashare_lab.paper.ledger import get_connection, init_schema, insert_order
from ashare_lab.paper.pipeline import (
    DailyRunContext,
    _BOOK_B_IDS,
    _force_reset_book_b,
    _step2_idempotency,
    run_backfill,
)

T = "2026-07-24"
PREV = "2026-07-23"
NEXT = "2026-07-25"


def _seed_day_state(db_path: Path, trade_date: str) -> int:
    """Seed a full settled day into *db_path*; return the pre-existing order id.

    Layout mirrors what a real day leaves behind:
    - a pre-existing order (created the day before, filled today) with its
      before-image journal row -- force_reset_day must restore it,
    - a pending T+1 order created by today's run -- must be deleted,
    - a filled order created and filled today -- must be deleted,
    - trades/positions/nav/signals/runs/sentiment rows for today -- all
      must be deleted.
    """
    conn = get_connection(db_path)
    init_schema(conn)

    prev_date = (dt.date.fromisoformat(trade_date) - dt.timedelta(days=1)).isoformat()
    next_date = (dt.date.fromisoformat(trade_date) + dt.timedelta(days=1)).isoformat()

    pre_id = insert_order(
        conn, trade_date, "SH600519", "buy", 100, 10.0, "filled", 0, prev_date,
    )
    conn.execute("UPDATE orders SET filled_qty = 100 WHERE id = ?", (pre_id,))
    conn.execute(
        "INSERT INTO order_settle_log "
        "(run_date, order_id, prev_status, prev_filled_qty, prev_carry_day, prev_price) "
        "VALUES (?, ?, 'pending', 0, 0, 10.0)",
        (trade_date, pre_id),
    )

    # T+1 pending placed by today's run (created_run_date = today)
    insert_order(
        conn, next_date, "SH600519", "buy", 200, None, "pending", 0, trade_date,
    )
    # Filled today by today's run
    insert_order(
        conn, trade_date, "SH600519", "sell", 50, 11.0, "filled", 0, trade_date,
    )

    conn.execute(
        "INSERT INTO trades "
        "(order_id, trade_date, symbol, side, fill_price, fill_qty, "
        " commission, stamp, transfer_fee) "
        "VALUES (?, ?, 'SH600519', 'buy', 10.0, 100, 1.0, 1.0, 0.0)",
        (pre_id, trade_date),
    )
    conn.execute(
        "INSERT INTO positions "
        "(trade_date, symbol, qty, avg_cost, market_value, buy_date) "
        "VALUES (?, 'SH600519', 100, 10.0, 1000.0, ?)",
        (trade_date, trade_date),
    )
    conn.execute(
        "INSERT INTO nav (trade_date, cash, market_value, total_nav) "
        "VALUES (?, 100000.0, 1000.0, 101000.0)",
        (trade_date,),
    )
    conn.execute(
        "INSERT INTO signals (trade_date, symbol, score, rank) "
        "VALUES (?, 'SH600519', 0.8, 1)",
        (trade_date,),
    )
    conn.execute(
        "INSERT INTO runs (trade_date, status, started_at) "
        "VALUES (?, 'settled', '2026-07-24T09:00:00')",
        (trade_date,),
    )
    conn.execute(
        "INSERT INTO sentiment_scores (trade_date, layer, target, score) "
        "VALUES (?, 'stock', 'SH600519', 0.5)",
        (trade_date,),
    )
    conn.execute(
        "INSERT INTO sentiment_events (trade_date, event_type, layer, target) "
        "VALUES (?, 'veto', 'stock', 'SH600519')",
        (trade_date,),
    )
    conn.commit()
    conn.close()
    return pre_id


def _assert_day_reset(
    conn, trade_date: str, next_date: str, pre_id: int, expect_restored: bool = True,
) -> None:
    """Assert force_reset_day semantics took effect on the book ledger.

    *expect_restored=False* covers a range reset's cascade: the day's
    pre-existing order was born in the previous day's run, so that
    day's reset deletes it and the replay re-places it.
    """
    # Derived rows for the day are gone.
    for tbl in ("trades", "positions", "nav", "signals", "runs",
                "sentiment_scores", "sentiment_events"):
        n = conn.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE trade_date = ?", (trade_date,)
        ).fetchone()[0]
        assert n == 0, f"{tbl} still has {n} row(s) for {trade_date}"

    # Orders born in this run are gone, including the T+1 pending.
    n = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE created_run_date = ?", (trade_date,)
    ).fetchone()[0]
    assert n == 0, f"{n} order(s) still carry created_run_date={trade_date}"
    n_pending = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE trade_date = ? AND status = 'pending'",
        (next_date,),
    ).fetchone()[0]
    assert n_pending == 0, f"T+1 pending orders survived the reset: {n_pending}"

    # The journal is gone...
    n = conn.execute(
        "SELECT COUNT(*) FROM order_settle_log WHERE run_date = ?", (trade_date,)
    ).fetchone()[0]
    assert n == 0

    # ...and the pre-existing order was either restored from it or
    # cascaded away by an earlier day's reset.
    row = conn.execute(
        "SELECT status, filled_qty, carry_day, price FROM orders WHERE id = ?",
        (pre_id,),
    ).fetchone()
    if expect_restored:
        assert row is not None, "pre-existing order vanished instead of being restored"
        assert tuple(row) == ("pending", 0, 0, 10.0)
    else:
        assert row is None, "pre-existing order survived its cascade delete"


class TestStep2ForceEntry:
    """run_daily's idempotency step resets Book B with production."""

    def test_step2_force_entry_resets_existing_book_b_dbs(self, tmp_path):
        prod_db = tmp_path / "paper.db"
        _seed_day_state(prod_db, T)
        pre_ids = {}
        for book_id in ("none", "n2"):
            pre_ids[book_id] = _seed_day_state(tmp_path / f"paper_b_{book_id}.db", T)
        # n1 / n3 do not exist: the skip path must treat that as a no-op.

        ctx = DailyRunContext.__new__(DailyRunContext)
        ctx.trade_date = T
        ctx.force = True
        ctx.db_path = prod_db
        ctx.conn = get_connection(prod_db)

        rc = _step2_idempotency(ctx)

        assert rc == -1  # sentinel: continue the run
        ctx.conn.close()
        for book_id in ("none", "n2"):
            conn = get_connection(tmp_path / f"paper_b_{book_id}.db")
            _assert_day_reset(conn, T, NEXT, pre_ids[book_id])
            conn.close()
        assert not (tmp_path / "paper_b_n1.db").exists()
        assert not (tmp_path / "paper_b_n3.db").exists()

        # Production reset semantics unchanged by the wiring.
        prod_conn = get_connection(prod_db)
        assert prod_conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_date = ?", (T,)
        ).fetchone()[0] == 0
        prod_conn.close()


class TestBackfillForceEntry:
    """run_backfill's force loop resets Book B for every replayed day."""

    def test_backfill_force_entry_resets_existing_book_b_dbs(self, tmp_path):
        days = [dt.date(2026, 7, 23), dt.date(2026, 7, 24)]
        prod_db = tmp_path / "paper.db"
        pre_ids = {}
        for d in days:
            _seed_day_state(prod_db, d.isoformat())
            for book_id in ("none", "n2"):
                pre_ids[(book_id, d.isoformat())] = _seed_day_state(
                    tmp_path / f"paper_b_{book_id}.db", d.isoformat(),
                )

        config = {"paper": {"db_path": "paper.db"}}
        with patch("ashare_lab.paper.pipeline.PROJECT_ROOT", tmp_path), \
             patch("ashare_lab.paper.pipeline.load_config", return_value=config), \
             patch("ashare_lab.paper.pipeline.trading_days_between", return_value=days), \
             patch("ashare_lab.paper.pipeline._resolve_prediction_file", return_value=None):
            rc = run_backfill("2026-07-23", "2026-07-24", force=True)

        # Both days skipped for missing predictions, but the reset ran first.
        assert rc == 1
        for book_id in ("none", "n2"):
            conn = get_connection(tmp_path / f"paper_b_{book_id}.db")
            for i, d in enumerate(days):
                # The earliest day's pre-existing order is restored from
                # the journal; later days' are cascaded away by the
                # previous day's reset and re-placed on replay.
                _assert_day_reset(
                    conn, d.isoformat(),
                    (d + dt.timedelta(days=1)).isoformat(),
                    pre_ids[(book_id, d.isoformat())],
                    expect_restored=(i == 0),
                )
            conn.close()


class TestHelperFailOpen:
    """The helper skips missing books and survives a failing one."""

    def test_skips_missing_and_corrupt_book_dbs(self, tmp_path, caplog):
        # n2 is healthy, none is a corrupt file, n1/n3 do not exist.
        pre_n2 = _seed_day_state(tmp_path / "paper_b_n2.db", T)
        (tmp_path / "paper_b_none.db").write_bytes(b"not a sqlite database")

        with caplog.at_level(logging.WARNING):
            _force_reset_book_b(tmp_path, T)

        assert "force-reset failed" in caplog.text
        assert "book_id=none" in caplog.text
        conn = get_connection(tmp_path / "paper_b_n2.db")
        _assert_day_reset(conn, T, NEXT, pre_n2)
        conn.close()
        assert not (tmp_path / "paper_b_n1.db").exists()
        assert not (tmp_path / "paper_b_n3.db").exists()

    def test_covers_all_four_book_ids(self):
        assert _BOOK_B_IDS == ("none", "n1", "n2", "n3")
