"""Unit tests for ashare_lab.paper.ledger -- 9-table SQLite schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper.ledger import (
    bump_carry_days,
    cleanup_old_backups,
    compute_nav,
    delete_expired_cooldowns,
    force_reset_day,
    get_connection,
    get_cooldowns,
    get_latest_cash,
    get_latest_positions,
    hot_backup,
    init_schema,
    insert_order,
    insert_signals,
    insert_trade,
    is_day_settled,
    log_settle_change,
    record_nav,
    record_run,
    set_cooldown,
    snapshot_positions,
    update_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
def conn(db_path: Path) -> sqlite3.Connection:
    c = get_connection(db_path)
    init_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema and connection tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_tables_created(self, conn: sqlite3.Connection) -> None:
        # Exclude sqlite_sequence (auto-created for AUTOINCREMENT)
        row = conn.execute(
            "SELECT count(*) AS cnt FROM sqlite_master "
            "WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchone()
        assert row["cnt"] == 14

    def test_idempotent_schema(self, conn: sqlite3.Connection) -> None:
        # calling init_schema a second time must not raise
        init_schema(conn)
        row = conn.execute(
            "SELECT count(*) AS cnt FROM sqlite_master "
            "WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchone()
        assert row["cnt"] == 14

    def test_wal_mode(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_foreign_keys_enabled(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_cooldowns_columns(self, conn: sqlite3.Connection) -> None:
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(cooldowns)").fetchall()
        }
        assert {"symbol", "cooldown_until", "holding_high"} == cols

    def test_paper_state_initial_row(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM paper_state WHERE key = 'is_soft_reduced'"
        ).fetchone()
        assert row["value"] == "false"

    def test_positions_columns(self, conn: sqlite3.Connection) -> None:
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(positions)").fetchall()
        }
        expected = {
            "trade_date", "symbol", "qty", "avg_cost",
            "market_value", "buy_date", "holding_high", "factor",
        }
        assert expected == cols

    def test_trades_has_trade_date(self, conn: sqlite3.Connection) -> None:
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        assert "trade_date" in cols

    def test_orders_status_check_constraint(
        self, conn: sqlite3.Connection
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            insert_order(
                conn, "2025-01-01", "SH600000", "buy", 100,
                10.0, "INVALID_STATUS", 0, "2025-01-01",
            )


# ---------------------------------------------------------------------------
# Order CRUD tests
# ---------------------------------------------------------------------------

class TestOrderCRUD:
    def test_insert_and_fetch(self, conn: sqlite3.Connection) -> None:
        oid = insert_order(
            conn, "2025-01-02", "SH600000", "buy", 200,
            10.5, "pending", 0, "2025-01-01",
        )
        assert oid > 0
        row = conn.execute(
            "SELECT * FROM orders WHERE id = ?", (oid,)
        ).fetchone()
        assert row["symbol"] == "SH600000"
        assert row["side"] == "buy"
        assert row["target_qty"] == 200
        assert row["status"] == "pending"
        assert row["created_run_date"] == "2025-01-01"
        # created_at is auto-computed and non-empty
        assert len(row["created_at"]) > 0

    def test_update_order_status(self, conn: sqlite3.Connection) -> None:
        oid = insert_order(
            conn, "2025-01-02", "SZ000001", "sell", 100,
            15.0, "pending", 0, "2025-01-02",
        )
        update_order(conn, oid, status="filled", filled_qty=100)
        row = conn.execute(
            "SELECT status, filled_qty FROM orders WHERE id = ?", (oid,)
        ).fetchone()
        assert row["status"] == "filled"
        assert row["filled_qty"] == 100

    def test_bump_carry_days(self, conn: sqlite3.Connection) -> None:
        oid1 = insert_order(
            conn, "2025-01-02", "SH600000", "buy", 100,
            10.0, "carry", 0, "2025-01-01",
        )
        oid2 = insert_order(
            conn, "2025-01-02", "SZ000001", "buy", 200,
            20.0, "carry", 1, "2025-01-01",
        )
        bump_carry_days(conn, [oid1, oid2])
        r1 = conn.execute(
            "SELECT carry_day FROM orders WHERE id = ?", (oid1,)
        ).fetchone()
        r2 = conn.execute(
            "SELECT carry_day FROM orders WHERE id = ?", (oid2,)
        ).fetchone()
        assert r1["carry_day"] == 1
        assert r2["carry_day"] == 2


# ---------------------------------------------------------------------------
# Trade tests
# ---------------------------------------------------------------------------

class TestTrade:
    def test_insert_trade_success(self, conn: sqlite3.Connection) -> None:
        oid = insert_order(
            conn, "2025-01-02", "SH600000", "buy", 100,
            10.0, "filled", 0, "2025-01-02",
        )
        tid = insert_trade(
            conn, oid, "2025-01-02", "SH600000", "buy",
            10.01, 100, 5.0, 0.0, 0.01,
        )
        assert tid > 0

    def test_insert_trade_fk_violation(
        self, conn: sqlite3.Connection
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            insert_trade(
                conn, 9999, "2025-01-02", "SH600000", "buy",
                10.0, 100, 5.0, 0.0, 0.01,
            )


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------

class TestPositions:
    def test_snapshot_and_latest(self, conn: sqlite3.Connection) -> None:
        snapshot_positions(conn, "2025-01-02", {
            "SH600000": {
                "qty": 300,
                "avg_cost": 10.0,
                "market_value": 3000.0,
                "buy_date": "2025-01-01",
                "holding_high": 10.5,
                "factor": 1.0,
            },
        })
        conn.commit()
        pos = get_latest_positions(conn)
        assert "SH600000" in pos
        assert pos["SH600000"]["qty"] == 300
        assert pos["SH600000"]["holding_high"] == 10.5
        assert pos["SH600000"]["factor"] == 1.0
        assert pos["SH600000"]["buy_date"] == "2025-01-01"

    def test_latest_positions_empty(self, conn: sqlite3.Connection) -> None:
        pos = get_latest_positions(conn)
        assert pos == {}


# ---------------------------------------------------------------------------
# NAV tests
# ---------------------------------------------------------------------------

class TestNav:
    def test_record_nav_all_columns(self, conn: sqlite3.Connection) -> None:
        record_nav(
            conn, "2025-01-02", 290000.0, 10000.0, 300000.0,
            299000.0, 300000.0, 3800.0, 6500.0,
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM nav WHERE trade_date = '2025-01-02'"
        ).fetchone()
        assert row["cash"] == 290000.0
        assert row["pre_trade_nav"] == 299000.0
        assert row["post_trade_nav"] == 300000.0
        assert row["benchmark_csi300"] == 3800.0
        assert row["benchmark_csi1000"] == 6500.0

    def test_get_latest_cash_default(self, conn: sqlite3.Connection) -> None:
        assert get_latest_cash(conn) == 300_000.0
        assert get_latest_cash(conn, default_cash=500_000.0) == 500_000.0

    def test_get_latest_cash_after_record(
        self, conn: sqlite3.Connection
    ) -> None:
        record_nav(
            conn, "2025-01-02", 280000.0, 20000.0, 300000.0,
            None, None, None, None,
        )
        conn.commit()
        assert get_latest_cash(conn) == 280_000.0


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignals:
    def test_insert_signals(self, conn: sqlite3.Connection) -> None:
        insert_signals(conn, "2025-01-02", [
            {"symbol": "SH600000", "score": 0.85, "rank": 1},
            {"symbol": "SZ000001", "score": 0.72, "rank": 2},
        ])
        conn.commit()
        rows = conn.execute(
            "SELECT count(*) AS cnt FROM signals WHERE trade_date = '2025-01-02'"
        ).fetchone()
        assert rows["cnt"] == 2


# ---------------------------------------------------------------------------
# Run record tests
# ---------------------------------------------------------------------------

class TestRun:
    def test_record_run(self, conn: sqlite3.Connection) -> None:
        record_run(conn, "2025-01-02", "settled")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM runs WHERE trade_date = '2025-01-02'"
        ).fetchone()
        assert row["status"] == "settled"
        assert len(row["started_at"]) > 0

    def test_is_day_settled_false(self, conn: sqlite3.Connection) -> None:
        assert is_day_settled(conn, "2025-01-02") is False

    def test_is_day_settled_true(self, conn: sqlite3.Connection) -> None:
        record_run(conn, "2025-01-02", "settled")
        conn.commit()
        assert is_day_settled(conn, "2025-01-02") is True

    def test_is_day_settled_excludes_skipped(
        self, conn: sqlite3.Connection
    ) -> None:
        record_run(conn, "2025-01-02", "skipped_stale")
        conn.commit()
        assert is_day_settled(conn, "2025-01-02") is False


# ---------------------------------------------------------------------------
# Force-reset day tests (F-F journal)
# ---------------------------------------------------------------------------

class TestForceResetDay:
    def test_full_reset_with_journal_restore(
        self, conn: sqlite3.Connection
    ) -> None:
        run_d = "2025-01-03"
        prev_d = "2025-01-02"

        # -- Seed a PRE-EXISTING order (created_run_date = prev day) --
        pre_oid = insert_order(
            conn, run_d, "SH600000", "buy", 300,
            10.0, "carry", 1, prev_d,
        )

        # Capture its before-image in the journal
        log_settle_change(conn, run_d, pre_oid)

        # Simulate settlement mutating the pre-existing order
        update_order(conn, pre_oid, status="filled", filled_qty=300, carry_day=2)

        # -- Seed a BORN-in-D order (created_run_date = run_d) --
        born_oid = insert_order(
            conn, run_d, "SZ000001", "sell", 100,
            20.0, "pending", 0, run_d,
        )

        # -- Seed derived data for run_d --
        insert_trade(
            conn, pre_oid, run_d, "SH600000", "buy",
            10.01, 300, 5.0, 0.0, 0.03,
        )
        snapshot_positions(conn, run_d, {
            "SH600000": {
                "qty": 300, "avg_cost": 10.0,
                "market_value": 3000.0, "buy_date": prev_d,
            },
        })
        record_nav(
            conn, run_d, 290000.0, 3000.0, 293000.0,
            None, None, None, None,
        )
        insert_signals(conn, run_d, [
            {"symbol": "SH600000", "score": 0.9, "rank": 1},
        ])
        record_run(conn, run_d, "settled")
        conn.commit()

        # -- Set a cooldown to verify it survives --
        set_cooldown(conn, "SH600001", "2025-01-10", 12.0)
        conn.commit()

        # -- Execute force_reset_day --
        force_reset_day(conn, run_d)

        # Verify: pre-existing order RESTORED to carry_day=1, status=carry
        pre_row = conn.execute(
            "SELECT status, carry_day, filled_qty FROM orders WHERE id = ?",
            (pre_oid,),
        ).fetchone()
        assert pre_row["status"] == "carry"
        assert pre_row["carry_day"] == 1
        assert pre_row["filled_qty"] == 0

        # Verify: born-in-D order DELETED
        born_row = conn.execute(
            "SELECT 1 FROM orders WHERE id = ?", (born_oid,),
        ).fetchone()
        assert born_row is None

        # Verify: derived data all gone
        for tbl in ("trades", "positions", "nav", "signals", "runs"):
            cnt = conn.execute(
                f"SELECT count(*) AS c FROM {tbl} WHERE trade_date = ?",
                (run_d,),
            ).fetchone()["c"]
            assert cnt == 0, f"{tbl} still has rows for {run_d}"

        # Verify: order_settle_log rows for run_d gone
        log_cnt = conn.execute(
            "SELECT count(*) AS c FROM order_settle_log WHERE run_date = ?",
            (run_d,),
        ).fetchone()["c"]
        assert log_cnt == 0

        # Verify: cooldowns untouched
        cds = get_cooldowns(conn)
        assert "SH600001" in cds


# ---------------------------------------------------------------------------
# Log settle change (first-touch-only)
# ---------------------------------------------------------------------------

class TestLogSettleChange:
    def test_first_touch_only(self, conn: sqlite3.Connection) -> None:
        oid = insert_order(
            conn, "2025-01-03", "SH600000", "buy", 200,
            10.0, "pending", 0, "2025-01-02",
        )
        # First call: captures pending/0/0
        log_settle_change(conn, "2025-01-03", oid)
        # Mutate the order
        update_order(conn, oid, status="filled", filled_qty=200, carry_day=1)
        # Second call: should be ignored (OR IGNORE)
        log_settle_change(conn, "2025-01-03", oid)
        conn.commit()

        row = conn.execute(
            "SELECT prev_status, prev_filled_qty, prev_carry_day "
            "FROM order_settle_log "
            "WHERE run_date = '2025-01-03' AND order_id = ?",
            (oid,),
        ).fetchone()
        # Still holds the FIRST pre-image
        assert row["prev_status"] == "pending"
        assert row["prev_filled_qty"] == 0
        assert row["prev_carry_day"] == 0


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------

class TestCooldowns:
    def test_roundtrip(self, conn: sqlite3.Connection) -> None:
        set_cooldown(conn, "SH600000", "2025-01-10", 12.5)
        conn.commit()
        cds = get_cooldowns(conn)
        assert "SH600000" in cds
        assert cds["SH600000"]["cooldown_until"] == "2025-01-10"
        assert cds["SH600000"]["holding_high"] == 12.5

    def test_delete_expired(self, conn: sqlite3.Connection) -> None:
        set_cooldown(conn, "SH600000", "2025-01-05", 10.0)
        set_cooldown(conn, "SZ000001", "2025-01-15", 20.0)
        conn.commit()
        delete_expired_cooldowns(conn, "2025-01-10")
        conn.commit()
        cds = get_cooldowns(conn)
        assert "SH600000" not in cds
        assert "SZ000001" in cds


# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------

class TestBackup:
    def test_hot_backup(
        self, db_path: Path, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        # Write data so backup is non-trivial
        record_run(conn, "2025-01-02", "settled")
        conn.commit()

        backup_dest = tmp_path / "backups" / "paper_2025-01-02.db"
        hot_backup(db_path, backup_dest)

        assert backup_dest.exists()
        # Verify the backup is a valid DB with the runs row
        bc = sqlite3.connect(str(backup_dest))
        bc.row_factory = sqlite3.Row
        row = bc.execute(
            "SELECT status FROM runs WHERE trade_date = '2025-01-02'"
        ).fetchone()
        bc.close()
        assert row["status"] == "settled"

    def test_cleanup_old_backups(self, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        from datetime import date, timedelta

        today = date.today()
        old = today - timedelta(days=10)
        recent = today - timedelta(days=2)
        (backup_dir / f"paper_{old.isoformat()}.db").touch()
        (backup_dir / f"paper_{recent.isoformat()}.db").touch()

        cleanup_old_backups(backup_dir, retention_days=7)

        remaining = list(backup_dir.glob("paper_*.db"))
        assert len(remaining) == 1
        assert recent.isoformat() in remaining[0].name


# ---------------------------------------------------------------------------
# compute_nav
# ---------------------------------------------------------------------------

class TestComputeNav:
    def test_basic(self) -> None:
        positions = {
            "A": {"qty": 100, "avg_cost": 10.0, "market_value": 1000.0},
            "B": {"qty": 200, "avg_cost": 5.0, "market_value": 1000.0},
        }
        prices = {
            "A": {"close": 11.0},
            "B": {"close": 6.0},
        }
        nav = compute_nav(positions, prices, 50000.0)
        assert nav == pytest.approx(50000.0 + 100 * 11.0 + 200 * 6.0)

    def test_missing_price_fallback(self) -> None:
        positions = {
            "A": {"qty": 100, "avg_cost": 10.0, "market_value": 1050.0},
        }
        # No price for A -- falls back to market_value / qty
        nav = compute_nav(positions, {}, 50000.0)
        assert nav == pytest.approx(50000.0 + 1050.0)

    def test_nan_price_fallback(self) -> None:
        positions = {
            "A": {"qty": 100, "avg_cost": 10.0, "market_value": 1000.0},
        }
        prices = {"A": {"close": float("nan")}}
        nav = compute_nav(positions, prices, 50000.0)
        # Falls back to market_value / qty = 10.0 per share
        assert nav == pytest.approx(50000.0 + 1000.0)

    def test_zero_qty_skipped(self) -> None:
        positions = {
            "A": {"qty": 0, "avg_cost": 10.0, "market_value": 0.0},
        }
        nav = compute_nav(positions, {}, 50000.0)
        assert nav == pytest.approx(50000.0)
