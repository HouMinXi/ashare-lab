"""SQLite 14-table ledger for the paper trading engine.

Schema: runs, orders, trades, positions, nav, signals,
cooldowns, paper_state, order_settle_log, reports,
sentiment_scores, sentiment_events, pipeline_runs,
graduation_status.

All queries use parameterised placeholders (?).  WAL mode and
foreign keys are enabled on every connection.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL -- 14 tables
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    trade_date TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date       TEXT    NOT NULL,
    symbol           TEXT    NOT NULL,
    side             TEXT    NOT NULL,
    target_qty       INTEGER NOT NULL,
    filled_qty       INTEGER NOT NULL DEFAULT 0,
    price            REAL,
    status           TEXT    NOT NULL
                     CHECK(status IN (
                         'pending','filled','partial',
                         'carry','cancelled','lot_skip'
                     )),
    carry_day        INTEGER NOT NULL DEFAULT 0,
    reset_count      INTEGER NOT NULL DEFAULT 0,
    created_run_date TEXT    NOT NULL,
    source           TEXT    NOT NULL DEFAULT 'signal',
    created_at       TEXT    NOT NULL
);

-- Migration: add reset_count if missing (idempotent)
-- SQLite does not support IF NOT EXISTS for ALTER TABLE,
-- so init_schema() handles this via try/except.

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     INTEGER NOT NULL,
    trade_date   TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    fill_price   REAL    NOT NULL,
    fill_qty     INTEGER NOT NULL,
    commission   REAL    NOT NULL,
    stamp        REAL    NOT NULL,
    transfer_fee REAL    NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE IF NOT EXISTS positions (
    trade_date   TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    qty          INTEGER NOT NULL,
    avg_cost     REAL    NOT NULL,
    market_value REAL    NOT NULL,
    buy_date     TEXT    NOT NULL DEFAULT '',
    holding_high REAL    NOT NULL DEFAULT 0.0,
    factor       REAL    NOT NULL DEFAULT 1.0,
    PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS nav (
    trade_date       TEXT PRIMARY KEY,
    cash             REAL NOT NULL,
    market_value     REAL NOT NULL,
    total_nav        REAL NOT NULL,
    pre_trade_nav    REAL,
    post_trade_nav   REAL,
    hedge_value      REAL DEFAULT 0.0,
    equity_value     REAL DEFAULT 0.0,
    benchmark_csi300 REAL,
    benchmark_csi1000 REAL
);

CREATE TABLE IF NOT EXISTS signals (
    trade_date TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    score      REAL    NOT NULL,
    rank       INTEGER NOT NULL,
    PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS cooldowns (
    symbol        TEXT PRIMARY KEY,
    cooldown_until TEXT NOT NULL,
    holding_high  REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS paper_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_settle_log (
    run_date        TEXT    NOT NULL,
    order_id        INTEGER NOT NULL,
    prev_status     TEXT    NOT NULL,
    prev_filled_qty INTEGER NOT NULL,
    prev_carry_day  INTEGER NOT NULL,
    prev_price      REAL,
    PRIMARY KEY (run_date, order_id)
);

CREATE TABLE IF NOT EXISTS reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT NOT NULL,
    mode            TEXT NOT NULL,
    report_text     TEXT NOT NULL,
    delivered_via   TEXT,
    delivery_status TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_report_td_created
    ON reports(trade_date, created_at DESC, id DESC);

-- scored_at/created_at use SQLite CURRENT_TIMESTAMP (YYYY-MM-DD HH:MM:SS UTC);
-- these columns are for cache expiry and debugging, not cross-table joins
CREATE TABLE IF NOT EXISTS sentiment_scores (
    trade_date TEXT    NOT NULL,
    layer      TEXT    NOT NULL,
    target     TEXT    NOT NULL,
    score      REAL    NOT NULL,
    news_count INTEGER NOT NULL DEFAULT 0,
    scored_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, layer, target)
);

CREATE TABLE IF NOT EXISTS sentiment_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date  TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    layer       TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    score       REAL,
    detail      TEXT,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date       TEXT NOT NULL,
    status           TEXT NOT NULL
                     CHECK(status IN ('success','error','stale')),
    duration_s       REAL NOT NULL,
    error_msg        TEXT,
    predictions_date TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_td_created
    ON pipeline_runs(trade_date, created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS graduation_status (
    id            INTEGER PRIMARY KEY CHECK(id = 1),
    graduated_at  TEXT,
    notified_at   TEXT
);

CREATE TABLE IF NOT EXISTS hedge_state (
    trade_date        TEXT PRIMARY KEY,
    active            INTEGER NOT NULL DEFAULT 0,
    drawdown_pct      REAL NOT NULL DEFAULT 0.0,
    equity_target_pct REAL NOT NULL DEFAULT 1.0,
    hedge_target_pct  REAL NOT NULL DEFAULT 0.0,
    days_in_hedge     INTEGER NOT NULL DEFAULT 0,
    days_in_recovery  INTEGER NOT NULL DEFAULT 0,
    peak_nav          REAL NOT NULL DEFAULT 0.0,
    leg_json          TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_orders_date_status
    ON orders (created_run_date, status);
CREATE INDEX IF NOT EXISTS idx_trades_date
    ON trades (trade_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_date
    ON sentiment_scores (trade_date);
"""

_INITIAL_STATE_SQL = """\
INSERT OR IGNORE INTO paper_state (key, value) VALUES ('is_soft_reduced', 'false');
"""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL, FK enforcement, and Row factory."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # No journal_size_limit: large backfills can exceed 64MB WAL.
    # SQLite auto-checkpoints on close; pipeline mutex prevents concurrent writes.
    return conn


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def init_schema(conn: sqlite3.Connection) -> None:
    """Create all 14 tables (idempotent) and seed initial state."""
    # Migration: reports/pipeline_runs from trade_date PK to auto-increment id.
    # Must run BEFORE executescript because _SCHEMA_SQL now references the
    # `id` column in CREATE INDEX statements -- on old-schema databases
    # executescript would fail with "no such column: id" before the migration
    # is ever reached if we left it after.
    _migrate_audit_table(conn, "reports")
    _migrate_audit_table(conn, "pipeline_runs")

    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO paper_state (key, value) "
        "VALUES ('is_soft_reduced', 'false')"
    )

    # Idempotent migrations for databases created before the hedge sleeve
    # additions.  SQLite raises OperationalError when a column already exists.
    try:
        conn.execute(
            "ALTER TABLE orders ADD COLUMN source TEXT NOT NULL DEFAULT 'signal'"
        )
    except sqlite3.OperationalError:
        pass

    for col in ("hedge_value", "equity_value"):
        try:
            conn.execute(
                f"ALTER TABLE nav ADD COLUMN {col} REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass

    # reset_count: tracks limit-down reset attempts per order
    try:
        conn.execute(
            "ALTER TABLE orders ADD COLUMN reset_count INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass

    # Phase 9 -- 09-03: suspension tracking columns
    try:
        conn.execute(
            "ALTER TABLE orders ADD COLUMN suspension_carry_day INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute(
            "ALTER TABLE orders ADD COLUMN cancel_reason TEXT DEFAULT NULL"
        )
    except sqlite3.OperationalError:
        pass

    conn.commit()


def _migrate_audit_table(conn: sqlite3.Connection, table: str) -> None:
    """Recreate audit table with auto-increment id if it has the old schema.

    Uses explicit BEGIN IMMEDIATE so all DDL is atomic -- SQLite DDL is
    transactional (unlike MySQL/PostgreSQL), but Python's executescript()
    implicitly commits, breaking the transaction.  We use execute() only.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None:
        return
    create_sql = row[0]
    if "PRIMARY KEY" in create_sql and "AUTOINCREMENT" not in create_sql:
        # Old schema: trade_date is PRIMARY KEY.  Migrate atomically.
        backup = f"{table}_old"
        old_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        col_list = ", ".join(old_cols)
        # Get the CREATE TABLE statement from _SCHEMA_SQL for this table
        new_create = _extract_create_table(_SCHEMA_SQL, table)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"DROP TABLE IF EXISTS {backup}")
            conn.execute(f"ALTER TABLE {table} RENAME TO {backup}")
            conn.execute(new_create)
            conn.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {backup}")
            conn.execute(f"DROP TABLE IF EXISTS {backup}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _extract_create_table(schema_sql: str, table: str) -> str:
    """Extract the CREATE TABLE statement for *table* from a multi-statement schema.

    Uses balanced-parentheses parsing instead of regex, so nested
    parens (CHECK constraints, sub-selects) are handled correctly.
    """
    marker = f"CREATE TABLE IF NOT EXISTS {table}"
    start = schema_sql.find(marker)
    if start == -1:
        raise ValueError(f"CREATE TABLE for {table} not found in schema")
    # Find the opening paren
    paren_start = schema_sql.find("(", start)
    if paren_start == -1:
        raise ValueError(f"No opening paren for {table}")
    # Walk balanced parens to find the closing one
    depth = 0
    for i in range(paren_start, len(schema_sql)):
        if schema_sql[i] == "(":
            depth += 1
        elif schema_sql[i] == ")":
            depth -= 1
            if depth == 0:
                return schema_sql[start : i + 1] + ";"
    raise ValueError(f"Unbalanced parens in CREATE TABLE for {table}")


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def insert_report(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str,
    report_text: str,
    delivered_via: str | None,
    delivery_status: str,
) -> None:
    """Append a report row (never overwrites history).

    *created_at* is computed internally (UTC ISO-8601).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO reports "
        "(trade_date, mode, report_text, delivered_via, delivery_status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (trade_date, mode, report_text, delivered_via, delivery_status, created_at),
    )


def insert_pipeline_run(
    conn: sqlite3.Connection,
    trade_date: str,
    status: str,
    duration_s: float,
    error_msg: str | None,
    predictions_date: str | None,
) -> None:
    """Append a pipeline run record (never overwrites history).

    *created_at* is computed internally (UTC ISO-8601).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO pipeline_runs "
        "(trade_date, status, duration_s, error_msg, predictions_date, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (trade_date, status, duration_s, error_msg, predictions_date, created_at),
    )


def get_report(conn: sqlite3.Connection, trade_date: str) -> sqlite3.Row | None:
    """Return the latest report row for *trade_date*."""
    return conn.execute(
        "SELECT * FROM reports WHERE trade_date = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (trade_date,),
    ).fetchone()


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------

def is_day_settled(conn: sqlite3.Connection, trade_date: str) -> bool:
    """Return True if *trade_date* has a 'settled' run record.

    Intentionally excludes 'skipped_stale' so the pipeline can auto-retry
    stale days when fresh data arrives.
    """
    row = conn.execute(
        "SELECT 1 FROM runs WHERE trade_date = ? AND status = 'settled'",
        (trade_date,),
    ).fetchone()
    return row is not None


def force_reset_day(conn: sqlite3.Connection, trade_date: str) -> None:
    """Idempotently undo a settled day so it can be re-run.

    All steps execute in ONE transaction, in strict order:
    1. Restore pre-existing orders from the before-image journal.
    2. Delete trades for this run (FK child before parent).
    3. Delete orders born in this run (created_run_date = trade_date).
    4. Delete the before-image log entries for this run.
    5. Delete derived rows (positions, nav, signals, runs).
    6. Leave cooldowns and paper_state untouched (converge on re-run).
    """
    with conn:
        # 1 -- restore pre-existing orders from the journal
        rows = conn.execute(
            "SELECT order_id, prev_status, prev_filled_qty, "
            "       prev_carry_day, prev_price "
            "FROM order_settle_log WHERE run_date = ?",
            (trade_date,),
        ).fetchall()
        for r in rows:
            conn.execute(
                "UPDATE orders "
                "SET status = ?, filled_qty = ?, carry_day = ?, price = ? "
                "WHERE id = ? AND created_run_date < ?",
                (
                    r["prev_status"],
                    r["prev_filled_qty"],
                    r["prev_carry_day"],
                    r["prev_price"],
                    r["order_id"],
                    trade_date,
                ),
            )

        # 2 -- delete trades (FK child first)
        conn.execute(
            "DELETE FROM trades WHERE trade_date = ?", (trade_date,)
        )

        # 3 -- delete orders born in this run
        conn.execute(
            "DELETE FROM orders WHERE created_run_date = ?", (trade_date,)
        )

        # 4 -- delete journal entries
        conn.execute(
            "DELETE FROM order_settle_log WHERE run_date = ?", (trade_date,)
        )

        # 5 -- delete derived snapshots
        for tbl in ("positions", "nav", "signals", "runs"):
            conn.execute(
                f"DELETE FROM {tbl} WHERE trade_date = ?", (trade_date,)
            )

        # 6 -- delete sentiment data (avoids duplicate veto events on re-run)
        conn.execute(
            "DELETE FROM sentiment_scores WHERE trade_date = ?",
            (trade_date,),
        )
        conn.execute(
            "DELETE FROM sentiment_events WHERE trade_date = ?",
            (trade_date,),
        )


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def insert_order(
    conn: sqlite3.Connection,
    trade_date: str,
    symbol: str,
    side: str,
    target_qty: int,
    price: float | None,
    status: str,
    carry_day: int,
    created_run_date: str,
    source: str = "signal",
) -> int:
    """Insert an order row; return the new rowid.

    *created_at* is computed internally (UTC ISO-8601).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO orders "
        "(trade_date, symbol, side, target_qty, price, status, "
        " carry_day, created_run_date, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trade_date, symbol, side, target_qty, price,
            status, carry_day, created_run_date, source, created_at,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def update_order(conn: sqlite3.Connection, order_id: int, **fields: object) -> None:
    """Update arbitrary order fields by *order_id*.

    Accepted keys: status, filled_qty, carry_day, price, reset_count,
    suspension_carry_day, cancel_reason.
    """
    allowed = {
        "status", "filled_qty", "carry_day", "price", "reset_count",
        "suspension_carry_day", "cancel_reason",
    }
    to_set = {k: v for k, v in fields.items() if k in allowed}
    if not to_set:
        return
    set_clause = ", ".join(f"{k} = ?" for k in to_set)
    vals = list(to_set.values()) + [order_id]
    conn.execute(
        f"UPDATE orders SET {set_clause} WHERE id = ?", vals
    )


def log_settle_change(
    conn: sqlite3.Connection, run_date: str, order_id: int
) -> None:
    """Capture the before-image of an order (first-touch-only)."""
    conn.execute(
        "INSERT OR IGNORE INTO order_settle_log "
        "(run_date, order_id, prev_status, prev_filled_qty, "
        " prev_carry_day, prev_price) "
        "SELECT ?, id, status, filled_qty, carry_day, price "
        "FROM orders WHERE id = ?",
        (run_date, order_id),
    )


def bump_carry_days(conn: sqlite3.Connection, order_ids: list[int]) -> None:
    """Increment carry_day for each order id in the list."""
    for oid in order_ids:
        conn.execute(
            "UPDATE orders SET carry_day = carry_day + 1 WHERE id = ?",
            (oid,),
        )


def bump_suspension_carry_days(
    conn: sqlite3.Connection, order_ids: list[int],
) -> None:
    """Increment suspension_carry_day for each order id in the list."""
    for oid in order_ids:
        conn.execute(
            "UPDATE orders SET suspension_carry_day = suspension_carry_day + 1 WHERE id = ?",
            (oid,),
        )


# ---------------------------------------------------------------------------
# Trade helpers
# ---------------------------------------------------------------------------

def insert_trade(
    conn: sqlite3.Connection,
    order_id: int,
    trade_date: str,
    symbol: str,
    side: str,
    fill_price: float,
    fill_qty: int,
    commission: float,
    stamp: float,
    transfer_fee: float,
) -> int:
    """Insert a trade row; return the new rowid."""
    cur = conn.execute(
        "INSERT INTO trades "
        "(order_id, trade_date, symbol, side, fill_price, fill_qty, "
        " commission, stamp, transfer_fee) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            order_id, trade_date, symbol, side,
            fill_price, fill_qty, commission, stamp, transfer_fee,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def snapshot_positions(
    conn: sqlite3.Connection,
    trade_date: str,
    positions_dict: dict[str, dict],
) -> None:
    """Write (or overwrite) daily position snapshot."""
    for symbol, p in positions_dict.items():
        conn.execute(
            "INSERT OR REPLACE INTO positions "
            "(trade_date, symbol, qty, avg_cost, market_value, "
            " buy_date, holding_high, factor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trade_date,
                symbol,
                p["qty"],
                p["avg_cost"],
                p["market_value"],
                p.get("buy_date", ""),
                p.get("holding_high", 0.0),
                p.get("factor", 1.0),
            ),
        )


def get_latest_positions(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return positions from the most recent snapshot date.

    Empty dict on Day 1 (no rows).
    """
    rows = conn.execute(
        "SELECT symbol, qty, avg_cost, market_value, "
        "       buy_date, holding_high, factor "
        "FROM positions "
        "WHERE trade_date = (SELECT MAX(trade_date) FROM positions)"
    ).fetchall()
    result: dict[str, dict] = {}
    for r in rows:
        result[r["symbol"]] = {
            "qty": r["qty"],
            "avg_cost": r["avg_cost"],
            "market_value": r["market_value"],
            "buy_date": r["buy_date"],
            "holding_high": r["holding_high"],
            "factor": r["factor"],
        }
    return result


# ---------------------------------------------------------------------------
# NAV helpers
# ---------------------------------------------------------------------------

def record_nav(
    conn: sqlite3.Connection,
    trade_date: str,
    cash: float,
    market_value: float,
    total_nav: float,
    pre_trade_nav: float | None,
    post_trade_nav: float | None,
    benchmark_csi300: float | None,
    benchmark_csi1000: float | None,
    hedge_value: float = 0.0,
    equity_value: float = 0.0,
) -> None:
    """Write (or overwrite) the daily NAV row."""
    conn.execute(
        "INSERT OR REPLACE INTO nav "
        "(trade_date, cash, market_value, total_nav, "
        " pre_trade_nav, post_trade_nav, hedge_value, equity_value, "
        " benchmark_csi300, benchmark_csi1000) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trade_date, cash, market_value, total_nav,
            pre_trade_nav, post_trade_nav, hedge_value, equity_value,
            benchmark_csi300, benchmark_csi1000,
        ),
    )


def get_latest_cash(
    conn: sqlite3.Connection, default_cash: float = 300_000.0
) -> float:
    """Return cash from the most recent NAV row.

    Returns *default_cash* on Day 1 (no rows).
    """
    row = conn.execute(
        "SELECT cash FROM nav "
        "WHERE trade_date = (SELECT MAX(trade_date) FROM nav)"
    ).fetchone()
    if row is None:
        return default_cash
    return float(row["cash"])


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def insert_signals(
    conn: sqlite3.Connection,
    trade_date: str,
    signals_list: list[dict],
) -> None:
    """Write (or overwrite) daily signal rows."""
    for s in signals_list:
        conn.execute(
            "INSERT OR REPLACE INTO signals "
            "(trade_date, symbol, score, rank) "
            "VALUES (?, ?, ?, ?)",
            (trade_date, s["symbol"], s["score"], s["rank"]),
        )


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def record_run(
    conn: sqlite3.Connection, trade_date: str, status: str
) -> None:
    """Write (or overwrite) a run record.

    *started_at* is computed internally (UTC ISO-8601).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO runs (trade_date, status, started_at) "
        "VALUES (?, ?, ?)",
        (trade_date, status, started_at),
    )


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------

def get_cooldowns(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return all active cooldowns as {symbol: {cooldown_until, holding_high}}."""
    rows = conn.execute(
        "SELECT symbol, cooldown_until, holding_high FROM cooldowns"
    ).fetchall()
    return {
        r["symbol"]: {
            "cooldown_until": r["cooldown_until"],
            "holding_high": float(r["holding_high"]),
        }
        for r in rows
    }


def set_cooldown(
    conn: sqlite3.Connection,
    symbol: str,
    cooldown_until: str,
    holding_high: float,
) -> None:
    """Insert or update a cooldown entry."""
    conn.execute(
        "INSERT OR REPLACE INTO cooldowns "
        "(symbol, cooldown_until, holding_high) VALUES (?, ?, ?)",
        (symbol, cooldown_until, holding_high),
    )


def delete_expired_cooldowns(
    conn: sqlite3.Connection, trade_date: str
) -> None:
    """Remove cooldowns whose expiry is strictly before *trade_date*."""
    conn.execute(
        "DELETE FROM cooldowns WHERE cooldown_until < ?", (trade_date,)
    )


# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------

def hot_backup(db_path: Path, backup_path: Path) -> None:
    """Create a consistent backup using the sqlite3 backup API.

    Opens a separate read-only source connection so the caller's
    connection is not disturbed.
    """
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def cleanup_old_backups(backup_dir: Path, retention_days: int) -> None:
    """Delete backup files older than *retention_days*.

    Skips backups/golden/ directory -- golden backups are immutable
    monthly snapshots that must never be cleaned up by retention policy.
    """
    cutoff = date.today() - timedelta(days=retention_days)
    golden_dir = backup_dir / "golden"
    golden_resolved = golden_dir.resolve() if golden_dir.exists() else None
    for f in backup_dir.glob("paper_*.db"):
        # Skip golden backups (check if file is under golden/)
        if golden_resolved is not None:
            try:
                f.resolve().relative_to(golden_resolved)
                continue  # is inside golden/
            except ValueError:
                pass  # not inside golden/
        # Expected filename: paper_YYYY-MM-DD.db
        stem = f.stem  # paper_YYYY-MM-DD
        try:
            date_str = stem[len("paper_"):]
            file_date = date.fromisoformat(date_str)
        except (ValueError, IndexError):
            continue
        if file_date < cutoff:
            f.unlink()


# ---------------------------------------------------------------------------
# Integrity check and enhanced backup (Phase 9 -- 09-01)
# ---------------------------------------------------------------------------


@dataclass
class IntegrityResult:
    """Outcome of a PRAGMA integrity_check."""

    is_healthy: bool
    detail: str
    check_duration_ms: float


@dataclass
class BackupResult:
    """Outcome of a backup operation."""

    success: bool
    backup_path: Path | None = None
    sha256: str | None = None
    size_bytes: int = 0
    integrity_verified: bool = False


def check_db_integrity(db_path: Path) -> IntegrityResult:
    """Run PRAGMA integrity_check on a SEPARATE connection.

    Uses a dedicated connection to avoid false negatives from a corrupt
    working connection.  On failure, attempts WAL checkpoint repair
    (PRAGMA wal_checkpoint(TRUNCATE)) and re-checks.

    Expected overhead: ~3.5ms per check.
    """
    start = time.monotonic()
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.DatabaseError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return IntegrityResult(
            is_healthy=False,
            detail=f"cannot open database: {exc}",
            check_duration_ms=round(elapsed_ms, 2),
        )

    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        detail = row[0] if row else "no result"
        is_healthy = detail == "ok"

        if not is_healthy:
            logger.warning(
                "DB integrity check failed: %s -- attempting WAL repair",
                detail,
            )
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
            row2 = conn.execute("PRAGMA integrity_check").fetchone()
            detail2 = row2[0] if row2 else "no result"
            if detail2 == "ok":
                is_healthy = True
                detail = "ok (after WAL checkpoint repair)"
            else:
                detail = detail2
    except sqlite3.DatabaseError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return IntegrityResult(
            is_healthy=False,
            detail=f"integrity check error: {exc}",
            check_duration_ms=round(elapsed_ms, 2),
        )
    finally:
        conn.close()

    elapsed_ms = (time.monotonic() - start) * 1000
    return IntegrityResult(
        is_healthy=is_healthy,
        detail=detail,
        check_duration_ms=round(elapsed_ms, 2),
    )


def hot_backup_with_integrity(
    db_path: Path, backup_path: Path,
) -> BackupResult:
    """Create a backup with pre-check integrity gate and SHA-256 verification.

    1. Run check_db_integrity on the source DB.
    2. If unhealthy: skip backup, log CRITICAL, return failure.
    3. If healthy: run sqlite3 backup, compute SHA-256, verify hash.
    """
    integrity = check_db_integrity(db_path)
    if not integrity.is_healthy:
        logger.critical(
            "Backup SKIPPED: DB integrity check failed (%s). "
            "Database may be corrupt. DO NOT overwrite good backup.",
            integrity.detail,
        )
        return BackupResult(success=False)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # SHA-256 verification (incremental to avoid loading entire file)
    h = hashlib.sha256()
    size_bytes = 0
    with backup_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size_bytes += len(chunk)
    sha256 = h.hexdigest()

    logger.info(
        "Backup created: %s (%d bytes, sha256=%s)",
        backup_path, size_bytes, sha256[:16],
    )

    return BackupResult(
        success=True,
        backup_path=backup_path,
        sha256=sha256,
        size_bytes=size_bytes,
        integrity_verified=True,
    )


def create_golden_backup(db_path: Path, backup_dir: Path) -> Path | None:
    """Create a monthly golden backup (immutable, chmod 444).

    Checks if backups/golden/YYYY-MM.db exists for the current month.
    If not: copy DB with integrity check, chmod 444.
    Never overwrites an existing golden backup.
    Returns the path if created, None if already exists or integrity failed.
    """
    golden_dir = backup_dir / "golden"
    golden_path = golden_dir / f"{date.today().strftime('%Y-%m')}.db"

    if golden_path.exists():
        logger.debug("Golden backup already exists: %s", golden_path)
        return None

    integrity = check_db_integrity(db_path)
    if not integrity.is_healthy:
        logger.warning(
            "Golden backup SKIPPED: DB integrity check failed (%s)",
            integrity.detail,
        )
        return None

    golden_dir.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(golden_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # Make immutable
    golden_path.chmod(0o444)
    logger.info("Golden backup created: %s", golden_path)
    return golden_path


def find_best_backup(backup_dir: Path, target_date: str | None = None) -> Path | None:
    """Find the best available backup by priority.

    Priority:
    1. Daily backup for target_date (if provided)
    2. Most recent daily backup passing integrity check
    3. Golden backup for current month (last resort)

    Returns the path to the best backup, or None if nothing found.
    """
    golden_dir = backup_dir / "golden"

    # Priority 1: exact date match
    if target_date:
        exact = backup_dir / f"paper_{target_date}.db"
        if exact.exists():
            check = check_db_integrity(exact)
            if check.is_healthy:
                return exact

    # Priority 2: most recent daily backup passing integrity
    daily_backups = sorted(
        backup_dir.glob("paper_*.db"),
        key=lambda f: f.name,
        reverse=True,
    )
    for bp in daily_backups:
        check = check_db_integrity(bp)
        if check.is_healthy:
            return bp

    # Priority 3: golden backup
    if golden_dir.exists():
        golden_files = sorted(golden_dir.glob("*.db"), reverse=True)
        for gp in golden_files:
            check = check_db_integrity(gp)
            if check.is_healthy:
                return gp

    return None


def restore_from_backup(db_path: Path, backup_dir: Path) -> bool:
    """Restore database from the best available backup.

    Uses find_best_backup() to locate a healthy backup, then restores
    via sqlite3 backup API.
    Returns True if restore succeeded.
    """
    best = find_best_backup(backup_dir)
    if best is None:
        logger.error("No healthy backup found for restore")
        return False

    logger.info("Restoring from backup: %s", best)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(best))
    dst = sqlite3.connect(str(db_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    logger.info("Restore completed from %s", best)
    return True


# ---------------------------------------------------------------------------
# Shared NAV computation (pure function, no DB access)
# ---------------------------------------------------------------------------

def compute_nav(
    positions: dict[str, dict],
    prices: dict[str, dict],
    cash: float,
) -> float:
    """Compute NAV from positions, prices, and cash.

    For each symbol, uses the close price from *prices* if available
    and finite.  Falls back to deriving a unit price from the position's
    own data when the close is missing or non-finite.
    """
    total = cash
    for sym, pos in positions.items():
        qty = pos["qty"]
        if qty == 0:
            continue
        close = prices.get(sym, {}).get("close")
        if close is None or (isinstance(close, float) and math.isnan(close)):
            mv = pos.get("market_value", qty * pos["avg_cost"])
            close = mv / qty if qty > 0 else pos["avg_cost"]
        total += qty * close
    return total
