"""SQLite 14-table ledger for the paper trading engine.

Schema: runs, orders, trades, positions, nav, signals,
cooldowns, paper_state, order_settle_log, reports,
sentiment_scores, sentiment_events, pipeline_runs,
graduation_status.

All queries use parameterised placeholders (?).  WAL mode and
foreign keys are enabled on every connection.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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
    created_run_date TEXT    NOT NULL,
    source           TEXT    NOT NULL DEFAULT 'signal',
    created_at       TEXT    NOT NULL
);

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
    trade_date      TEXT PRIMARY KEY,
    mode            TEXT NOT NULL,
    report_text     TEXT NOT NULL,
    delivered_via   TEXT,
    delivery_status TEXT,
    created_at      TEXT NOT NULL
);

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
    trade_date       TEXT PRIMARY KEY,
    status           TEXT NOT NULL
                     CHECK(status IN ('success','error','stale')),
    duration_s       REAL NOT NULL,
    error_msg        TEXT,
    predictions_date TEXT,
    created_at       TEXT NOT NULL
);

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
    conn.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
    return conn


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def init_schema(conn: sqlite3.Connection) -> None:
    """Create all 14 tables (idempotent) and seed initial state."""
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

    conn.commit()


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
    """Insert or replace a report row.
    
    *created_at* is computed internally (UTC ISO-8601).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO reports "
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
    """Insert or replace a pipeline run record.

    *created_at* is computed internally (UTC ISO-8601).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs "
        "(trade_date, status, duration_s, error_msg, predictions_date, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (trade_date, status, duration_s, error_msg, predictions_date, created_at),
    )


def get_report(conn: sqlite3.Connection, trade_date: str) -> sqlite3.Row | None:
    """Return the report row for *trade_date*."""
    return conn.execute(
        "SELECT * FROM reports WHERE trade_date = ?", (trade_date,)
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

    Accepted keys: status, filled_qty, carry_day, price.
    """
    allowed = {"status", "filled_qty", "carry_day", "price"}
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
    """Delete backup files older than *retention_days*."""
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=retention_days)
    for f in backup_dir.glob("paper_*.db"):
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
