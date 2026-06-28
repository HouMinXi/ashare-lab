"""Shared pytest fixtures for paper engine acceptance tests.

Provides synthetic prices, signals, DB connections, and config dicts
so acceptance tests run deterministically with no external dependencies.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.paper.ledger import get_connection, init_schema


# -- Database fixture (real temp file, not :memory:) -----------------------

@pytest.fixture()
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Create a WAL-mode SQLite DB in a temp directory."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_schema(conn)
    yield conn
    conn.close()


# -- Config fixtures -------------------------------------------------------

@pytest.fixture()
def paper_config() -> dict:
    """Minimal paper config matching baseline.yaml paper: section."""
    return {
        "carry_days": 3,
        "volume_participation_pct": 0.05,
        "topk": 15,
        "n_drop": 1,
        "initial_cash": 300_000,
        "backup_retention_days": 7,
        "slippage": 0.001,
        "listing_min_days": 60,
        "liquidity_min_turnover": 50_000_000,
        "predictions_dir": "predictions",
        "min_signal_coverage": 100,
        "db_path": "paper.db",
        "risk": {
            "drawdown_hard": 0.15,
            "daily_loss": 0.03,
            "concentration": 0.15,
            "market_regime_decline": 0.08,
            "market_regime_days": 10,
            "trailing_stop": 0.20,
            "trailing_cooldown_days": 10,
            "industry_cap": 0.30,
            "soft_drawdown": 0.10,
            "soft_drawdown_recovery": 0.95,
            "default_topk": 15,
            "reduced_topk": 7,
        },
    }


# -- Synthetic prices (20 symbols, edge cases included) --------------------

@pytest.fixture()
def synthetic_prices() -> dict[str, dict]:
    """20 synthetic stocks with deterministic edge cases.

    SZ000001..SZ000020.  Includes:
    - SZ000005: limit-up (change=0.10)
    - SZ000010: limit-down (change=-0.10)
    - SZ000015: suspended (volume=0)
    - SZ000020: close > 300 (excluded by filter)
    """
    prices: dict[str, dict] = {}
    for i in range(1, 21):
        sym = f"SZ{i:06d}"
        close = 5.0 + i * 2.25
        change = 0.01
        volume = 1_000_000.0
        factor = 1.0
        threshold = 0.099

        if i == 5:
            change = 0.10   # limit-up
        elif i == 10:
            change = -0.10  # limit-down
        elif i == 15:
            volume = 0.0    # suspended
        elif i == 20:
            close = 350.0   # high-price (filter exclusion)

        prices[sym] = {
            "close": close,
            "change": change,
            "volume": volume,
            "factor": factor,
            "threshold": threshold,
        }
    return prices


# -- Synthetic signals (20 symbols, descending scores) ---------------------

@pytest.fixture()
def synthetic_signals() -> dict[str, float]:
    """Prediction scores for 20 symbols, descending 0.95 to 0.05."""
    return {
        f"SZ{i:06d}": 1.0 - i * 0.05
        for i in range(1, 21)
    }


# -- Synthetic prediction files (for integration smoke) --------------------

@pytest.fixture()
def synthetic_prediction_files(tmp_path: Path) -> Path:
    """Write one prediction parquet per date for 2025-01-06..10.

    Uses hardcoded synthetic symbols (SZ000001..SZ000120) so no qlib
    dependency is needed.  Returns the predictions directory path.
    """
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()

    dates = [
        "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-01-09", "2025-01-10",
    ]
    symbols = [f"SZ{i:06d}" for i in range(1, 121)]
    for date_str in dates:
        scores = [1.0 - (i / len(symbols)) for i in range(len(symbols))]
        df = pd.DataFrame({
            "instrument": symbols,
            "score": scores,
        })
        df.to_parquet(pred_dir / f"{date_str}.parquet", index=False)

    return pred_dir


# -- Day-1 state -----------------------------------------------------------

@pytest.fixture()
def day1_state() -> dict:
    """Initial state: 300K cash, no positions."""
    return {
        "cash": 300_000.0,
        "positions": {},
        "trade_date": "2025-01-02",
    }


# -- Populated Database ----------------------------------------------------

@pytest.fixture()
def populated_db(db_conn: sqlite3.Connection, paper_config: dict) -> sqlite3.Connection:
    from ashare_lab.paper.ledger import (
        record_nav, insert_trade, insert_order, snapshot_positions, set_cooldown
    )
    
    # First day NAV (for cumulative return base)
    record_nav(db_conn, "2024-12-30", 300_000.0, 0.0, 300_000.0, None, None, None, None)
    
    # Previous day NAV
    record_nav(db_conn, "2025-01-05", 150_000.0, 150_000.0, 300_000.0, None, None, 3000.0, 5000.0)
    
    # Current day NAV (simulating daily loss and drawdown)
    # daily_loss_pct threshold is 0.02, we'll set it to 3% drop
    total_nav = 300_000.0 * 0.97
    record_nav(db_conn, "2025-01-06", 50_000.0, total_nav - 50_000.0, total_nav, None, None, 2900.0, 4800.0)
    
    # To test regime check, insert 10 older nav rows
    base_date = 15
    for i in range(10):
        dt_str = f"2024-12-{base_date+i:02d}"
        csi1000 = 6000.0 - i * 10  # declining
        record_nav(db_conn, dt_str, 300_000.0, 0.0, 300_000.0, None, None, None, csi1000)

    # Trades for current day
    oid1 = insert_order(db_conn, "2025-01-06", "SZ000001", "buy", 1000, 10.0, "filled", 0, "2025-01-06")
    insert_trade(db_conn, oid1, "2025-01-06", "SZ000001", "buy", 10.0, 1000, 5.0, 0.0, 1.0)

    # Positions
    snapshot_positions(db_conn, "2025-01-06", {
        "SZ000001": {"qty": 1000, "avg_cost": 9.0, "market_value": 10000.0},
    })

    # Pending orders for tomorrow (trade_date='2025-01-07', created='2025-01-06')
    insert_order(db_conn, "2025-01-07", "SZ000002", "sell", 500, None, "pending", 0, "2025-01-06")
    
    # Cooldown
    set_cooldown(db_conn, "SZ000003", "2025-01-10", 15.0)

    db_conn.commit()
    return db_conn


# -- Sentiment fixtures ----------------------------------------------------

@pytest.fixture()
def sentiment_config() -> dict:
    """Flat sentiment sub-config matching baseline.yaml paper.sentiment."""
    return {
        "enabled": True,
        "stock_threshold": -2,
        "industry_threshold": -2,
        "global_threshold": -3,
        "news_count": 10,
        "rate_limit_base": 0.0,
        "rate_limit_jitter": 0.0,
        "deepseek_model": "deepseek-v4-flash",
        "deepseek_timeout": 5,
    }


@pytest.fixture()
def sentiment_db() -> sqlite3.Connection:
    """In-memory SQLite with ledger schema for sentiment cache tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    yield conn
    conn.close()
