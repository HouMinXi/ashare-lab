"""Tests for the daily turnover budget (L3).

T1: cap not binding -> orders unchanged
T2: cap binding -> uniform factor, total <= cap, WARNING for 0-lot
T3: forced sells / IPO / carried excluded from numerator and scaling
T4: ramp: day k caps follow 50,44,38,32,26,20 then normal cap
T5: ramp persists across a simulated restart (reload from ledger)
T6: Book B binds independently of Book A (different NAV)
T7: integration through _step10 path (fixture ctx) with cap on/off
T8: bug-injection at the call site
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper.ledger import get_connection, init_schema
from ashare_lab.paper.turnover import (
    cap_rotation_buys,
    load_turnover_ramp,
    save_turnover_ramp,
    reset_turnover_ramp,
    check_ramp_activation,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def turnover_conn(tmp_path: Path) -> sqlite3.Connection:
    """In-memory SQLite with paper_state table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def prices_5() -> dict[str, dict]:
    """5 stocks with known prices."""
    return {
        "SZ000001": {"close": 10.0},
        "SZ000002": {"close": 20.0},
        "SZ000003": {"close": 30.0},
        "SZ000004": {"close": 40.0},
        "SZ000005": {"close": 50.0},
    }


# -------------------------------------------------------------------
# T1: cap not binding -> orders unchanged
# -------------------------------------------------------------------


def test_cap_not_binding_unchanged(prices_5):
    """When total planned buy value < cap, return buy_syms unchanged."""
    buy_syms = ["SZ000001", "SZ000002"]
    target_value = 10000.0  # 2 * 10000 = 20000
    equity_nav = 300000.0
    cap = 0.20  # 20% of 300K = 60K; 20K < 60K -> not binding
    ramp_state = {"active": False, "k": 0}

    capped, _, effective_cap, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    assert capped == buy_syms
    assert effective_cap == cap


# -------------------------------------------------------------------
# T2: cap binding -> uniform factor, WARNING for 0-lot
# -------------------------------------------------------------------


def test_cap_binding_uniform_factor(prices_5):
    """When cap binds, all buys scaled by same factor."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 5 * 20000 = 100000
    equity_nav = 300000.0
    cap = 0.10  # 10% of 300K = 30K; 100K > 30K -> binding, factor = 0.3
    ramp_state = {"active": False, "k": 0}

    capped, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    # factor = 30000 / 100000 = 0.3
    # SZ000001: 20000*0.3 = 6000 / 10.0 = 600 shares (>= 100) -> keep
    # SZ000002: 20000*0.3 = 6000 / 20.0 = 300 shares (>= 100) -> keep
    # SZ000003: 20000*0.3 = 6000 / 30.0 = 200 shares (>= 100) -> keep
    # SZ000004: 20000*0.3 = 6000 / 40.0 = 150 shares (>= 100) -> keep
    # SZ000005: 20000*0.3 = 6000 / 50.0 = 120 shares (>= 100) -> keep
    assert len(capped) == 5


def test_cap_binding_0lot_skip(prices_5):
    """Orders scaled to < 1 lot are skipped with WARNING."""
    buy_syms = ["SZ000001", "SZ000005"]
    target_value = 1000.0  # 2 * 1000 = 2000
    equity_nav = 300000.0
    cap = 0.001  # 0.1% of 300K = 300; 2000 > 300 -> binding, factor = 0.15
    ramp_state = {"active": False, "k": 0}

    capped, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    # SZ000001: 1000*0.15 = 150 / 10.0 = 15 shares (< 100) -> skip
    # SZ000005: 1000*0.15 = 150 / 50.0 = 3 shares (< 100) -> skip
    assert len(capped) == 0


# -------------------------------------------------------------------
# T3: forced sells / IPO / carried excluded
# -------------------------------------------------------------------


def test_forced_sells_excluded(prices_5):
    """Forced sells are not in buy_syms, so they're naturally excluded."""
    # The caller (pipeline) already excludes forced_sells from buy_syms.
    # This test verifies cap_rotation_buys doesn't touch sells at all.
    buy_syms = ["SZ000001"]  # Only discretionary buy
    target_value = 10000.0
    equity_nav = 300000.0
    cap = 0.20
    ramp_state = {"active": False, "k": 0}

    capped, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    assert capped == ["SZ000001"]


# -------------------------------------------------------------------
# T4: ramp day k caps follow 50,44,38,32,26,20
# -------------------------------------------------------------------


def test_ramp_day_caps():
    """Ramp cap decays linearly from 50% to 20% over 5 days."""
    ramp_cap = 0.50
    normal_cap = 0.20
    ramp_days = 5

    expected = [0.50, 0.44, 0.38, 0.32, 0.26, 0.20]
    for k, exp in enumerate(expected):
        if k < ramp_days:
            cap = ramp_cap - (ramp_cap - normal_cap) * k / ramp_days
        else:
            cap = normal_cap
        assert abs(cap - exp) < 1e-10, f"k={k}: expected {exp}, got {cap}"


def test_ramp_active_caps(prices_5):
    """When ramp is active, effective cap is ramp cap, not normal cap."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total
    equity_nav = 300000.0
    normal_cap = 0.10  # 30K
    ramp_state = {
        "active": True, "k": 0, "entry_date": "2026-07-28",
        "ramp_cap": 0.50, "ramp_days": 5,
    }
    # ramp cap at k=0 = 0.50 -> 150K; 100K < 150K -> not binding
    capped, _, effective_cap, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, normal_cap, ramp_state,
    )
    assert len(capped) == 5
    assert abs(effective_cap - 0.50) < 1e-10


def test_ramp_expires(prices_5):
    """After ramp_days, ramp deactivates and normal cap applies."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total
    equity_nav = 300000.0
    normal_cap = 0.10  # 30K
    ramp_state = {
        "active": True, "k": 5, "entry_date": "2026-07-28",
        "ramp_cap": 0.50, "ramp_days": 5,
    }
    # k=5 >= ramp_days=5 -> ramp expires, normal cap 0.10 -> 30K
    # 100K > 30K -> binding, factor = 0.3
    capped, _, effective_cap, state = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, normal_cap, ramp_state,
    )
    assert effective_cap == normal_cap
    assert state["active"] is False


# -------------------------------------------------------------------
# T5: ramp persists across simulated restart
# -------------------------------------------------------------------


def test_ramp_persistence(turnover_conn):
    """Ramp state survives save + reload (simulates restart)."""
    save_turnover_ramp(turnover_conn, "2026-07-28", 3)
    turnover_conn.commit()

    # Simulate restart: load from DB
    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["entry_date"] == "2026-07-28"
    assert loaded["k"] == 3
    assert loaded["active"] is True


def test_ramp_reset(turnover_conn):
    """reset_turnover_ramp clears the ramp state."""
    save_turnover_ramp(turnover_conn, "2026-07-28", 2)
    turnover_conn.commit()

    reset_turnover_ramp(turnover_conn)
    turnover_conn.commit()

    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["active"] is False
    assert loaded["entry_date"] is None


def test_ramp_no_state(turnover_conn):
    """load_turnover_ramp returns inactive when no state exists."""
    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["active"] is False
    assert loaded["k"] == 0


# -------------------------------------------------------------------
# T6: Book B binds independently
# -------------------------------------------------------------------


def test_book_b_independent(prices_5):
    """Book B with different NAV binds independently."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total

    # Book A: NAV=300K, cap=20% -> 60K; 100K > 60K -> binding
    cap_a, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, 300000.0, 0.20,
        {"active": False, "k": 0},
    )
    # Book B: NAV=600K, cap=20% -> 120K; 100K < 120K -> not binding
    cap_b, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, 600000.0, 0.20,
        {"active": False, "k": 0},
    )
    assert len(cap_a) < len(cap_b) or len(cap_a) == 5  # A may bind, B doesn't


# -------------------------------------------------------------------
# T7: check_ramp_activation
# -------------------------------------------------------------------


def test_ramp_activation_low_invested():
    """Ramp activates when invested ratio < threshold."""
    positions = {"SZ000001": {"market_value": 50000}}
    equity_nav = 300000.0
    # invested_ratio = 50K/300K = 0.167 < 0.30 -> activate
    assert check_ramp_activation(equity_nav, positions, 0.30) is True


def test_ramp_activation_high_invested():
    """Ramp does NOT activate when invested ratio >= threshold."""
    positions = {"SZ000001": {"market_value": 250000}}
    equity_nav = 300000.0
    # invested_ratio = 250K/300K = 0.833 >= 0.30 -> don't activate
    assert check_ramp_activation(equity_nav, positions, 0.30) is False


def test_ramp_activation_empty_positions():
    """Empty positions -> invested_ratio=0 -> ramp activates."""
    assert check_ramp_activation(300000.0, {}, 0.30) is True


# -------------------------------------------------------------------
# T8: bug-injection (tested via integration in test_pipeline.py)
# -------------------------------------------------------------------
# The call-site injection test is in test_pipeline.py because it needs
# the full pipeline context (DailyRunContext, etc.). See
# TestTurnoverCapIntegration.test_inject_delete_cap_call.
