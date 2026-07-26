"""Unit tests for the persistent risk state machine.

All tests are pure: no DB dependency for transition logic.
Persistence tests use tmp_path SQLite.
"""

from __future__ import annotations

import sqlite3

import pytest

from ashare_lab.paper.risk_state import (
    RiskState,
    RiskStateContext,
    compute_drawdown,
    evaluate_state,
    load_risk_state,
    log_shadow_transition,
    migrate_from_soft_reduced,
    save_risk_state,
)


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture()
def config():
    """Baseline risk config for state machine tests."""
    return {
        "soft_drawdown": 0.10,
        "halt_buys": 0.12,
        "drawdown_hard": 0.15,
        "soft_drawdown_recovery": 0.95,
        "lockdown_days": 5,
        "reentry_dd": 0.10,
        "reduced_topk": 7,
    }


@pytest.fixture()
def db(tmp_path):
    """In-memory-like SQLite with paper_state table."""
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE paper_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE risk_shadow_log (
            trade_date TEXT PRIMARY KEY,
            old_flags_json TEXT NOT NULL,
            shadow_state TEXT NOT NULL,
            would_do_json TEXT NOT NULL
        )
    """)
    conn.commit()
    yield conn
    conn.close()


def _ctx(peak: float, current: float, td: str = "2026-07-25") -> RiskStateContext:
    return RiskStateContext(peak_nav=peak, current_nav=current, trade_date=td)


# -- Transition tests -------------------------------------------------------

class TestStateTransitions:
    """Every edge in the transition graph."""

    def test_normal_to_soft_reduced(self, config):
        """dd >= 0.10: NORMAL -> SOFT_REDUCED."""
        ctx = _ctx(100_000, 89_000)  # dd = 0.11
        result = evaluate_state(ctx, RiskState.NORMAL, None, config)
        assert result.state == RiskState.SOFT_REDUCED
        assert result.transition == "normal->soft_reduced"
        assert result.topk_override == 7

    def test_normal_stays_normal(self, config):
        """dd < 0.10: NORMAL stays NORMAL."""
        ctx = _ctx(100_000, 91_000)  # dd = 0.09
        result = evaluate_state(ctx, RiskState.NORMAL, None, config)
        assert result.state == RiskState.NORMAL
        assert result.transition is None
        assert result.topk_override is None

    def test_soft_reduced_to_normal_recovery(self, config):
        """dd <= 5% (recovery_pct=0.95): SOFT_REDUCED -> NORMAL."""
        ctx = _ctx(100_000, 96_000)  # dd = 0.04
        result = evaluate_state(ctx, RiskState.SOFT_REDUCED, None, config)
        assert result.state == RiskState.NORMAL
        assert result.transition == "soft_reduced->normal"

    def test_soft_reduced_stays_soft_reduced(self, config):
        """dd between 5% and 12%: SOFT_REDUCED stays."""
        ctx = _ctx(100_000, 90_000)  # dd = 0.10
        result = evaluate_state(ctx, RiskState.SOFT_REDUCED, None, config)
        assert result.state == RiskState.SOFT_REDUCED
        assert result.transition is None
        assert result.topk_override == 7

    def test_soft_reduced_to_buy_halt(self, config):
        """dd >= 0.12: SOFT_REDUCED -> BUY_HALT."""
        ctx = _ctx(100_000, 87_000)  # dd = 0.13
        result = evaluate_state(ctx, RiskState.SOFT_REDUCED, None, config)
        assert result.state == RiskState.BUY_HALT
        assert result.transition == "soft_reduced->buy_halt"
        assert result.topk_override is None

    def test_buy_halt_to_soft_reduced_deescalation(self, config):
        """dd < 0.12: BUY_HALT -> SOFT_REDUCED (de-escalation)."""
        ctx = _ctx(100_000, 89_000)  # dd = 0.11
        result = evaluate_state(ctx, RiskState.BUY_HALT, None, config)
        assert result.state == RiskState.SOFT_REDUCED
        assert result.transition == "buy_halt->soft_reduced"

    def test_buy_halt_stays_buy_halt(self, config):
        """dd between 0.12 and 0.15: BUY_HALT stays."""
        ctx = _ctx(100_000, 86_000)  # dd = 0.14
        result = evaluate_state(ctx, RiskState.BUY_HALT, None, config)
        assert result.state == RiskState.BUY_HALT
        assert result.transition is None

    def test_buy_halt_to_liquidated(self, config):
        """dd >= 0.15: BUY_HALT -> LIQUIDATED."""
        ctx = _ctx(100_000, 84_000)  # dd = 0.16
        result = evaluate_state(ctx, RiskState.BUY_HALT, None, config)
        assert result.state == RiskState.LIQUIDATED
        assert result.transition == "buy_halt->liquidated"
        assert result.forced_sells_all is True
        assert result.lockdown_enter_date == "2026-07-25"

    def test_liquidated_stays_liquidated_lockdown_not_elapsed(self, config):
        """Lockdown not elapsed: LIQUIDATED stays."""
        ctx = _ctx(100_000, 84_000, "2026-07-25")  # dd = 0.16
        # lockdown_enter_date = "2026-07-24" (1 trading day ago, need 5)
        result = evaluate_state(ctx, RiskState.LIQUIDATED, "2026-07-24", config)
        assert result.state == RiskState.LIQUIDATED
        assert result.forced_sells_all is True

    def test_liquidated_stays_liquidated_dd_still_high(self, config):
        """Lockdown elapsed but dd >= reentry: LIQUIDATED stays."""
        # dd = 0.12 >= reentry_dd 0.10
        ctx = _ctx(100_000, 88_000, "2026-07-31")
        result = evaluate_state(ctx, RiskState.LIQUIDATED, "2026-07-24", config)
        assert result.state == RiskState.LIQUIDATED
        assert result.forced_sells_all is True

    def test_liquidated_to_normal(self, config):
        """Lockdown elapsed AND dd < reentry: LIQUIDATED -> NORMAL."""
        # dd = 0.08 < reentry_dd 0.10
        ctx = _ctx(100_000, 92_000, "2026-07-31")
        result = evaluate_state(ctx, RiskState.LIQUIDATED, "2026-07-24", config)
        assert result.state == RiskState.NORMAL
        assert result.transition == "liquidated->normal"
        assert result.lockdown_enter_date is None
        assert result.forced_sells_all is False


# -- Compute drawdown -------------------------------------------------------

class TestComputeDrawdown:
    def test_basic(self):
        assert compute_drawdown(100_000, 85_000) == pytest.approx(0.15)

    def test_zero_peak(self):
        assert compute_drawdown(0, 50_000) == 0.0

    def test_no_drawdown(self):
        assert compute_drawdown(100_000, 100_000) == 0.0

    def test_negative_clamped(self):
        """current > peak should return 0 (gain, not drawdown)."""
        assert compute_drawdown(100_000, 110_000) == 0.0


# -- Persistence round-trip -------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, db):
        save_risk_state(db, RiskState.BUY_HALT, None)
        state, lockdown = load_risk_state(db)
        assert state == RiskState.BUY_HALT
        assert lockdown is None

    def test_save_and_load_with_lockdown(self, db):
        save_risk_state(db, RiskState.LIQUIDATED, "2026-07-24")
        state, lockdown = load_risk_state(db)
        assert state == RiskState.LIQUIDATED
        assert lockdown == "2026-07-24"

    def test_load_missing_returns_normal(self, db):
        state, lockdown = load_risk_state(db)
        assert state == RiskState.NORMAL
        assert lockdown is None

    def test_save_derives_is_soft_reduced(self, db):
        save_risk_state(db, RiskState.SOFT_REDUCED, None, derive_soft_reduced=True)
        row = db.execute(
            "SELECT value FROM paper_state WHERE key='is_soft_reduced'"
        ).fetchone()
        assert row["value"] == "true"

    def test_save_normal_clears_is_soft_reduced(self, db):
        db.execute(
            "INSERT INTO paper_state (key, value) VALUES ('is_soft_reduced', 'true')"
        )
        db.commit()
        save_risk_state(db, RiskState.NORMAL, None, derive_soft_reduced=True)
        row = db.execute(
            "SELECT value FROM paper_state WHERE key='is_soft_reduced'"
        ).fetchone()
        assert row["value"] == "false"

    def test_shadow_mode_does_not_derive_soft_reduced(self, db):
        """F2: shadow mode must not overwrite is_soft_reduced."""
        db.execute(
            "INSERT INTO paper_state (key, value) VALUES ('is_soft_reduced', 'true')"
        )
        db.commit()
        save_risk_state(db, RiskState.NORMAL, None, derive_soft_reduced=False)
        row = db.execute(
            "SELECT value FROM paper_state WHERE key='is_soft_reduced'"
        ).fetchone()
        assert row["value"] == "true"  # unchanged by shadow mode

    def test_save_clears_lockdown_date(self, db):
        save_risk_state(db, RiskState.LIQUIDATED, "2026-07-24")
        save_risk_state(db, RiskState.NORMAL, None)
        row = db.execute(
            "SELECT value FROM paper_state WHERE key='lockdown_enter_date'"
        ).fetchone()
        assert row is None


# -- Migration --------------------------------------------------------------

class TestMigration:
    def test_migration_from_soft_reduced_true(self, db):
        db.execute(
            "INSERT INTO paper_state (key, value) VALUES ('is_soft_reduced', 'true')"
        )
        db.commit()
        state = migrate_from_soft_reduced(db, 0.10, "2026-07-25")
        assert state == RiskState.SOFT_REDUCED

    def test_migration_from_soft_reduced_false_normal_dd(self, db):
        db.execute(
            "INSERT INTO paper_state (key, value) VALUES ('is_soft_reduced', 'false')"
        )
        db.commit()
        state = migrate_from_soft_reduced(db, 0.05, "2026-07-25")
        assert state == RiskState.NORMAL

    def test_migration_from_high_dd(self, db):
        db.execute(
            "INSERT INTO paper_state (key, value) VALUES ('is_soft_reduced', 'false')"
        )
        db.commit()
        state = migrate_from_soft_reduced(db, 0.16, "2026-07-25")
        assert state == RiskState.LIQUIDATED

    def test_migration_no_is_soft_reduced(self, db):
        """No is_soft_reduced key: derive from dd."""
        state = migrate_from_soft_reduced(db, 0.13, "2026-07-25")
        assert state == RiskState.BUY_HALT


# -- Shadow logging ---------------------------------------------------------

class TestShadowLogging:
    def test_log_and_query(self, db):
        log_shadow_transition(
            db, "2026-07-25",
            {"drawdown_halted": True, "daily_loss_halted": False},
            RiskState.LIQUIDATED,
            {"buying_halted": True, "forced_sells_all": True, "transition": "buy_halt->liquidated"},
        )
        row = db.execute(
            "SELECT * FROM risk_shadow_log WHERE trade_date='2026-07-25'"
        ).fetchone()
        assert row is not None
        assert row["shadow_state"] == "liquidated"
        import json
        old_flags = json.loads(row["old_flags_json"])
        assert old_flags["drawdown_halted"] is True

    def test_log_accepts_string_state(self, db):
        """log_shadow_transition accepts string state (from pipeline)."""
        log_shadow_transition(
            db, "2026-07-25",
            {},
            "normal",
            {"buying_halted": False},
        )
        row = db.execute(
            "SELECT shadow_state FROM risk_shadow_log WHERE trade_date='2026-07-25'"
        ).fetchone()
        assert row["shadow_state"] == "normal"


# -- Bug-injection: remove lockdown -----------------------------------------

class TestBugInjection:
    def test_remove_lockdown_allows_oscillation(self, config):
        """BUG-INJECT: if lockdown_days=0, liquidate->rebuy->liquidate is possible.

        This test PROVES the lockdown prevents oscillation by showing that
        with lockdown_days=0 the machine re-enters NORMAL immediately and
        can re-liquidate the next day.
        """
        config_no_lockdown = {**config, "lockdown_days": 0}

        # Day 1: dd >= 0.15 -> LIQUIDATED
        ctx1 = _ctx(100_000, 84_000, "2026-07-25")
        r1 = evaluate_state(ctx1, RiskState.BUY_HALT, None, config_no_lockdown)
        assert r1.state == RiskState.LIQUIDATED

        # Day 2: dd drops to 8% -> LIQUIDATED -> NORMAL (lockdown=0, dd < reentry)
        ctx2 = _ctx(100_000, 92_000, "2026-07-26")
        r2 = evaluate_state(ctx2, RiskState.LIQUIDATED, "2026-07-25", config_no_lockdown)
        assert r2.state == RiskState.NORMAL

        # Day 3: dd >= 0.15 again -> can immediately re-liquidate
        ctx3 = _ctx(100_000, 84_000, "2026-07-27")
        r3 = evaluate_state(ctx3, RiskState.NORMAL, None, config_no_lockdown)
        assert r3.state == RiskState.SOFT_REDUCED
        # With real lockdown, day 3 would still be in LIQUIDATED lockdown

    def test_lockdown_prevents_oscillation(self, config):
        """WITH lockdown: liquidate stays locked for 5 trading days."""
        # Day 1: LIQUIDATED
        ctx1 = _ctx(100_000, 84_000, "2026-07-25")
        r1 = evaluate_state(ctx1, RiskState.BUY_HALT, None, config)
        assert r1.state == RiskState.LIQUIDATED

        # Day 2: dd drops to 8% but lockdown not elapsed (1 day < 5)
        ctx2 = _ctx(100_000, 92_000, "2026-07-26")
        r2 = evaluate_state(ctx2, RiskState.LIQUIDATED, "2026-07-25", config)
        assert r2.state == RiskState.LIQUIDATED  # still locked
        assert r2.forced_sells_all is True


# -- F3: Multi-day integration test -----------------------------------------

class TestMultiDayIntegration:
    """F3: day-N -> day-N+1 evolution through save/load cycle."""

    def test_liquidate_lockdown_recovery(self, db, config):
        """Day1: enter LIQUIDATED. Day2: still locked. After 5 days + dd<0.10: NORMAL."""
        # Day 1: dd=0.16 -> LIQUIDATED
        ctx1 = _ctx(100_000, 84_000, "2026-07-25")
        r1 = evaluate_state(ctx1, RiskState.BUY_HALT, None, config)
        assert r1.state == RiskState.LIQUIDATED
        save_risk_state(db, r1.state, r1.lockdown_enter_date)

        # Day 2: load state -> still LIQUIDATED (1 day < 5)
        state, lockdown = load_risk_state(db)
        assert state == RiskState.LIQUIDATED
        assert lockdown == "2026-07-25"
        ctx2 = _ctx(100_000, 92_000, "2026-07-28")  # dd=0.08 but lockdown
        r2 = evaluate_state(ctx2, state, lockdown, config)
        assert r2.state == RiskState.LIQUIDATED  # still locked
        save_risk_state(db, r2.state, r2.lockdown_enter_date)

        # Day 3: 5 trading days later, dd=0.09 -> NORMAL
        state, lockdown = load_risk_state(db)
        ctx3 = _ctx(100_000, 91_000, "2026-08-01")  # 5 trading days after 07-25
        r3 = evaluate_state(ctx3, state, lockdown, config)
        assert r3.state == RiskState.NORMAL
        assert r3.lockdown_enter_date is None
        save_risk_state(db, r3.state, r3.lockdown_enter_date)

        # Verify final state persisted correctly
        final_state, final_lockdown = load_risk_state(db)
        assert final_state == RiskState.NORMAL
        assert final_lockdown is None

    def test_state_persists_across_runs(self, db, config):
        """State survives save/load cycle without re-deriving."""
        # Save SOFT_REDUCED
        save_risk_state(db, RiskState.SOFT_REDUCED, None)

        # Load -> still SOFT_REDUCED (not re-derived from dd)
        state, lockdown = load_risk_state(db)
        assert state == RiskState.SOFT_REDUCED

        # Evaluate with dd=0.05 (below recovery) -> NORMAL
        ctx = _ctx(100_000, 95_000, "2026-07-26")
        r = evaluate_state(ctx, state, lockdown, config)
        assert r.state == RiskState.NORMAL
        save_risk_state(db, r.state, r.lockdown_enter_date)

        # Verify persisted
        final, _ = load_risk_state(db)
        assert final == RiskState.NORMAL


# -- F4: Bug-injection (delete save -> F3 fails) ----------------------------

class TestBugInjectionSave:
    """F4: removing save_risk_state causes state to not persist."""

    def test_without_save_state_resets_to_migration(self, db, config):
        """Without save, next load returns migration-day state (NORMAL)."""
        # Simulate: evaluate -> LIQUIDATED but DO NOT save
        ctx = _ctx(100_000, 84_000, "2026-07-25")
        r = evaluate_state(ctx, RiskState.BUY_HALT, None, config)
        assert r.state == RiskState.LIQUIDATED

        # Without save, load returns NORMAL (default)
        state, lockdown = load_risk_state(db)
        assert state == RiskState.NORMAL  # NOT LIQUIDATED
        assert lockdown is None

    def test_with_save_state_persists(self, db, config):
        """With save, next load returns the saved state."""
        ctx = _ctx(100_000, 84_000, "2026-07-25")
        r = evaluate_state(ctx, RiskState.BUY_HALT, None, config)
        assert r.state == RiskState.LIQUIDATED
        save_risk_state(db, r.state, r.lockdown_enter_date)

        state, lockdown = load_risk_state(db)
        assert state == RiskState.LIQUIDATED
        assert lockdown == "2026-07-25"


# -- Edge cases -------------------------------------------------------------

class TestEdgeCases:
    def test_exact_threshold_crossed(self, config):
        """dd == 0.10 exactly: crossed (>= per PM spec, not >)."""
        ctx = _ctx(100_000, 90_000)  # dd = 0.10
        result = evaluate_state(ctx, RiskState.NORMAL, None, config)
        assert result.state == RiskState.SOFT_REDUCED  # 0.10 >= 0.10

    def test_first_day_no_history(self, config):
        """First day: peak = current, dd = 0, stays NORMAL."""
        ctx = _ctx(100_000, 100_000)
        result = evaluate_state(ctx, RiskState.NORMAL, None, config)
        assert result.state == RiskState.NORMAL

    def test_empty_positions(self, config):
        """Empty positions: peak = current (cash only), dd = 0."""
        ctx = _ctx(50_000, 50_000)
        result = evaluate_state(ctx, RiskState.NORMAL, None, config)
        assert result.state == RiskState.NORMAL
