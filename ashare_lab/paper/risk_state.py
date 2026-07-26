"""Persistent risk state machine for drawdown-driven control.

Replaces the per-day stateless hard breaker with an explicit state
machine that prevents whipsaw oscillation (liquidate->rebuy->liquidate).

States (drawdown-driven, persisted in paper_state):
  NORMAL       dd < soft (0.10)
  SOFT_REDUCED dd >= soft (0.10), dd < halt_buys (0.12)
  BUY_HALT     dd >= halt_buys (0.12), dd < liquidate (0.15)
  LIQUIDATED   dd >= liquidate (0.15) -- force-sell ALL + lockdown

Transitions:
  NORMAL       -> SOFT_REDUCED   dd >= soft
  SOFT_REDUCED -> NORMAL         dd < recovery_pct * peak (0.95, hysteresis)
  SOFT_REDUCED -> BUY_HALT       dd >= halt_buys
  BUY_HALT     -> SOFT_REDUCED   dd < halt_buys (de-escalation)
  BUY_HALT     -> LIQUIDATED     dd >= liquidate
  LIQUIDATED   -> NORMAL         lockdown elapsed AND dd < reentry_dd

Daily overlays (NOT persistent, NOT states, NOT lockdown):
  daily_loss, market_regime, staleness are per-day inputs that block
  buying for that day only, on top of whatever persistent state is
  active.  They never change the persistent state and never count
  toward lockdown.

Boundary with hedge sleeve:
  hedge = pre-drawdown layer (activate_dd=0.03, anti_whipsaw_days=10).
  This module = drawdown layer (10/12/15%).
  final buying_halted = state_machine_halted OR hedge_halted.
  hedge is currently disabled (locked until graduation gate).

Shadow mode (default):
  The state machine computes states and logs every transition to
  risk_shadow_log, but the existing risk.py logic keeps full authority.
  Zero behavior change in shadow.  Enforce mode requires a separate
  order after the G2 gate passes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class RiskState(str, Enum):
    """Persistent risk states driven by drawdown."""

    NORMAL = "normal"
    SOFT_REDUCED = "soft_reduced"
    BUY_HALT = "buy_halt"
    LIQUIDATED = "liquidated"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass
class RiskStateContext:
    """Input context for state machine evaluation."""

    peak_nav: float
    current_nav: float
    trade_date: str
    is_shadow: bool = True


@dataclass
class RiskStateResult:
    """Output from state machine evaluation."""

    state: RiskState
    lockdown_enter_date: str | None
    forced_sells_all: bool  # True only in LIQUIDATED
    topk_override: int | None  # reduced_topk in SOFT_REDUCED, else None
    transition: str | None  # e.g. "NORMAL->SOFT_REDUCED", None if unchanged


def compute_drawdown(peak_nav: float, current_nav: float) -> float:
    """Compute drawdown as fraction of peak. Returns 0 if peak <= 0."""
    if peak_nav <= 0:
        return 0.0
    return max(0.0, (peak_nav - current_nav) / peak_nav)


def evaluate_state(
    ctx: RiskStateContext,
    current_state: RiskState,
    lockdown_enter_date: str | None,
    config: dict,
) -> RiskStateResult:
    """Evaluate state transition based on current drawdown.

    Args:
        ctx: current nav context
        current_state: persisted state from previous run
        lockdown_enter_date: date when LIQUIDATED was entered (for lockdown)
        config: paper.risk config dict

    Returns:
        RiskStateResult with new state, lockdown info, and actions.
    """
    dd = compute_drawdown(ctx.peak_nav, ctx.current_nav)
    soft = config.get("soft_drawdown", 0.10)
    halt_buys = config.get("halt_buys", 0.12)
    liquidate = config.get("drawdown_hard", 0.15)
    recovery_pct = config.get("soft_drawdown_recovery", 0.95)
    lockdown_days = config.get("lockdown_days", 5)
    reentry_dd = config.get("reentry_dd", 0.10)

    old_state = current_state
    new_state = current_state
    new_lockdown_date = lockdown_enter_date
    forced_sells_all = False
    topk_override = None

    if current_state == RiskState.LIQUIDATED:
        # Check lockdown exit: days elapsed AND dd < reentry
        from ashare_lab.data.calendar import trading_days_between  # noqa: PLC0415
        import datetime as dt  # noqa: PLC0415

        if lockdown_enter_date:
            elapsed = len(
                trading_days_between(
                    dt.date.fromisoformat(lockdown_enter_date),
                    dt.date.fromisoformat(ctx.trade_date),
                )
            )
        else:
            elapsed = lockdown_days  # no date = treat as elapsed

        if elapsed >= lockdown_days and dd < reentry_dd:
            new_state = RiskState.NORMAL
            new_lockdown_date = None
        else:
            new_state = RiskState.LIQUIDATED
            forced_sells_all = True

    elif current_state == RiskState.BUY_HALT:
        if dd >= liquidate:
            new_state = RiskState.LIQUIDATED
            new_lockdown_date = ctx.trade_date
            forced_sells_all = True
        elif dd < halt_buys:
            # De-escalation to SOFT_REDUCED
            new_state = RiskState.SOFT_REDUCED
        else:
            new_state = RiskState.BUY_HALT

    elif current_state == RiskState.SOFT_REDUCED:
        if dd >= halt_buys:
            new_state = RiskState.BUY_HALT
        elif ctx.current_nav >= ctx.peak_nav * recovery_pct:
            # Recovery: dd <= 5% (existing hysteresis)
            new_state = RiskState.NORMAL
        else:
            new_state = RiskState.SOFT_REDUCED
            # topk_override applies in SOFT_REDUCED
            topk_override = config.get("reduced_topk", 7)

    else:  # NORMAL
        if dd >= soft:
            new_state = RiskState.SOFT_REDUCED
            topk_override = config.get("reduced_topk", 7)
        else:
            new_state = RiskState.NORMAL

    transition = None
    if new_state != old_state:
        transition = f"{old_state.value}->{new_state.value}"

    return RiskStateResult(
        state=new_state,
        lockdown_enter_date=new_lockdown_date,
        forced_sells_all=forced_sells_all,
        topk_override=topk_override,
        transition=transition,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_RISK_STATE_KEY = "risk_state"
_LOCKDOWN_DATE_KEY = "lockdown_enter_date"


def load_risk_state(conn) -> tuple[RiskState, str | None]:
    """Load persisted risk state and lockdown date from paper_state.

    Returns (state, lockdown_enter_date).  If no risk_state key exists,
    returns (NORMAL, None) -- migration happens separately.
    """
    row = conn.execute(
        "SELECT value FROM paper_state WHERE key = ?", (_RISK_STATE_KEY,)
    ).fetchone()
    if row is None:
        return RiskState.NORMAL, None

    try:
        state = RiskState(row["value"])
    except ValueError:
        logger.warning("Invalid risk_state value %r, defaulting to NORMAL", row["value"])
        return RiskState.NORMAL, None

    lockdown_row = conn.execute(
        "SELECT value FROM paper_state WHERE key = ?", (_LOCKDOWN_DATE_KEY,)
    ).fetchone()
    lockdown_date = lockdown_row["value"] if lockdown_row else None

    return state, lockdown_date


def save_risk_state(
    conn, state: RiskState, lockdown_enter_date: str | None,
) -> None:
    """Persist risk state and lockdown date to paper_state.

    Also derives is_soft_reduced for backward compatibility until
    report.py stops reading it.
    """
    conn.execute(
        "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
        (_RISK_STATE_KEY, state.value),
    )
    if lockdown_enter_date:
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
            (_LOCKDOWN_DATE_KEY, lockdown_enter_date),
        )
    else:
        conn.execute(
            "DELETE FROM paper_state WHERE key = ?", (_LOCKDOWN_DATE_KEY,)
        )
    # Backward compat: derive is_soft_reduced from state
    is_soft = "true" if state == RiskState.SOFT_REDUCED else "false"
    conn.execute(
        "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
        ("is_soft_reduced", is_soft),
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_from_soft_reduced(
    conn, current_dd: float, trade_date: str,
) -> RiskState:
    """One-time migration: derive risk_state from existing is_soft_reduced.

    Called when risk_state key is absent in paper_state.  After migration,
    risk_state is the source of truth; is_soft_reduced becomes derived.

    Args:
        conn: SQLite connection
        current_dd: current drawdown fraction
        trade_date: current trade date

    Returns:
        The initial RiskState.
    """
    row = conn.execute(
        "SELECT value FROM paper_state WHERE key = 'is_soft_reduced'"
    ).fetchone()

    if row and row["value"] == "true":
        # Existing soft-reduced state maps to SOFT_REDUCED
        initial = RiskState.SOFT_REDUCED
    else:
        # Derive from current drawdown
        # (thresholds will be read from config by caller)
        if current_dd >= 0.15:
            initial = RiskState.LIQUIDATED
        elif current_dd >= 0.12:
            initial = RiskState.BUY_HALT
        elif current_dd >= 0.10:
            initial = RiskState.SOFT_REDUCED
        else:
            initial = RiskState.NORMAL

    lockdown_date = trade_date if initial == RiskState.LIQUIDATED else None
    save_risk_state(conn, initial, lockdown_date)
    logger.info(
        "Migrated risk_state: is_soft_reduced=%s, dd=%.4f -> %s",
        row["value"] if row else "absent", current_dd, initial.value,
    )
    return initial


# ---------------------------------------------------------------------------
# Shadow logging
# ---------------------------------------------------------------------------

_SHADOW_LOG_SQL = """\
CREATE TABLE IF NOT EXISTS risk_shadow_log (
    trade_date      TEXT NOT NULL,
    old_flags_json  TEXT NOT NULL,
    shadow_state    TEXT NOT NULL,
    would_do_json   TEXT NOT NULL,
    PRIMARY KEY (trade_date)
)
"""


def ensure_shadow_log_table(conn) -> None:
    """Create risk_shadow_log table if it does not exist."""
    conn.execute(_SHADOW_LOG_SQL)


def log_shadow_transition(
    conn,
    trade_date: str,
    old_flags: dict,
    shadow_state: RiskState | str,
    would_do: dict,
) -> None:
    """Write a shadow mode log entry for this trade date.

    Args:
        conn: SQLite connection
        trade_date: current trade date
        old_flags: dict of old-logic flags (drawdown_halted, etc.)
        shadow_state: the state machine's computed state (enum or str)
        would_do: dict of what the state machine would have done
            (buying_halted, forced_sells_all, topk_override, transition)
    """
    state_str = shadow_state.value if isinstance(shadow_state, RiskState) else shadow_state
    conn.execute(
        "INSERT OR REPLACE INTO risk_shadow_log "
        "(trade_date, old_flags_json, shadow_state, would_do_json) "
        "VALUES (?, ?, ?, ?)",
        (
            trade_date,
            json.dumps(old_flags, sort_keys=True),
            state_str,
            json.dumps(would_do, sort_keys=True),
        ),
    )
