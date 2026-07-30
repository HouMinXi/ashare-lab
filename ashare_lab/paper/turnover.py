"""Daily turnover budget for the paper pipeline.

Caps discretionary rotation buy value as a fraction of equity NAV.
Forced sells, IPO exits, liquidation sells, and carried orders are
excluded from both the numerator and the scaling.

Ramp: when the book's invested ratio (market_value / equity_nav) is
below a threshold (cold start / re-entry after liquidation), a higher
cap applies, decaying linearly over N trading days to the normal cap.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def cap_rotation_buys(
    buy_syms: list[str],
    target_value: float,
    prices: dict[str, dict],
    equity_nav: float,
    cap: float,
    ramp_state: dict,
) -> tuple[list[str], float, dict]:
    """Cap discretionary rotation buys by a turnover budget.

    Parameters
    ----------
    buy_syms : list[str]
        Symbols to buy (already filtered: no forced sells, no IPO exits).
    target_value : float
        Per-stock target value (pre-lot, pre-cap).
    prices : dict
        Price data keyed by symbol.
    equity_nav : float
        Equity NAV (ex-hedge) for the cap denominator.
    cap : float
        Normal cap fraction (e.g. 0.20 = 20% of equity NAV).
    ramp_state : dict
        Ramp state from load_turnover_ramp(). Keys: entry_date, k,
        active. Modified in-place to reflect new state.

    Returns
    -------
    (capped_syms, effective_cap, ramp_state)
        capped_syms: symbols that survive the cap (may be shorter than buy_syms)
        effective_cap: the cap that was actually applied (may be ramp cap)
        ramp_state: updated ramp state (caller should persist)
    """
    if not buy_syms or equity_nav <= 0:
        return buy_syms, target_value, cap, ramp_state

    # Compute effective cap (ramp or normal)
    effective_cap = cap
    if ramp_state.get("active", False):
        ramp_cap = ramp_state.get("ramp_cap", cap)
        ramp_days = ramp_state.get("ramp_days", 5)
        k = ramp_state.get("k", 0)
        if k < ramp_days:
            # Linear decay: ramp_cap -> cap over ramp_days
            effective_cap = ramp_cap - (ramp_cap - cap) * k / ramp_days
            logger.info(
                "[turnover] ramp active: day %d/%d, effective_cap=%.2f%%",
                k, ramp_days, effective_cap * 100,
            )
        else:
            # Ramp expired, deactivate
            ramp_state["active"] = False
            logger.info("[turnover] ramp expired after %d days, using normal cap=%.2f%%",
                        ramp_days, cap * 100)

    # Compute total planned buy value (pre-lot, pre-cap)
    total_planned = 0.0
    for sym in buy_syms:
        close = prices.get(sym, {}).get("close", 0)
        if close <= 0:
            continue
        total_planned += target_value

    cap_value = effective_cap * equity_nav

    # Cap not binding: return as-is
    if total_planned <= cap_value:
        return buy_syms, target_value, effective_cap, ramp_state

    # Cap binding: scale all buys by uniform factor
    factor = cap_value / total_planned if total_planned > 0 else 0.0
    logger.info(
        "[turnover] cap binding: planned=%.0f, cap=%.0f (=%.1f%% NAV), factor=%.4f",
        total_planned, cap_value, effective_cap * 100, factor,
    )

    # Filter: keep only symbols where scaled target_value >= 1 lot
    capped_syms = []
    for sym in buy_syms:
        close = prices.get(sym, {}).get("close", 0)
        if close <= 0:
            continue
        scaled_value = target_value * factor
        scaled_qty = int(scaled_value / close)
        if scaled_qty >= 100:  # 1 lot minimum
            capped_syms.append(sym)
        else:
            logger.warning(
                "[turnover] skip %s: scaled_value=%.0f < 1 lot (price=%.2f)",
                sym, scaled_value, close,
            )

    scaled_target_value = target_value * factor
    return capped_syms, scaled_target_value, effective_cap, ramp_state


def load_turnover_ramp(conn: sqlite3.Connection) -> dict:
    """Load ramp state from paper_state KV table.

    Returns dict with keys: entry_date (str|None), k (int), active (bool).
    """
    row = conn.execute(
        "SELECT value FROM paper_state WHERE key='turnover_ramp'"
    ).fetchone()
    if row is None:
        return {"entry_date": None, "k": 0, "active": False}
    try:
        parts = row["value"].split("|")
        if len(parts) == 2:
            return {"entry_date": parts[0], "k": int(parts[1]), "active": True}
    except (ValueError, IndexError):
        pass
    return {"entry_date": None, "k": 0, "active": False}


def save_turnover_ramp(conn: sqlite3.Connection, entry_date: str, k: int) -> None:
    """Persist ramp state to paper_state KV table."""
    value = f"{entry_date}|{k}"
    conn.execute(
        "INSERT OR REPLACE INTO paper_state (key, value) VALUES ('turnover_ramp', ?)",
        (value,),
    )


def reset_turnover_ramp(conn: sqlite3.Connection) -> None:
    """Clear ramp state (e.g. on LIQUIDATED exit)."""
    conn.execute(
        "DELETE FROM paper_state WHERE key='turnover_ramp'"
    )


def check_ramp_activation(
    equity_nav: float,
    positions: dict[str, dict],
    invested_ratio_threshold: float,
) -> bool:
    """Check if ramp should be active (book is mostly cash)."""
    if equity_nav <= 0:
        return False
    market_value = sum(pos.get("market_value", 0) for pos in positions.values())
    invested_ratio = market_value / equity_nav
    return invested_ratio < invested_ratio_threshold
