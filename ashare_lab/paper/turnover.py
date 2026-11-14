"""Daily turnover budget for the paper pipeline.

Caps discretionary rotation buy value as a fraction of equity NAV.
Forced sells, IPO exits, liquidation sells, and carried orders are
excluded from both the numerator and the scaling.

Ramp: when the book's invested ratio (market_value / equity_nav) is
below a threshold (cold start / re-entry after liquidation), a higher
cap applies, decaying linearly over N trading days to the normal cap.
"""

from __future__ import annotations

import math

import logging
import sqlite3

logger = logging.getLogger(__name__)

# A-share cash equities trade in 100-share lots; a buy worth less
# than one lot is not an order, and funding such a name would spend a
# budget slot to send nothing.
LOT_SIZE = 100


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
        Symbols to buy (already filtered: no forced sells, no IPO exits), in
        rank order.  The allocation is order-sensitive: the budget funds the
        front of this list first.
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
    (funded_syms, value_per_name, effective_cap, ramp_state)
        funded_syms: symbols to order today, in rank order.  Empty when the
            budget cannot cover a single lot.
        value_per_name: what each funded symbol is sized at.  Normally the
            target value; in the fallback branch the whole budget for one name.
        effective_cap: the cap that was actually applied (may be ramp cap)
        ramp_state: updated ramp state (caller should persist)
    """
    if not buy_syms or equity_nav <= 0:
        return buy_syms, target_value, cap, ramp_state

    if target_value <= 0:
        # Nothing to size: a zero target means no order, not an unscaled list.
        return [], target_value, cap, ramp_state

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

    # Cap binding: fund a ranked prefix at full size.  Scaling every buy by a
    # uniform factor and then discarding whatever falls below one lot deletes
    # the plan exactly when the book is emptiest -- twelve planned buys become a
    # quarter of a position each and none survives, which is how a book three
    # names light stops buying altogether.  The budget limits how many names may
    # move today, not how large each one is.
    planned_value = target_value

    def _orderable(sym: str, budget: float) -> bool:
        close = prices.get(sym, {}).get("close")
        if not isinstance(close, (int, float)) or not math.isfinite(close) or close <= 0:
            return False
        # One lot has to fit inside what this name would be given, or the order
        # rounds to zero shares and still consumes a budget slot.
        return close * LOT_SIZE <= budget

    # Floor with tolerance: 0.20 * 15 / 0.95 is 3.157 in decimal but can land a
    # hair under an integer in binary, and floor() would then fund one name
    # fewer than the budget allows.
    names_today = math.floor(cap_value / target_value + 1e-9)
    if names_today >= 1:
        candidates = [s for s in buy_syms if _orderable(s, target_value)]
        chosen = candidates[:names_today]
    else:
        # Not even one full position fits: put the whole budget into the first
        # name that can still buy a lot, instead of sending nothing.
        candidates = [s for s in buy_syms if _orderable(s, cap_value)]
        chosen = candidates[:1]
        if chosen:
            target_value = cap_value

    logger.info(
        "[turnover] cap binding: planned=%d x %.0f = %.0f, cap=%.0f (%.1f%% NAV); "
        "funding %d at %.0f, deferring %d",
        len(buy_syms), planned_value, total_planned, cap_value,
        effective_cap * 100, len(chosen), target_value,
        len(buy_syms) - len(chosen),
    )
    return chosen, target_value, effective_cap, ramp_state


def charge_ramp_day(
    conn: sqlite3.Connection,
    ramp_state: dict | None,
    orders_inserted: int,
    trade_date: str,
) -> bool:
    """Advance the ramp by one day when today actually deployed capital.

    A cap that funded nothing -- or prices that rounded every order to zero
    shares -- spends no budget, so charging it would shorten the window that
    exists to fund deployment.  Returns True when the state was written; the
    caller commits.
    """
    if not ramp_state or not ramp_state.get("active", False) or orders_inserted <= 0:
        return False
    save_turnover_ramp(
        conn,
        ramp_state.get("entry_date") or trade_date,
        ramp_state.get("k", 0) + 1,
    )
    return True


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
