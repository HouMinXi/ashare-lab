"""Fill simulation engine for the A-share paper trading system.

Implements all A-share trading rules: T+1 settlement, limit-up/down
blocking, 100-share lot rounding, order carry-over with 3-day max
and TopK recheck, suspension handling, sell-before-buy ordering,
cash protection, lot-size skip, directional slippage, partial fill
volume cap, and index exit forced sell.

Pure helper functions have no DB access or side effects.
settle_day orchestrates one trading day atomically.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field

from ashare_lab.paper.fees import calculate_fees
from ashare_lab.paper.ledger import (
    compute_nav,
    insert_order,
    insert_trade,
    log_settle_change,
    record_nav,
    snapshot_positions,
    update_order,
)

logger = logging.getLogger(__name__)

# Re-export compute_nav so downstream (pipeline 03-06) can import from
# either ledger or engine without coupling to internal module layout.
__all__ = [
    "round_lots",
    "get_limit_threshold",
    "is_limit_up",
    "is_limit_down",
    "is_suspended",
    "apply_slippage",
    "cap_fill_by_volume",
    "compute_nav",
    "settle_day",
    "SettleResult",
]


# ---------------------------------------------------------------------------
# Pure helper functions -- no DB access, no side effects
# ---------------------------------------------------------------------------


def round_lots(qty: int | float, side: str, lot_size: int = 100) -> int:
    """Round share quantity per A-share lot rules (D-16).

    Buy: round DOWN to a multiple of *lot_size*.
    Sell: allow any integer quantity (odd lots permitted).
    """
    if side == "buy":
        return int(qty // lot_size) * lot_size
    return int(qty)


def get_limit_threshold(
    symbol: str, st_names: set[str] | None = None
) -> float:
    """Return the daily price-change limit for *symbol*.

    ST stocks: 0.049 (5% limit).
    STAR Market (688xxx, 689xxx) and ChiNext (300xxx): 0.199 (20%).
    All others: 0.099 (10%).
    """
    if st_names and symbol in st_names:
        return 0.049
    prefix = symbol[:3]
    if prefix in ("688", "689", "300"):
        return 0.199
    return 0.099


def is_limit_up(change: float, threshold: float = 0.099) -> bool:
    """Return True if *change* is at or above the limit-up threshold."""
    if change != change:  # NaN guard
        return False
    return change >= threshold


def is_limit_down(change: float, threshold: float = 0.099) -> bool:
    """Return True if *change* is at or below the limit-down threshold."""
    if change != change:  # NaN guard
        return False
    return change <= -threshold


def is_suspended(volume: float) -> bool:
    """Return True if the stock is halted (zero or NaN volume)."""
    if volume != volume:  # NaN
        return True
    return volume == 0.0


def apply_slippage(
    close: float, side: str, slippage: float = 0.001
) -> float:
    """Apply directional slippage (always unfavourable to investor)."""
    if side == "buy":
        return close * (1.0 + slippage)
    return close * (1.0 - slippage)


def cap_fill_by_volume(
    target_qty: int,
    daily_volume: float,
    side: str,
    participation_pct: float = 0.05,
) -> int:
    """Cap fill quantity by daily volume participation (D-46).

    *side* is required (no default) so sell caps skip lot rounding.
    """
    if daily_volume != daily_volume or daily_volume == 0.0:  # NaN or zero
        return 0
    max_fill = int(daily_volume * participation_pct)
    if side == "buy":
        max_fill = round_lots(max_fill, "buy")
    return min(target_qty, max_fill)


# ---------------------------------------------------------------------------
# SettleResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SettleResult:
    """Outcome of a single-day settlement."""

    pre_trade_nav: float = 0.0
    post_trade_nav: float = 0.0
    cash: float = 0.0
    fills: list[dict] = field(default_factory=list)
    carries_to_bump: list[dict] = field(default_factory=list)
    carries_suspended: list[dict] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)
    lot_skips: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# settle_day orchestrator
# ---------------------------------------------------------------------------


def settle_day(
    conn: sqlite3.Connection,
    trade_date: str,
    orders: list[dict],
    prices: dict[str, dict],
    current_positions: dict[str, dict],
    cash: float,
    topk_symbols: set[str],
    benchmarks: dict[str, float],
    config: dict,
) -> SettleResult:
    """Settle one trading day atomically.

    Processes sell orders before buy orders (D-26).  All DB mutations
    happen inside a single transaction (``with conn:``).
    """
    from ashare_lab.paper.engine_settle import (
        settle_sell_orders,
        settle_buy_orders,
        check_suspension_resume,
        update_positions_post_trade,
    )

    carry_days_limit = config.get("carry_days", 3)
    slippage = config.get("slippage", 0.001)
    participation_pct = config.get("volume_participation_pct", 0.05)

    result = SettleResult()

    # Separate sells and buys, each sorted by order_id (FIFO)
    sell_orders = sorted(
        [o for o in orders if o["side"] == "sell"],
        key=lambda o: o["id"],
    )
    buy_orders = sorted(
        [o for o in orders if o["side"] == "buy"],
        key=lambda o: o["id"],
    )

    # Step 1: pre-trade NAV
    result.pre_trade_nav = compute_nav(current_positions, prices, cash)

    with conn:
        # Step 2: process SELL orders first
        # T+1: sell proceeds are NOT available for same-day buys.
        # settle_sell_orders returns (cash, sell_proceeds) separately.
        cash, sell_proceeds = settle_sell_orders(
            conn, trade_date, sell_orders, prices,
            current_positions, cash, carry_days_limit,
            slippage, participation_pct, result,
        )

        # Step 3: process BUY orders (after sells, using pre-sell cash only)
        cash = settle_buy_orders(
            conn, trade_date, buy_orders, prices,
            current_positions, cash, topk_symbols,
            carry_days_limit, slippage, participation_pct, result,
        )

        # Step 4b: Suspension resume detection
        check_suspension_resume(conn, trade_date, prices, current_positions)

        # Step 5-5b: update positions
        update_positions_post_trade(conn, trade_date, prices, current_positions)

        # ---------------------------------------------------------------
        # Step 6: post-trade NAV
        # ---------------------------------------------------------------
        # T+1: sell_proceeds added to cash for next-day availability.
        # NAV includes sell_proceeds (they are real cash, just unsettled).
        cash_with_proceeds = cash + sell_proceeds
        result.post_trade_nav = compute_nav(
            current_positions, prices, cash_with_proceeds
        )
        result.cash = cash_with_proceeds

        # ---------------------------------------------------------------
        # Step 7: equity check (D-28)
        # ---------------------------------------------------------------
        if cash < 0:
            logger.error(
                "NAV check: negative cash %.2f on %s", cash, trade_date
            )
        for sym, pos in current_positions.items():
            if pos["qty"] < 0:
                logger.error(
                    "NAV check: negative qty %d for %s on %s",
                    pos["qty"],
                    sym,
                    trade_date,
                )

        # ---------------------------------------------------------------
        # Step 8: record NAV
        # ---------------------------------------------------------------
        market_value = result.post_trade_nav - cash
        record_nav(
            conn,
            trade_date,
            cash,
            market_value,
            result.post_trade_nav,
            result.pre_trade_nav,
            result.post_trade_nav,
            benchmarks.get("csi300"),
            benchmarks.get("csi1000"),
        )

    return result
