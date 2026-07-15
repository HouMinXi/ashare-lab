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


def _check_lot_budget(
    target_value: float, close: float, lot_size: int = 100
) -> tuple[int, bool]:
    """Check whether the per-stock budget can buy at least one lot.

    Returns (rounded_qty, is_lot_skip).
    """
    if close <= 0:
        return (0, True)
    raw_qty = target_value / close
    rounded = round_lots(raw_qty, "buy", lot_size)
    if rounded == 0:
        return (0, True)
    return (rounded, False)


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
    carry_days_limit = config.get("carry_days", 3)
    max_resets = config.get("max_limit_down_resets", 5)
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
        # ---------------------------------------------------------------
        # Step 2: process SELL orders first
        # ---------------------------------------------------------------
        for order in sell_orders:
            oid = order["id"]
            symbol = order["symbol"]
            target_qty = order["target_qty"]
            carry_day = order["carry_day"]

            # Before-image journal (F-F)
            log_settle_change(conn, trade_date, oid)

            pdata = prices.get(symbol, {})
            volume = pdata.get("volume", 0.0)
            close = pdata.get("close")
            change = pdata.get("change", 0.0)
            threshold = pdata.get("threshold", 0.099)

            # Missing price: fallback to avg_cost from position
            if close is None:
                pos = current_positions.get(symbol)
                if pos is None:
                    # Delisted, no position -- cancel
                    update_order(conn, oid, status="cancelled")
                    result.cancels.append(symbol)
                    continue
                close = pos["avg_cost"]

            # Suspension check (D-32)
            if is_suspended(volume):
                update_order(conn, oid, status="carry")
                result.carries_suspended.append(
                    {"order_id": oid, "symbol": symbol, "side": "sell"}
                )
                continue

            # Limit-down block
            if is_limit_down(change, threshold):
                if carry_day < carry_days_limit:
                    update_order(conn, oid, status="carry")
                    result.carries_to_bump.append(
                        {"order_id": oid, "symbol": symbol, "side": "sell"}
                    )
                else:
                    # Reset carry_day so the order retries next time
                    # the stock opens. Cancelling strands the position
                    # with no exit (2024-02 CSI1000 crash lesson).
                    # Track reset count via reset_count column.
                    # After max_resets, cancel to prevent infinite
                    # carry on permanently halted stocks.
                    reset_count = order.get("reset_count", 0) + 1
                    if reset_count >= max_resets:
                        update_order(conn, oid, status="cancelled")
                        result.cancels.append(symbol)
                    else:
                        update_order(
                            conn, oid, status="carry",
                            carry_day=-1, reset_count=reset_count)
                        result.carries_to_bump.append(
                            {"order_id": oid, "symbol": symbol,
                             "side": "sell",
                             "reason": f"limit_down_reset_{reset_count}"}
                        )
                continue

            # Fillable path
            fill_price = apply_slippage(close, "sell", slippage)
            fill_qty = cap_fill_by_volume(
                target_qty, volume, "sell", participation_pct
            )

            if fill_qty == 0:
                # Low liquidity, not a true halt
                update_order(conn, oid, status="carry")
                result.carries_to_bump.append(
                    {"order_id": oid, "symbol": symbol, "side": "sell"}
                )
                continue

            if fill_qty < target_qty:
                # Partial sell
                update_order(
                    conn, oid, status="partial", filled_qty=fill_qty
                )
                remainder_qty = target_qty - fill_qty
                new_id = insert_order(
                    conn,
                    trade_date=trade_date,
                    symbol=symbol,
                    side="sell",
                    target_qty=remainder_qty,
                    price=None,
                    status="carry",
                    carry_day=0,  # fresh window: partial fill proves liquidity
                    created_run_date=trade_date,
                )
                result.carries_to_bump.append(
                    {"order_id": new_id, "symbol": symbol, "side": "sell"}
                )
            else:
                fill_qty = target_qty
                update_order(
                    conn, oid, status="filled", filled_qty=fill_qty
                )

            # Fee calculation and trade insertion
            notional = fill_price * fill_qty
            fees = calculate_fees(notional, "sell")
            insert_trade(
                conn,
                order_id=oid,
                trade_date=trade_date,
                symbol=symbol,
                side="sell",
                fill_price=fill_price,
                fill_qty=fill_qty,
                commission=fees.commission,
                stamp=fees.stamp,
                transfer_fee=fees.transfer,
            )

            # Update positions: reduce qty, keep avg_cost unchanged (D-04)
            pos = current_positions.get(symbol, {})
            old_qty = pos.get("qty", 0)
            new_qty = old_qty - fill_qty
            if new_qty <= 0:
                current_positions.pop(symbol, None)
            else:
                pos["qty"] = new_qty
                pos["market_value"] = new_qty * close

            # Cash proceeds = notional - fees
            cash += notional - fees.total

            result.fills.append(
                {
                    "order_id": oid,
                    "symbol": symbol,
                    "side": "sell",
                    "fill_qty": fill_qty,
                    "fill_price": fill_price,
                    "fees": fees.total,
                }
            )

        # ---------------------------------------------------------------
        # Step 3: process BUY orders (after sells)
        # ---------------------------------------------------------------
        for order in buy_orders:
            oid = order["id"]
            symbol = order["symbol"]
            target_qty = order["target_qty"]
            carry_day = order["carry_day"]

            # Before-image journal (F-F)
            log_settle_change(conn, trade_date, oid)

            # TopK recheck: BUY carry orders only
            if carry_day > 0 and symbol not in topk_symbols:
                update_order(conn, oid, status="cancelled")
                result.cancels.append(symbol)
                continue

            pdata = prices.get(symbol, {})
            volume = pdata.get("volume", 0.0)
            close = pdata.get("close")
            change = pdata.get("change", 0.0)
            threshold = pdata.get("threshold", 0.099)
            factor = pdata.get("factor", 1.0)

            # Missing price: fallback to avg_cost from position
            if close is None:
                pos = current_positions.get(symbol)
                if pos is None:
                    # Delisted buy candidate with no position -- cancel
                    update_order(conn, oid, status="cancelled")
                    result.cancels.append(symbol)
                    continue
                close = pos["avg_cost"]

            # Suspension check (D-32)
            if is_suspended(volume):
                update_order(conn, oid, status="carry")
                result.carries_suspended.append(
                    {"order_id": oid, "symbol": symbol, "side": "buy"}
                )
                continue

            # Limit-up block
            if is_limit_up(change, threshold):
                if carry_day < carry_days_limit:
                    update_order(conn, oid, status="carry")
                    result.carries_to_bump.append(
                        {"order_id": oid, "symbol": symbol, "side": "buy"}
                    )
                else:
                    update_order(conn, oid, status="cancelled")
                    result.cancels.append(symbol)
                continue

            # Lot-skip guard (defensive -- should not reach here)
            if target_qty == 0:
                update_order(conn, oid, status="lot_skip")
                result.lot_skips.append(symbol)
                logger.warning(
                    "Lot-skip: %s target_qty=0 on %s", symbol, trade_date
                )
                continue

            # Cash check (D-25)
            fill_price_est = apply_slippage(close, "buy", slippage)
            est_cost = fill_price_est * target_qty
            fees_est = calculate_fees(est_cost, "buy")
            total_cost_est = est_cost + fees_est.total
            if total_cost_est > cash:
                if carry_day < carry_days_limit:
                    update_order(conn, oid, status="carry")
                    result.carries_to_bump.append(
                        {"order_id": oid, "symbol": symbol, "side": "buy"}
                    )
                else:
                    update_order(conn, oid, status="cancelled")
                    result.cancels.append(symbol)
                continue

            # Volume cap
            fill_qty = cap_fill_by_volume(
                target_qty, volume, "buy", participation_pct
            )

            # Cap=0 guard: low liquidity, not a true halt
            if fill_qty == 0:
                update_order(conn, oid, status="carry")
                result.carries_to_bump.append(
                    {"order_id": oid, "symbol": symbol, "side": "buy"}
                )
                continue

            # Fillable path
            fill_price = apply_slippage(close, "buy", slippage)
            actual_notional = fill_price * fill_qty
            fees = calculate_fees(actual_notional, "buy")

            # Final cash check for actual fill amount
            if actual_notional + fees.total > cash:
                if carry_day < carry_days_limit:
                    update_order(conn, oid, status="carry")
                    result.carries_to_bump.append(
                        {"order_id": oid, "symbol": symbol, "side": "buy"}
                    )
                else:
                    update_order(conn, oid, status="cancelled")
                    result.cancels.append(symbol)
                continue

            if fill_qty < target_qty:
                # Partial buy
                update_order(
                    conn, oid, status="partial", filled_qty=fill_qty
                )
                remainder_qty = target_qty - fill_qty
                new_id = insert_order(
                    conn,
                    trade_date=trade_date,
                    symbol=symbol,
                    side="buy",
                    target_qty=remainder_qty,
                    price=None,
                    status="carry",
                    carry_day=0,  # fresh window: partial fill proves liquidity
                    created_run_date=trade_date,
                )
                result.carries_to_bump.append(
                    {"order_id": new_id, "symbol": symbol, "side": "buy"}
                )
            else:
                fill_qty = target_qty
                update_order(
                    conn, oid, status="filled", filled_qty=fill_qty
                )

            # Trade insertion
            insert_trade(
                conn,
                order_id=oid,
                trade_date=trade_date,
                symbol=symbol,
                side="buy",
                fill_price=fill_price,
                fill_qty=fill_qty,
                commission=fees.commission,
                stamp=fees.stamp,
                transfer_fee=fees.transfer,
            )

            # Deduct cash
            cash -= actual_notional + fees.total

            # Update positions
            pos = current_positions.get(symbol)
            if pos is not None:
                old_qty = pos["qty"]
                old_avg = pos["avg_cost"]
                new_qty = old_qty + fill_qty
                if new_qty > 0:
                    pos["avg_cost"] = (
                        (old_avg * old_qty + fill_price * fill_qty) / new_qty
                    )
                pos["qty"] = new_qty
                pos["market_value"] = new_qty * close
                pos["holding_high"] = max(
                    pos.get("holding_high", 0.0), fill_price
                )
            else:
                current_positions[symbol] = {
                    "qty": fill_qty,
                    "avg_cost": fill_price,
                    "market_value": fill_qty * close,
                    "buy_date": trade_date,
                    "holding_high": fill_price,
                    "factor": factor,
                }

            result.fills.append(
                {
                    "order_id": oid,
                    "symbol": symbol,
                    "side": "buy",
                    "fill_qty": fill_qty,
                    "fill_price": fill_price,
                    "fees": fees.total,
                }
            )

        # ---------------------------------------------------------------
        # Step 5: update holding_high for ALL positions (period high)
        # ---------------------------------------------------------------
        for sym, pos in current_positions.items():
            pdata = prices.get(sym, {})
            close_price = pdata.get("close")
            if close_price is not None and not (
                isinstance(close_price, float) and math.isnan(close_price)
            ):
                if close_price > pos.get("holding_high", 0.0):
                    pos["holding_high"] = close_price
            # Update market_value with current close
            if close_price is not None and not (
                isinstance(close_price, float) and math.isnan(close_price)
            ):
                pos["market_value"] = pos["qty"] * close_price

        # Step 5b: snapshot positions
        snapshot_positions(conn, trade_date, current_positions)

        # ---------------------------------------------------------------
        # Step 6: post-trade NAV
        # ---------------------------------------------------------------
        result.post_trade_nav = compute_nav(
            current_positions, prices, cash
        )
        result.cash = cash

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
