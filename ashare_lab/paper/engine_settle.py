"""Sub-functions extracted from settle_day for decomposition.

This module contains the extracted sub-functions from engine.py's settle_day.
After validation, these will be merged back into engine.py.
"""

import logging
import math
import sqlite3

from ashare_lab.paper.fees import calculate_fees
from ashare_lab.paper.ledger import (
    insert_order,
    insert_trade,
    log_settle_change,
    snapshot_positions,
    update_order,
)

logger = logging.getLogger(__name__)


def settle_sell_orders(
    conn: sqlite3.Connection,
    trade_date: str,
    sell_orders: list[dict],
    prices: dict[str, dict],
    current_positions: dict[str, dict],
    cash: float,
    carry_days_limit: int,
    slippage: float,
    participation_pct: float,
    result,  # SettleResult
) -> tuple[float, float]:
    """Process sell orders. Returns (cash, sell_proceeds).

    sell_proceeds is the total cash received from sells (T+1: not
    available for same-day buys, but available next trading day).
    """
    sell_proceeds = 0.0
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

        # Suspension check (D-32) with 25-day timeout (Phase 9 -- 09-03)
        if _is_suspended(volume):
            suspension_carry = order.get("suspension_carry_day", 0)
            if suspension_carry >= 25:
                # Suspension timeout: cancel with reason
                update_order(
                    conn, oid, status="cancelled",
                    cancel_reason="suspension_timeout",
                )
                result.cancels.append(symbol)
                logger.warning(
                    "Suspension timeout: %s cancelled after %d days",
                    symbol, suspension_carry,
                )
            else:
                update_order(conn, oid, status="carry")
                result.carries_suspended.append(
                    {"order_id": oid, "symbol": symbol, "side": "sell"}
                )
            continue

        # Limit-down block
        if _is_limit_down(change, threshold):
            if carry_day < carry_days_limit:
                update_order(conn, oid, status="carry")
                result.carries_to_bump.append(
                    {"order_id": oid, "symbol": symbol, "side": "sell"}
                )
            else:
                # NEVER cancel limit-down orders -- cancelling
                # strands the position with no exit (2024-02
                # CSI1000 crash lesson). Reset carry_day so the
                # order retries next time the stock opens.
                # reset_count increments for monitoring only.
                reset_count = order.get("reset_count", 0) + 1
                update_order(
                    conn, oid, status="carry",
                    carry_day=0, reset_count=reset_count)
                result.carries_to_bump.append(
                    {"order_id": oid, "symbol": symbol,
                     "side": "sell",
                     "reason": f"limit_down_reset_{reset_count}"}
                )
            continue

        # Fillable path
        fill_price = _apply_slippage(close, "sell", slippage)
        fill_qty = _cap_fill_by_volume(
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

        # Cash proceeds = notional - fees (T+1: not available same day)
        proceeds = notional - fees.total
        sell_proceeds += proceeds

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

    return cash, sell_proceeds


def settle_buy_orders(
    conn: sqlite3.Connection,
    trade_date: str,
    buy_orders: list[dict],
    prices: dict[str, dict],
    current_positions: dict[str, dict],
    cash: float,
    topk_symbols: set[str],
    carry_days_limit: int,
    slippage: float,
    participation_pct: float,
    result,  # SettleResult
) -> float:
    """Process buy orders. Returns updated cash."""
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

        # Suspension check (D-32) with 25-day timeout (Phase 9 -- 09-03)
        if _is_suspended(volume):
            suspension_carry = order.get("suspension_carry_day", 0)
            if suspension_carry >= 25:
                update_order(
                    conn, oid, status="cancelled",
                    cancel_reason="suspension_timeout",
                )
                result.cancels.append(symbol)
                logger.warning(
                    "Suspension timeout: %s buy cancelled after %d days",
                    symbol, suspension_carry,
                )
            else:
                update_order(conn, oid, status="carry")
                result.carries_suspended.append(
                    {"order_id": oid, "symbol": symbol, "side": "buy"}
                )
            continue

        # Limit-up block
        if _is_limit_up(change, threshold):
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
        fill_price_est = _apply_slippage(close, "buy", slippage)

        # Carry order re-sizing: if this is a carry order and the
        # estimated cost exceeds 10% of current cash, scale down
        # target_qty.  Orders are sized at creation-time NAV; after
        # a market decline the original qty may be oversized relative
        # to current portfolio value.  The 10% cap prevents a single
        # carry order from consuming more than its fair share of cash.
        if carry_day > 0 and fill_price_est > 0:
            est_cost = fill_price_est * target_qty
            max_cost = cash * 0.10
            if est_cost > max_cost and max_cost >= fill_price_est * 100:
                new_qty = min(
                    target_qty,
                    int(max_cost / fill_price_est / 100) * 100,
                )
                if new_qty >= 100 and new_qty < target_qty:
                    logger.info(
                        "Carry re-size: %s qty %d -> %d (cost %.0f -> %.0f, "
                        "cash=%.0f)",
                        symbol, target_qty, new_qty, est_cost,
                        new_qty * fill_price_est, cash,
                    )
                    target_qty = new_qty
                    # Persist re-sized qty to DB so next carry day
                    # uses the adjusted value, not the original.
                    update_order(conn, oid, target_qty=new_qty)
                    # Recalculate est_cost after re-size
                    est_cost = fill_price_est * target_qty
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
        fill_qty = _cap_fill_by_volume(
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
        fill_price = _apply_slippage(close, "buy", slippage)
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
            else:
                pos["avg_cost"] = 0.0
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

    return cash


def check_suspension_resume(
    conn: sqlite3.Connection,
    trade_date: str,
    prices: dict[str, dict],
    current_positions: dict[str, dict],
) -> None:
    """Check for resumed suspended stocks and create forced sells."""
    cancelled_suspended = conn.execute(
        "SELECT id, symbol, side, target_qty FROM orders "
        "WHERE cancel_reason = 'suspension_timeout' "
        "AND trade_date <= ? "
        "AND id = (SELECT MAX(o2.id) FROM orders o2 "
        "          WHERE o2.symbol = orders.symbol "
        "          AND o2.cancel_reason = 'suspension_timeout' "
        "          AND o2.trade_date <= ?)",
        (trade_date, trade_date),
    ).fetchall()
    for crow in cancelled_suspended:
        csym = crow["symbol"]
        # Only re-create SELL orders (not buys cancelled for other reasons)
        if crow["side"] != "sell":
            continue
        pdata = prices.get(csym, {})
        cvol = pdata.get("volume", 0.0)
        if _is_suspended(cvol):
            continue  # still suspended
        # Check position still exists
        pos = current_positions.get(csym)
        if pos is None or pos.get("qty", 0) <= 0:
            continue
        # Idempotency: check for existing pending sell
        existing = conn.execute(
            "SELECT 1 FROM orders "
            "WHERE symbol = ? AND side = 'sell' "
            "AND status IN ('pending', 'carry') "
            "AND cancel_reason IS NULL",
            (csym,),
        ).fetchone()
        if existing is not None:
            continue
        # Create forced sell order for full position
        target_qty = pos["qty"]
        new_id = insert_order(
            conn,
            trade_date=trade_date,
            symbol=csym,
            side="sell",
            target_qty=target_qty,
            price=None,
            status="pending",
            carry_day=0,
            created_run_date=trade_date,
            source="forced_liquidation",
        )
        logger.info(
            "Suspension resume: forced sell %s x%d (order %d)",
            csym, target_qty, new_id,
        )


def update_positions_post_trade(
    conn: sqlite3.Connection,
    trade_date: str,
    prices: dict[str, dict],
    current_positions: dict[str, dict],
) -> None:
    """Update holding_high and snapshot positions."""
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

    # Snapshot positions
    snapshot_positions(conn, trade_date, current_positions)


# Helper functions (extracted from engine.py)
def _is_suspended(volume: float) -> bool:
    """Check if a stock is suspended (zero volume)."""
    return volume == 0.0 or (isinstance(volume, float) and math.isnan(volume))


def _is_limit_down(change: float, threshold: float) -> bool:
    """Check if a stock is at limit-down."""
    return change <= -threshold


def _is_limit_up(change: float, threshold: float) -> bool:
    """Check if a stock is at limit-up."""
    return change >= threshold


def _apply_slippage(price: float, side: str, slippage: float) -> float:
    """Apply slippage to a price."""
    if side == "sell":
        return price * (1 - slippage)
    else:
        return price * (1 + slippage)


def _cap_fill_by_volume(
    target_qty: int,
    volume: float,
    side: str,
    participation_pct: float,
) -> int:
    """Cap fill quantity by volume participation."""
    if not math.isfinite(volume):
        return target_qty if volume > 0 else 0
    if volume <= 0:
        return 0
    max_qty = int(volume * participation_pct)
    return min(target_qty, max_qty)
