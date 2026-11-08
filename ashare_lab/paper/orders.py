"""Buy-side order placement, shared by both paper books.

Book A and Book B each carried their own copy of this sequence: drop the symbols
whose buy is still in flight, apply the turnover budget and the ramp, insert
what survived, then charge the ramp for the day. Two copies meant every fix had
to be made twice -- and twice in one session it was not, once when in-flight
dedupe landed in A but not B, once when the ramp day counter did.

Sells are not here: Book B's sell path deliberately differs (it derives the
desired quantity from its own book snapshot instead of subtracting what is
already pending).
"""

from __future__ import annotations

import logging
import math

# The turnover helpers are called through the module rather than imported by
# name: tests patch ashare_lab.paper.turnover.cap_rotation_buys to simulate a
# bypassed budget, and a direct import would keep pointing at the original.
from ashare_lab.paper import turnover
from ashare_lab.paper.engine import round_lots
from ashare_lab.paper.ledger import insert_order, symbols_with_buy_in_flight

logger = logging.getLogger(__name__)


def _valid_close(price) -> bool:
    """A close we can size an order from.

    NaN counts as invalid: `not nan` and `nan <= 0` are both False, so the older
    check let it through to round_lots, which raises on NaN and aborted that
    book's placement for the day.
    """
    return isinstance(price, (int, float)) and math.isfinite(price) and price > 0


def place_buys(
    conn,
    *,
    buy_syms: list[str],
    target_value: float,
    prices: dict[str, dict],
    equity_nav: float,
    positions: dict[str, dict],
    paper_cfg: dict,
    next_td_str: str,
    trade_date: str,
    label: str = "",
) -> tuple[list[str], float, int]:
    """Filter, budget and insert today's discretionary buy orders.

    Returns (funded symbols, value per name, orders inserted). The ramp is
    charged for the day only when something was actually inserted, and the
    caller's commit happens here for the same reason.
    """
    in_flight = symbols_with_buy_in_flight(conn, buy_syms)
    if in_flight:
        logger.info(
            "[orders] %s%d buy(s) already in flight, not ordering again: %s",
            f"{label}: " if label else "", len(in_flight), ", ".join(sorted(in_flight)),
        )
        buy_syms = [s for s in buy_syms if s not in in_flight]

    ramp_state = None
    turnover_cap = paper_cfg.get("turnover_cap")
    if turnover_cap and buy_syms and equity_nav > 0:
        ramp_state = turnover.load_turnover_ramp(conn)
        invested_threshold = paper_cfg.get("turnover_ramp_invested_ratio", 0.30)
        if turnover.check_ramp_activation(equity_nav, positions, invested_threshold):
            if not ramp_state.get("active", False):
                ramp_state["active"] = True
                ramp_state["k"] = 0
                ramp_state["entry_date"] = trade_date
                logger.info(
                    "[turnover] ramp activated on %s (invested_ratio < %.0f%%)",
                    trade_date, invested_threshold * 100,
                )
        else:
            if ramp_state.get("active", False):
                turnover.reset_turnover_ramp(conn)
                ramp_state = {"entry_date": None, "k": 0, "active": False}
                logger.info(
                    "[turnover] ramp deactivated (invested_ratio >= %.0f%%)",
                    invested_threshold * 100,
                )
        ramp_state["ramp_cap"] = paper_cfg.get("turnover_ramp_cap", 0.50)
        ramp_state["ramp_days"] = paper_cfg.get("turnover_ramp_days", 5)
        buy_syms, target_value, _effective_cap, ramp_state = turnover.cap_rotation_buys(
            buy_syms, target_value, prices, equity_nav, turnover_cap, ramp_state,
        )

    inserted = 0
    for symbol in buy_syms:
        close_price = prices.get(symbol, {}).get("close")
        if not _valid_close(close_price):
            logger.warning("Skip buy %s: invalid close price %s", symbol, close_price)
            continue
        target_qty = round_lots(target_value / close_price, "buy")
        if target_qty <= 0:
            continue
        insert_order(conn, next_td_str, symbol, "buy", target_qty, None, "pending", 0, trade_date)
        inserted += 1

    if turnover.charge_ramp_day(conn, ramp_state, inserted, trade_date):
        conn.commit()

    return buy_syms, target_value, inserted
