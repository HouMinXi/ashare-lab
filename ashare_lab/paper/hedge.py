"""Hedge sleeve module for dynamic equity/hedge allocation.

When portfolio drawdown exceeds a configurable activation threshold, the
sleeve shifts capital from equity into defensive legs (treasury ETF, gold
ETF, money market) along a linear ramp.  A ratcheted peak NAV and a
recovery hysteresis band reduce whipsaws.

The floor is config.min_equity_pct (default: 20% equity / 80% hedge at max
drawdown).  It is a *soft* floor -- protects against
progressive drawdowns but does not guarantee single-day gap risk, because
ChiNext/STAR stocks (41.5% of CSI1000) have 20% daily limits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ashare_lab.paper.engine import round_lots

logger = logging.getLogger(__name__)

__all__ = [
    "HedgeLeg",
    "HedgeConfig",
    "HedgeState",
    "HedgeOrder",
    "compute_hedge_state",
    "update_peak_nav",
    "generate_hedge_orders",
    "load_hedge_state",
    "save_hedge_state",
    "_load_hedge_config",
    "_fetch_hedge_prices",
    "LEG_TYPES",
]

LEG_TYPES = frozenset(
    {"treasury_etf", "gold_etf", "money_market", "reverse_repo", "cgb_bond"}
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HedgeLeg:
    """A single defensive leg inside the hedge sleeve."""

    symbol: str
    weight: float
    leg_type: str


@dataclass
class HedgeConfig:
    """Parameters for the hedge sleeve.

    The floor is defined by the last segment of *equity_ramp* plus
    *min_equity_pct*.  Fields without defaults precede defaulted fields
    so the dataclass can be constructed positionally.
    """

    equity_ramp: list[tuple[float, float]]
    legs: list[HedgeLeg]
    activate_dd: float = 0.03  # used by caller (pipeline step 9b), not by compute_hedge_state
    min_equity_pct: float = 0.20
    anti_whipsaw_days: int = 10
    recovery_pct: float = 0.97
    max_single_day_rebalance: float = 0.30
    min_delta_pct: float = 0.01  # ignore rebalance deltas < this fraction of NAV


@dataclass
class HedgeState:
    """Mutable snapshot of the sleeve for one trading day."""

    active: bool
    drawdown_pct: float
    equity_target_pct: float
    hedge_target_pct: float
    days_in_hedge: int
    days_in_recovery: int  # consecutive days below recovery_dd; resets if DD rises
    peak_nav: float
    leg_allocations: dict[str, float]


@dataclass
class HedgeOrder:
    """An order produced by the hedge sleeve.

    The *source* field is propagated into the orders table so pipeline
    settlement can distinguish hedge-driven trades from signal-driven
    trades.
    """

    symbol: str
    side: str
    target_qty: int
    source: str = "hedge"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_hedge_state(
    current_nav: float,
    peak_nav: float,
    days_in_hedge: int,
    config: HedgeConfig,
    prev_active: bool = False,
    prev_days_in_recovery: int = 0,
) -> HedgeState:
    """Return the target hedge state for the current drawdown.

    This is a pure function -- it does not mutate *days_in_hedge*.
    The caller MUST persist the returned state (including days_in_hedge)
    and increment days_in_hedge on each trading day while the sleeve is
    active.  Failing to persist causes anti-whipsaw to reset every day.

    Inactive days keep 100% equity exposure so the sleeve does not stack
    an additional buffer on top of ``risk_degree``.  Once active, the
    sleeve follows *equity_ramp* and only exits when drawdown recovers
    below ``1 - recovery_pct`` for at least ``anti_whipsaw_days``.

    The ramp floor is a *soft floor* -- it protects against progressive
    drawdowns but cannot guarantee against a single-day gap.
    """
    dd = (
        (peak_nav - current_nav) / peak_nav if peak_nav > 0 else 0.0
    )
    recovery_dd = 1.0 - config.recovery_pct

    if days_in_hedge == 0 and prev_active:
        logger.warning("hedge inconsistent state: prev_active=True but days_in_hedge=0, resetting")
        prev_active = False

    if days_in_hedge == 0 and not prev_active:
        return HedgeState(
            active=False,
            drawdown_pct=dd,
            equity_target_pct=1.0,
            hedge_target_pct=0.0,
            days_in_hedge=0,
            days_in_recovery=0,
            peak_nav=peak_nav,
            leg_allocations={},
        )

    ramp = config.equity_ramp  # pre-sorted in _load_hedge_config
    active_max_eq = ramp[0][1]
    equity_pct = _interpolate_ramp(
        dd, ramp, active_max_eq, config.min_equity_pct
    )

    # Track consecutive recovery days (resets if DD rises above threshold)
    if dd < recovery_dd:
        days_in_recovery = prev_days_in_recovery + 1
    else:
        days_in_recovery = 0

    should_exit = (
        dd < recovery_dd
        and days_in_recovery >= config.anti_whipsaw_days
        and prev_active
    )
    if should_exit:
        return HedgeState(
            active=False,
            drawdown_pct=dd,
            equity_target_pct=1.0,
            hedge_target_pct=0.0,
            days_in_hedge=0,
            days_in_recovery=0,
            peak_nav=peak_nav,
            leg_allocations={},
        )

    hedge_pct = 1.0 - equity_pct
    allocs = {leg.symbol: hedge_pct * leg.weight for leg in config.legs}
    return HedgeState(
        active=True,
        drawdown_pct=dd,
        equity_target_pct=equity_pct,
        hedge_target_pct=hedge_pct,
        days_in_hedge=days_in_hedge,
        days_in_recovery=days_in_recovery,
        peak_nav=peak_nav,
        leg_allocations=allocs,
    )


def _interpolate_ramp(
    dd: float,
    ramp: list[tuple[float, float]],
    max_eq: float,
    min_eq: float,
) -> float:
    """Linearly interpolate equity percentage across drawdown breakpoints.

    *ramp* must be sorted by drawdown ascending.  Points below the first
    breakpoint clamp to *max_eq*; points above the last clamp to *min_eq*.
    """
    if dd < ramp[0][0]:
        return max_eq
    if dd >= ramp[-1][0]:
        return min_eq
    for i in range(len(ramp) - 1):
        dd_lo, eq_lo = ramp[i]
        dd_hi, eq_hi = ramp[i + 1]
        if dd_lo <= dd <= dd_hi:
            if dd_hi > dd_lo:
                t = (dd - dd_lo) / (dd_hi - dd_lo)
            else:
                t = 0.0  # degenerate segment: return lower-bound equity
            return eq_lo + t * (eq_hi - eq_lo)
    return min_eq


def update_peak_nav(current_nav: float, peak_nav: float) -> float:
    """Ratchet peak NAV upward only."""
    return max(current_nav, peak_nav)


def generate_hedge_orders(
    current_holdings: dict,
    target_state: HedgeState,
    total_nav: float,
    prices: dict,
    config: HedgeConfig,
    buying_halted: bool = False,  # intentionally unused: hedge buys bypass halt by design
) -> list[HedgeOrder]:
    """Produce hedge orders that move holdings toward *target_state*.

    Hedge buys deliberately bypass ``buying_halted`` -- they are the
    defensive assets the sleeve buys into a crash.  Sells are always
    allowed.  A gross-turnover cap prevents a single-day rebalance from
    exceeding ``config.max_single_day_rebalance`` of NAV.
    """
    if not target_state.active:
        return []

    threshold = total_nav * config.min_delta_pct
    orders: list[HedgeOrder] = []
    deltas: list[tuple[str, int, str, float, float]] = []

    for leg in config.legs:
        sym = leg.symbol
        target_cny = total_nav * target_state.leg_allocations.get(sym, 0.0)
        current_cny = current_holdings.get(sym, {}).get("market_value") or 0.0
        delta_cny = target_cny - current_cny

        if sym not in prices or prices[sym].get("close") is None:
            logger.warning("hedge price missing: %s, skipping leg", sym)
            continue

        close = prices[sym]["close"]
        if close <= 0:
            logger.warning("hedge non-positive price: %s = %s, skipping", sym, close)
            continue

        if delta_cny < -threshold:
            side = "sell"
            qty = round_lots(abs(delta_cny) / close, side)
        elif delta_cny > threshold:
            side = "buy"
            qty = round_lots(delta_cny / close, side)
        else:
            continue

        if qty <= 0:
            continue

        deltas.append((sym, int(qty), side, current_cny, delta_cny))

    # Sell-first ordering, with market value used as a stable tie-breaker.
    sells = [d for d in deltas if d[2] == "sell"]
    buys = [d for d in deltas if d[2] == "buy"]
    sells.sort(key=lambda d: d[3], reverse=True)
    buys.sort(key=lambda d: d[3])
    sorted_deltas = sells + buys

    gross = sum(abs(d[4]) for d in sorted_deltas)
    cap_cny = config.max_single_day_rebalance * total_nav
    if gross > cap_cny > 0:
        scale = cap_cny / gross
        scaled: list[tuple[str, int, str, float, float]] = []
        for sym, qty, side, mv, delta_cny in sorted_deltas:
            new_qty = round_lots(qty * scale, side)
            if new_qty > 0:
                scaled.append((sym, new_qty, side, mv, delta_cny * scale))
        sorted_deltas = scaled

    for sym, qty, side, _mv, _delta in sorted_deltas:
        orders.append(HedgeOrder(symbol=sym, side=side, target_qty=qty))

    return orders


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_hedge_state(conn, trade_date: str) -> HedgeState | None:
    """Load the sleeve state for *trade_date*, or None if absent."""
    row = conn.execute(
        "SELECT * FROM hedge_state WHERE trade_date = ?", (trade_date,)
    ).fetchone()
    if not row:
        return None
    leg_allocs = json.loads(row["leg_json"])
    return HedgeState(
        active=bool(row["active"]),
        drawdown_pct=row["drawdown_pct"],
        equity_target_pct=row["equity_target_pct"],
        hedge_target_pct=row["hedge_target_pct"],
        days_in_hedge=row["days_in_hedge"],
        days_in_recovery=row["days_in_recovery"],
        peak_nav=row["peak_nav"],
        leg_allocations=leg_allocs,
    )


def save_hedge_state(conn, trade_date: str, state: HedgeState) -> None:
    """Persist *state* to the hedge_state table."""
    conn.execute(
        """INSERT OR REPLACE INTO hedge_state
        (trade_date, active, drawdown_pct, equity_target_pct,
         hedge_target_pct, days_in_hedge, days_in_recovery, peak_nav, leg_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade_date,
            int(state.active),
            state.drawdown_pct,
            state.equity_target_pct,
            state.hedge_target_pct,
            state.days_in_hedge,
            state.days_in_recovery,
            state.peak_nav,
            json.dumps(state.leg_allocations),
        ),
    )


# ---------------------------------------------------------------------------
# Config / price helpers
# ---------------------------------------------------------------------------


def _load_hedge_config(cfg_dict: dict) -> HedgeConfig:
    """Build a HedgeConfig from the YAML ``paper.hedge`` dict."""
    legs = [HedgeLeg(**leg) for leg in cfg_dict.get("legs", [])]
    ramp = [tuple(x) for x in cfg_dict.get("equity_ramp", [])]

    if len(ramp) < 2:
        raise ValueError("equity_ramp requires at least 2 breakpoints for linear interpolation")
    ramp.sort(key=lambda x: x[0])
    for i, (dd_val, eq_val) in enumerate(ramp):
        if not (0.0 <= eq_val <= 1.0):
            raise ValueError(f"equity_ramp[{i}] equity {eq_val} outside [0,1]")
    for i in range(len(ramp) - 1):
        if ramp[i][1] < ramp[i + 1][1]:
            raise ValueError(
                f"equity_ramp must be non-increasing: [{i}].equity={ramp[i][1]} < [{i+1}].equity={ramp[i+1][1]}"
            )
    if not legs:
        raise ValueError("at least one hedge leg required")

    total_weight = sum(leg.weight for leg in legs)
    if abs(total_weight - 1.0) >= 0.001:
        raise ValueError(f"leg weights must sum to 1.0, got {total_weight}")

    invalid_types = {leg.leg_type for leg in legs} - LEG_TYPES
    if invalid_types:
        raise ValueError(f"invalid leg_type {invalid_types}, allowed: {LEG_TYPES}")

    activate_dd = cfg_dict.get("activate_dd", 0.03)
    min_eq = cfg_dict.get("min_equity_pct", 0.20)
    if abs(ramp[-1][1] - min_eq) > 0.001:
        raise ValueError(
            f"equity_ramp[-1][1]={ramp[-1][1]} must equal min_equity_pct={min_eq}"
        )
    recovery_pct = cfg_dict.get("recovery_pct", 0.97)
    recovery_dd = 1.0 - recovery_pct
    if recovery_dd >= ramp[0][0]:
        raise ValueError(
            "recovery_dd must be strictly below activate_dd (ramp[0][0])"
        )

    if abs(ramp[0][0] - activate_dd) > 0.001:
        raise ValueError(
            "equity_ramp first breakpoint must equal activate_dd"
        )

    max_rebal = cfg_dict.get("max_single_day_rebalance", 0.30)
    if max_rebal <= 0:
        raise ValueError(f"max_single_day_rebalance must be > 0, got {max_rebal}")

    return HedgeConfig(
        equity_ramp=ramp,
        legs=legs,
        activate_dd=activate_dd,
        min_equity_pct=min_eq,
        anti_whipsaw_days=cfg_dict.get("anti_whipsaw_days", 10),
        recovery_pct=recovery_pct,
        max_single_day_rebalance=max_rebal,
        min_delta_pct=cfg_dict.get("min_delta_pct", 0.01),
    )


def _fetch_hedge_prices(ctx: "DailyRunContext", leg_symbols: list[str]) -> None:
    """Fetch closing prices for *leg_symbols* and write them into ctx.prices.

    Uses akshare ETF spot data.  Symbols already present in ``ctx.prices``
    are skipped.  Missing prices are logged and left absent.
    """
    try:
        import akshare as ak  # noqa: PLC0415
    except ImportError:
        logger.warning("akshare not installed, hedge prices unavailable")
        return

    # Skip fetch if all symbols already have prices
    if all(
        sym in ctx.prices and ctx.prices[sym].get("close") is not None
        for sym in leg_symbols
    ):
        return

    try:
        df = ak.fund_etf_spot_em()
    except Exception:
        logger.warning("hedge ETF spot fetch failed", exc_info=True)
        return
    if df is None or df.empty:
        logger.warning("hedge empty ETF spot response")
        return

    for sym in leg_symbols:
        if sym in ctx.prices and ctx.prices[sym].get("close") is not None:
            continue

        try:
            row = df[df["代码"].astype(str).str.strip() == sym]
            if row.empty:
                logger.error("hedge price missing: %s — leg dropped, allocation will be imbalanced", sym)
                continue
            close = float(row.iloc[0].get("最新价", 0))
            if close <= 0:
                logger.warning("hedge invalid price for %s: %s", sym, close)
                continue
        except (ImportError, ValueError, KeyError, OSError):
            logger.warning("hedge price fetch failed for %s", sym, exc_info=True)
            continue

        # ETF spot API returns price only; placeholders for pipeline compatibility.
        # Large finite volume: ETFs are always liquid, but inf would
        # crash int(volume * pct) in _cap_fill_by_volume if it ever
        # reaches settle.  1e12 is above any real daily volume.
        ctx.prices[sym] = {
            "close": close,
            "factor": 1.0,
            "change": 0.0,
            "volume": 1e12,
            "threshold": 0.10,
        }
