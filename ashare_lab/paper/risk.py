"""7-dimension risk control framework for the paper trading engine.

Dimensions:
1. Max drawdown circuit breaker (D-35)
2. Daily loss limit (D-36)
3. Position concentration cap (D-37)
4. Market regime filter (D-38)
5. Trailing stop with cooldown (D-39)
6. CSRC industry concentration (D-40)
7. Soft drawdown topk reduction (D-45)

Each dimension is a pure check function.  run_all_risk_checks aggregates
them into a single frozen RiskCheckResult.  All thresholds are read from
the config dict -- no hardcoded defaults in function signatures.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

from ashare_lab.data.calendar import next_trading_day, trading_days_between

logger = logging.getLogger(__name__)

__all__ = [
    "RiskCheckResult",
    "check_drawdown_breaker",
    "check_daily_loss",
    "check_concentration",
    "check_market_regime",
    "check_trailing_stop",
    "manage_trailing_cooldown",
    "check_industry_concentration",
    "check_soft_drawdown",
    "check_prediction_staleness",
    "check_suspension_risk",
    "run_all_risk_checks",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskCheckResult:
    """Aggregated outcome of all risk checks for one trading day.

    Attributes:
        buying_halted: True if any hard-halt dimension fired.
        forced_sells: symbol -> qty to sell (concentration excess or
            trailing stop full-position exit).
        blocked_industries: CSRC industries over the cap.
        blocked_rebuys: symbols still in trailing-stop cooldown.
        topk_override: reduced topk when soft drawdown active, else None.
        cooldown_entries: symbol -> {cooldown_until, holding_high} for
            symbols that triggered trailing stop this run.  Pipeline
            iterates this to call set_cooldown().
    """

    buying_halted: bool
    forced_sells: dict[str, int]
    blocked_industries: set[str]
    blocked_rebuys: set[str]
    topk_override: int | None
    cooldown_entries: dict[str, dict]


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_drawdown_breaker(
    peak_nav: float, current_nav: float, threshold: float,
) -> bool:
    """Hard circuit breaker: halt all buying if drawdown exceeds threshold.

    Selling is still allowed (D-35).
    """
    if peak_nav <= 0:
        return False
    drawdown = (peak_nav - current_nav) / peak_nav
    return drawdown > threshold


def check_daily_loss(
    yesterday_nav: float, today_nav: float, threshold: float,
) -> bool:
    """Halt buying for the day if single-day loss exceeds threshold (D-36).

    Returns False when yesterday_nav <= 0 (Day 1 guard).
    """
    if yesterday_nav <= 0:
        return False
    daily_loss = (yesterday_nav - today_nav) / yesterday_nav
    return daily_loss > threshold


def check_concentration(
    position_value: float, total_nav: float, cap: float,
) -> tuple[bool, float]:
    """Check whether a single position exceeds the NAV cap (D-37).

    Returns (is_over, excess_value).  excess_value tells the engine
    how much to sell down.
    """
    if total_nav <= 0:
        return (False, 0.0)
    pct = position_value / total_nav
    if pct > cap:
        return (True, position_value - cap * total_nav)
    return (False, 0.0)


def check_market_regime(
    csi1000_closes: list[float],
    decline_threshold: float,
    lookback_days: int,
) -> bool:
    """Halt buying if trailing N-day market decline exceeds threshold (D-38).

    Requires at least lookback_days + 1 data points.  Slices to the
    window so a longer series does not silently measure a wider span.
    No forced liquidation -- only blocks buying.
    """
    if len(csi1000_closes) < lookback_days + 1:
        return False
    window = csi1000_closes[-(lookback_days + 1):]
    if window[0] <= 0:
        return False
    cumulative_return = (window[-1] / window[0]) - 1
    return cumulative_return < -decline_threshold


def check_trailing_stop(
    holding_high: float, current_price: float, stop_pct: float,
) -> bool:
    """Trigger full-position sell if decline from holding-period high
    exceeds stop_pct (D-39).
    """
    if holding_high <= 0:
        return False
    decline = (holding_high - current_price) / holding_high
    return decline > stop_pct


def manage_trailing_cooldown(
    cooldown_dict: dict[str, dict], trade_date: str,
) -> dict[str, dict]:
    """Remove expired cooldown entries, retain active ones.

    Expiry uses strict less-than: cooldown_until < trade_date means
    the cooldown has fully elapsed.  Entries where
    cooldown_until >= trade_date are still active (the Nth cooldown
    day is inside the window).

    Returns a new dict with expired entries removed.
    """
    return {
        symbol: entry
        for symbol, entry in cooldown_dict.items()
        if entry["cooldown_until"] >= trade_date
    }


def check_industry_concentration(
    industry_positions: dict[str, float], total_nav: float, cap: float,
) -> set[str]:
    """Return set of CSRC industries where exposure exceeds cap (D-40)."""
    if total_nav <= 0:
        return set()
    return {
        industry
        for industry, value in industry_positions.items()
        if value / total_nav > cap
    }


def check_soft_drawdown(
    peak_nav: float,
    current_nav: float,
    is_currently_reduced: bool,
    soft_threshold: float,
    recovery_pct: float,
    default_topk: int,
    reduced_topk: int,
) -> int | None:
    """Reduce topk when soft drawdown active; restore on recovery (D-45).

    Uses is_currently_reduced to prevent oscillation: entry logic only
    fires when NOT already reduced, exit logic only fires when reduced.
    Returns reduced_topk to reduce, None to restore/keep default.
    """
    if peak_nav <= 0:
        return None

    if is_currently_reduced:
        # Exit logic: restore when NAV recovers to recovery_pct of peak
        if current_nav >= peak_nav * recovery_pct:
            return None
        return reduced_topk
    else:
        # Entry logic: reduce when drawdown exceeds soft threshold
        drawdown = (peak_nav - current_nav) / peak_nav
        if drawdown > soft_threshold:
            return reduced_topk
        return None


# ---------------------------------------------------------------------------
# Dimension 8: Prediction staleness (Phase 9 -- 09-02)
# ---------------------------------------------------------------------------


def check_prediction_staleness(
    trade_date: str, pred_date: str, max_stale_days: int,
) -> tuple[bool, dict]:
    """Check prediction staleness using TRADING DAY count.

    Returns (should_halt_buying, detail_dict).

    Tiers:
        FRESH (age 0): no action
        STALE_WARN (age 1-2): log warning, no auto-action
        STALE_REJECT (age >= max_stale_days): halt buying

    Uses qlib trading day calendar for accurate counting.
    Falls back to calendar-day arithmetic (conservative) if qlib
    unavailable.
    """
    age_days = 0
    try:
        from datetime import date as _date  # noqa: PLC0415
        td = _date.fromisoformat(trade_date)
        pd = _date.fromisoformat(pred_date)
        trading_days = trading_days_between(pd, td)
        # trading_days_between returns days in [start, end) so subtract 1
        # to get the gap (pred_date itself is day 0).
        age_days = max(0, len(trading_days) - 1)
    except Exception:
        # Conservative fallback: calendar days
        from datetime import date as _date  # noqa: PLC0415
        td = _date.fromisoformat(trade_date)
        pd = _date.fromisoformat(pred_date)
        cal_days = (td - pd).days
        # Trading days ~ calendar days * 5/7 (conservative: round up)
        age_days = max(0, cal_days)

    if age_days == 0:
        tier = "FRESH"
    elif age_days < max_stale_days:
        tier = "STALE_WARN"
    else:
        tier = "STALE_REJECT"

    should_halt = tier == "STALE_REJECT"

    detail = {
        "age_days": age_days,
        "tier": tier,
        "threshold": max_stale_days,
        "pred_date": pred_date,
    }

    if tier == "STALE_WARN":
        logger.warning(
            "Prediction staleness: %s is %d trading days old "
            "(threshold: %d). Tier: %s",
            pred_date, age_days, max_stale_days, tier,
        )
    elif tier == "STALE_REJECT":
        logger.error(
            "Prediction staleness REJECT: %s is %d trading days old "
            "(threshold: %d). Buying halted.",
            pred_date, age_days, max_stale_days,
        )

    return should_halt, detail


# ---------------------------------------------------------------------------
# Dimension 9: Suspension risk awareness (Phase 9 -- 09-03, alert-only)
# ---------------------------------------------------------------------------


def check_suspension_risk(
    positions: dict[str, dict],
    prices: dict[str, dict],
    trade_date: str,
) -> dict[str, float]:
    """Score suspension risk for held positions (alert-only, no auto-action).

    Scores 6 dimensions per symbol:
    1. ST status (5% limit = higher suspension risk)
    2. Zero volume (currently suspended)
    3. Near zero volume (< 10% of normal)
    4. Large position (> 10% NAV concentration)
    5. Low market cap proxy (price < 5 CNY)
    6. Extreme price change (|change| > 8%)

    Returns dict of symbol -> risk_score (0.0-1.0).
    Only logged to daily report, zero decision power.
    """
    scores: dict[str, float] = {}
    for symbol, pos in positions.items():
        pdata = prices.get(symbol, {})
        score = 0.0

        # Dimension 1: ST status (from limit threshold)
        threshold = pdata.get("threshold", 0.099)
        if threshold <= 0.05:
            score += 0.2

        # Dimension 2: currently suspended
        volume = pdata.get("volume", 0.0)
        if volume == 0.0 or (isinstance(volume, float) and volume != volume):
            score += 0.3

        # Dimension 3: near-zero volume
        elif volume < 1_000_000:
            score += 0.1

        # Dimension 4: large position concentration
        # (approximate: market_value > 50000 CNY)
        if pos.get("market_value", 0) > 50_000:
            score += 0.1

        # Dimension 5: low price proxy for small-cap risk
        close = pdata.get("close")
        if close is not None and close < 5.0:
            score += 0.1

        # Dimension 6: extreme price change
        change = pdata.get("change", 0.0)
        if abs(change) > 0.08:
            score += 0.2

        if score > 0:
            scores[symbol] = round(min(score, 1.0), 2)

    return scores


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_all_risk_checks(
    nav_history: list[dict],
    yesterday_nav: float,
    current_positions: dict[str, dict],
    current_prices: dict[str, dict],
    csi1000_closes: list[float],
    industry_map: dict[str, str],
    cooldown_dict: dict[str, dict],
    cash: float,
    is_soft_reduced: bool,
    config: dict,
    trade_date: str,
    pred_date: str | None = None,
) -> RiskCheckResult:
    """Aggregate all 8 risk dimensions into a single result.

    All thresholds are read from *config* (paper.risk section).
    *yesterday_nav* is passed explicitly by the pipeline to avoid
    mis-deriving it after record_nav has already written today's row.
    *trade_date* is required to compute cooldown_until dates.
    *pred_date* is the prediction file date for staleness check.
    """
    # -- Compute total/current NAV --
    total_nav = (
        sum(pos["market_value"] for pos in current_positions.values())
        + cash
    )
    current_nav = total_nav

    # -- Peak NAV from history (or current if no history) --
    if nav_history:
        peak_nav = max(row["total_nav"] for row in nav_history)
    else:
        peak_nav = current_nav

    # Ensure peak is at least current (first day edge case)
    if current_nav > peak_nav:
        peak_nav = current_nav

    # -- 1. Drawdown hard halt --
    drawdown_halted = check_drawdown_breaker(
        peak_nav, current_nav, config["drawdown_hard"],
    )

    # -- 2. Daily loss halt --
    daily_loss_halted = check_daily_loss(
        yesterday_nav, current_nav, config["daily_loss"],
    )

    # -- 3. Concentration check + forced sells --
    forced_sells: dict[str, int] = {}
    for symbol, pos in current_positions.items():
        is_over, excess_cny = check_concentration(
            pos["market_value"], total_nav, config["concentration"],
        )
        if is_over:
            close_price = (
                current_prices.get(symbol, {}).get("close", pos["avg_cost"])
            )
            if close_price <= 0:
                continue
            excess_qty = max(1, math.ceil(excess_cny / close_price))
            forced_sells[symbol] = min(excess_qty, pos["qty"])

    # -- 4. Market regime filter --
    regime_halted = check_market_regime(
        csi1000_closes,
        config["market_regime_decline"],
        config["market_regime_days"],
    )

    # -- 5. Trailing stop + cooldown entries --
    cooldown_entries: dict[str, dict] = {}
    for symbol, pos in current_positions.items():
        holding_high = pos.get("holding_high")
        if holding_high is None:
            continue
        close = current_prices.get(symbol, {}).get("close")
        if close is None:
            logger.warning(
                "Trailing stop: no close price for %s on %s (delisted?)",
                symbol,
                trade_date,
            )
            continue
        if check_trailing_stop(
            holding_high, close, config["trailing_stop"],
        ):
            # Full-position exit: assignment supersedes any partial
            # concentration sell for the same symbol (F-D)
            forced_sells[symbol] = pos["qty"]
            # Compute cooldown_until: next_trading_day applied N times
            cooldown_until_date = dt.date.fromisoformat(trade_date)
            assert config["trailing_cooldown_days"] >= 1, (
                "trailing_cooldown_days must be >= 1"
            )
            for _ in range(config["trailing_cooldown_days"]):
                cooldown_until_date = next_trading_day(cooldown_until_date)
            cooldown_entries[symbol] = {
                "cooldown_until": cooldown_until_date.isoformat(),
                "holding_high": holding_high,
            }

    # -- 6. Industry concentration --
    industry_positions_by_csrc: dict[str, float] = {}
    for s, pos in current_positions.items():
        ind = industry_map.get(s)
        if ind:
            industry_positions_by_csrc[ind] = (
                industry_positions_by_csrc.get(ind, 0.0)
                + pos["market_value"]
            )
    blocked_industries = check_industry_concentration(
        industry_positions_by_csrc, total_nav, config["industry_cap"],
    )

    # -- 7. Soft drawdown topk reduction --
    topk_override = check_soft_drawdown(
        peak_nav,
        current_nav,
        is_soft_reduced,
        config["soft_drawdown"],
        config["soft_drawdown_recovery"],
        config["default_topk"],
        config["reduced_topk"],
    )

    # -- 8. Prediction staleness --
    # Skip when pred_date is None (step 10 hasn't run yet).
    staleness_halted = False
    if pred_date is not None:
        max_stale = config.get("max_stale_trading_days", 3)
        staleness_halted, staleness_detail = check_prediction_staleness(
            trade_date, pred_date, max_stale,
        )
        if staleness_detail.get("tier") != "FRESH":
            logger.info(
                "Staleness check: pred_date=%s age=%d tier=%s halted=%s",
                pred_date,
                staleness_detail["age_days"],
                staleness_detail["tier"],
                staleness_halted,
            )

    # -- blocked_rebuys from cooldown_dict (>= for full N-day) --
    blocked_rebuys = {
        s
        for s, cd in cooldown_dict.items()
        if cd["cooldown_until"] >= trade_date
    }

    # -- Merge halt flags --
    buying_halted = (
        drawdown_halted or daily_loss_halted
        or regime_halted or staleness_halted
    )

    # -- Hard drawdown forced liquidation (emergency override) --
    # When drawdown_hard fires, force-sell ALL positions (full qty,
    # overriding any partial sells from concentration or trailing
    # stop). Previous code only halted buying, leaving portfolio
    # frozen between 15-20% drawdown with no exit path.
    if drawdown_halted:
        for symbol, pos in current_positions.items():
            forced_sells[symbol] = pos["qty"]
            logger.warning(
                "Hard drawdown forced sell: %s x%d",
                symbol, pos["qty"])

    return RiskCheckResult(
        buying_halted=buying_halted,
        forced_sells=forced_sells,
        blocked_industries=blocked_industries,
        blocked_rebuys=blocked_rebuys,
        topk_override=topk_override,
        cooldown_entries=cooldown_entries,
    )
