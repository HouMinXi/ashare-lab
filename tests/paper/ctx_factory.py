"""One DailyRunContext builder for tests that drive a daily step.

Four private builders had drifted apart: one omitted `industry_map` and
`benchmarks`, which stayed invisible only because that test's buy list was
empty; another left `volume` out of its prices, so a fill looked like a
suspension.  Anything the steps read has to be set here once.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from ashare_lab.paper.pipeline import DailyRunContext
from ashare_lab.paper.risk import RiskCheckResult


class _Unset:
    """Distinguishes "use the default" from an explicit None."""


_UNSET = _Unset()


def make_daily_ctx(
    trade_date: str = "2025-06-20",
    *,
    config: dict | None = None,
    paper_cfg: dict | None = None,
    conn=None,
    db_path: Path | None = None,
    pred_path: Path | None = None,
    prices: dict | None = None,
    positions: dict | None = None,
    cash: float = 300_000.0,
    total_nav: float | None = None,
    buying_halted: bool = False,
    forced_sells: dict | None = None,
    universe: list | None = None,
    market_data: dict | None = None,
    signals: dict | None = None,
    hedge_state=_UNSET,
    hedge_symbols: set | None = None,
):
    ctx = DailyRunContext(
        trade_date=trade_date, force=False, steps=None,
        pred_path=pred_path, start_time=0.0, predictions_date_str=trade_date,
    )
    if conn is not None:
        ctx.conn = conn
    if db_path is not None:
        ctx.db_path = db_path

    ctx.config = config if config is not None else {"paper": paper_cfg or {}}
    ctx.paper_cfg = paper_cfg if paper_cfg is not None else ctx.config.get("paper", {})
    ctx.prices = prices or {}
    ctx.current_positions = positions or {}
    ctx.cash = cash
    ctx.total_nav = cash if total_nav is None else total_nav
    ctx.signals_raw = signals or {}
    ctx.risk_result = RiskCheckResult(
        buying_halted=buying_halted,
        forced_sells=forced_sells or {},
        blocked_industries=set(),
        blocked_rebuys=set(),
        topk_override=None,
        cooldown_entries={},
        suspension_risk={},
        shadow_log=None,
    )
    ctx.hedge_state = MagicMock(active=False) if isinstance(hedge_state, _Unset) else hedge_state
    ctx.hedge_symbols = hedge_symbols or set()
    ctx.ipo_listing_syms = set()
    ctx.st_names = set()
    ctx.industry_map = {}
    ctx.benchmarks = {"csi1000": 1.0}
    ctx.universe_symbols = list(ctx.prices) if universe is None else universe
    ctx.market_data = market_data if market_data is not None else {
        sym: {"close": p.get("close", 0.0), "listing_days": 100,
              "avg_turnover_20d": 1e8}
        for sym, p in ctx.prices.items()
    }
    return ctx
