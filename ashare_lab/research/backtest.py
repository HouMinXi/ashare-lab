"""TopkDropout backtest with A-share cost model for one walk-forward window.

Provides run_backtest(). All qlib imports are deferred inside the function
body so this module is importable without a qlib runtime.
"""

from __future__ import annotations

import logging

import pandas as pd

from ashare_lab.config import load_config

log = logging.getLogger(__name__)


def run_backtest(
    window: dict,
    pred: pd.Series,
    n_drop: int,
    slippage_override: float | None,
) -> tuple:
    """Run a TopkDropoutStrategy backtest for one walk-forward window.

    Assumes qlib.init() is already active (called by train_window() upstream
    in the same process). Reads test_start and test_end from window; accepts
    either a WindowDict or a WindowResult as the window arg.

    Exchange and backtest_daily parameters are loaded from config and passed
    explicitly. n_drop overrides config strategy.<track>.n_drop.

    Args:
        window: Dict with at least "test_start" and "test_end" (ISO strings).
        pred: MultiIndex Series (datetime, instrument) -> float. Post-price-
            filter predictions. May be empty (all filtered -> cash-only).
        n_drop: Number of positions dropped per rebalance. Overrides config.
        slippage_override: Slippage fraction (e.g. 0.001). If None, uses
            config["cost_model"]["slippage"].

    Returns:
        Tuple (portfolio_df, bench_close, lot_skip_count):
            portfolio_df: DataFrame with single DatetimeIndex and "return"
                column (daily portfolio return).
            bench_close: Series with single DatetimeIndex of benchmark close
                prices (MultiIndex xs-unpacked, no trailing column lookup).
            lot_skip_count: int count of orders where order_amount != trade_amount
                (lot-size skips). 0 if no trades. None if indicator structure
                is unrecognised (missing order_amount/trade_amount columns).
    """
    from qlib.contrib.evaluate import backtest_daily  # noqa: PLC0415
    from qlib.contrib.strategy import TopkDropoutStrategy  # noqa: PLC0415
    from qlib.data import D  # noqa: PLC0415

    cfg = load_config()
    cm = cfg["cost_model"]
    benchmark_symbol: str = cm["benchmark"]

    test_start: str = window["test_start"]
    test_end: str = window["test_end"]

    slippage = slippage_override if slippage_override is not None else cm["slippage"]

    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=cfg["strategy"]["topk"],
        n_drop=n_drop,
    )

    # Build Exchange kwargs dict to pass to backtest_daily.
    exchange_kwargs = {
        "impact_cost": slippage,
        "trade_unit": cm["trade_unit"],
        "open_cost": cm["open_cost"],
        "close_cost": cm["close_cost"],
        "min_cost": cm["min_cost"],
        "deal_price": cm["deal_price"],
        "limit_threshold": cm["limit_threshold"],
    }

    log.info(
        "backtest: %s..%s | topk=%d n_drop=%d slippage=%.4f",
        test_start,
        test_end,
        cfg["strategy"]["topk"],
        n_drop,
        slippage,
    )

    portfolio_metric, indicator = backtest_daily(
        start_time=test_start,
        end_time=test_end,
        strategy=strategy,
        account=cm["account"],
        benchmark=benchmark_symbol,
        exchange_kwargs=exchange_kwargs,
    )

    # portfolio_metric may be a freq-dict {"1day": DataFrame} or a DataFrame.
    if isinstance(portfolio_metric, dict):
        portfolio_df: pd.DataFrame = portfolio_metric["1day"]
    else:
        portfolio_df = portfolio_metric

    # Fetch benchmark close with 14-day pre-fetch for pct_change alignment.
    test_start_ts = pd.Timestamp(test_start)
    pre_fetch_start = (test_start_ts - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    bench_df = D.features(
        instruments=[benchmark_symbol],
        fields=["$close"],
        start_time=pre_fetch_start,
        end_time=test_end,
    )
    # xs removes the "instrument" level; result has a single DatetimeIndex.
    # In pandas 2.x, xs on a single-column DataFrame returns a DataFrame, not
    # a Series.  Extract the "$close" column to guarantee a Series.
    bench_close: pd.Series = bench_df.xs(benchmark_symbol, level="instrument")["$close"]

    # Unpack indicator: may be a freq-dict {"1day": date-keyed dict} or
    # directly a date-keyed dict (date -> DataFrame).
    if isinstance(indicator, dict) and "1day" in indicator:
        indicator = indicator["1day"]

    # indicator is now expected to be a date-keyed dict.
    lot_skip_count: int | None
    if isinstance(indicator, dict):
        if len(indicator) == 0:
            # No trades -> no skipped orders.
            lot_skip_count = 0
        else:
            try:
                all_df = pd.concat(indicator.values())
            except (TypeError, ValueError) as exc:
                log.warning("indicator concat failed: %s; setting lot_skip_count=None", exc)
                return portfolio_df, bench_close, None

            if "order_amount" in all_df.columns and "trade_amount" in all_df.columns:
                diff = (all_df["order_amount"] - all_df["trade_amount"]).abs()
                lot_skip_count = int((diff > 1e-6).sum())
            else:
                log.warning(
                    "indicator lacks order_amount/trade_amount; columns: %s",
                    list(all_df.columns),
                )
                lot_skip_count = None
    else:
        # Unrecognised structure.
        log.warning(
            "indicator is neither freq-dict nor date-dict (type=%s); "
            "setting lot_skip_count=None",
            type(indicator).__name__,
        )
        lot_skip_count = None

    log.info(
        "backtest done: portfolio rows=%d, lot_skip_count=%s",
        len(portfolio_df),
        lot_skip_count,
    )

    return portfolio_df, bench_close, lot_skip_count
