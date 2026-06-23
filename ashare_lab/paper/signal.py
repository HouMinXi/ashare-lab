"""Signal generation, TopkDropout order logic, and candidate filtering.

Reads the daily prediction file (predictions/{trade_date}.parquet) produced
by the upstream producer (research/predict.py).  The engine never loads a
model, never imports torch, and never builds Alpha158 -- it depends on the
prediction-file contract, not on the producer code.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_lab.config import PREDICTIONS_DIR, load_config
from ashare_lab.paper.ledger import insert_signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TopkDropout order generation (D-20)
# ---------------------------------------------------------------------------


def topk_dropout_orders(
    signals: dict[str, float],
    current_positions: set[str],
    topk: int,
    n_drop: int,
) -> tuple[list[str], list[str]]:
    """Generate buy/sell lists via TopkDropout rebalancing.

    Rank all signals descending by score.  Sell positions not in top-K,
    capped to *n_drop* worst-ranked per day (gradual exit during soft
    drawdown).  Buy refills to *topk* (cold start fills the whole book;
    steady state buys n_drop; soft-drawdown shrink buys 0).

    Returns (sell_list, buy_list).
    """
    if not signals:
        return [], []

    ranked = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
    top_k_symbols = {sym for sym, _ in ranked[:topk]}

    # Sell: positions not in top-K, capped to n_drop worst-ranked
    sell_cands = [s for s in current_positions if s not in top_k_symbols]
    sell_cands.sort(key=lambda s: signals.get(s, float("-inf")))
    sell_list = sell_cands[:n_drop]

    # Buy: refill to topk
    candidates = [sym for sym, _ in ranked[:topk] if sym not in current_positions]
    keep = len(current_positions) - len(sell_list)
    n_buy = max(0, topk - keep)
    buy_list = candidates[:n_buy]

    return sell_list, buy_list


# ---------------------------------------------------------------------------
# Candidate filtering (D-17, D-44, close cap)
# ---------------------------------------------------------------------------


def filter_candidates(
    symbols: list[str],
    market_data: dict[str, dict],
    listing_min_days: int = 60,
    liquidity_min_turnover: float = 50_000_000.0,
    close_max: float = 300.0,
) -> list[str]:
    """Return symbols passing listing-age, liquidity, and price filters.

    *market_data* maps symbol to a dict with keys: close, listing_days,
    avg_turnover_20d.  The pipeline pre-computes these so this function
    stays pure.
    """
    passed: list[str] = []
    n_listing = 0
    n_liquidity = 0
    n_price = 0
    n_missing = 0

    for sym in symbols:
        md = market_data.get(sym)
        if md is None:
            n_missing += 1
            continue

        if md["listing_days"] < listing_min_days:
            n_listing += 1
            continue

        if md["avg_turnover_20d"] < liquidity_min_turnover:
            n_liquidity += 1
            continue

        if md["close"] > close_max:
            n_price += 1
            continue

        passed.append(sym)

    if n_listing or n_liquidity or n_price or n_missing:
        logger.info(
            "filter_candidates: %d passed, excluded: "
            "listing=%d liquidity=%d price=%d missing=%d",
            len(passed), n_listing, n_liquidity, n_price, n_missing,
        )
    return passed


# ---------------------------------------------------------------------------
# generate_signals -- read the daily prediction file (Option 3 decouple)
# ---------------------------------------------------------------------------


def generate_signals(
    trade_date: str,
    conn: sqlite3.Connection | None = None,
    pred_path: Path | None = None,
    topk: int = 15,
) -> dict[str, float]:
    """Read the daily prediction file and return instrument -> score.

    The prediction file is produced upstream by research/predict.py.  The
    score column is the already-blended 60/40 TRA/nTRA output, consumed as
    an opaque float.

    Validates schema, coverage, and score sanity before returning.  Writes
    the TopK symbols to the signals table when *conn* is provided.

    Raises FileNotFoundError if the prediction file is missing.
    Raises ValueError on malformed/short/non-finite data.
    """
    if pred_path is None:
        pred_path = PREDICTIONS_DIR / f"{trade_date}.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}"
        )

    df = pd.read_parquet(pred_path)

    # Schema check
    for col in ("instrument", "score"):
        if col not in df.columns:
            raise ValueError(
                f"Prediction file missing required column: {col}"
            )

    # Coverage check
    cfg = load_config()
    min_cov = cfg["paper"]["min_signal_coverage"]
    if len(df) < min_cov:
        raise ValueError(
            f"Prediction file has {len(df)} rows, "
            f"minimum coverage requires {min_cov}"
        )

    # Score sanity (defense-in-depth)
    if not pd.api.types.is_numeric_dtype(df["score"]):
        raise ValueError("score column is not numeric")
    if df["score"].isna().any():
        raise ValueError("score column contains NaN values")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise ValueError("score column contains non-finite values")

    scores = dict(zip(df["instrument"], df["score"]))

    # Duplicate instrument guard
    if len(scores) != len(df):
        raise ValueError("duplicate instrument rows in prediction file")

    # Write TopK to signals table for audit
    if conn is not None:
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        signals_list = [
            {"symbol": sym, "score": sc, "rank": i + 1}
            for i, (sym, sc) in enumerate(top)
        ]
        insert_signals(conn, trade_date, signals_list)

    return scores
