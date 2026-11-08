"""Tushare-based same-day data fetch for qlib incremental update.

Primary source: tushare pro.daily() returns all ~5510 A-share stocks for a
given trade date. Output format matches fallback.py _CSV_COLUMNS so the
existing _write_csvs / _dump_bin_update pipeline works without change.

Cross-validation: AKShare stock_zh_a_hist() on a small sample of stocks
to spot-check tushare data. Non-fatal on failure.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd
import tushare as ts

log = logging.getLogger(__name__)

_CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "factor", "change"]

# ~10 CSI1000 stocks for cross-validation sampling
CSI1000_SAMPLE_SYMBOLS: list[str] = [
    "sz000002",
    "sz000063",
    "sz000100",
    "sz000157",
    "sz000338",
    "sh601012",
    "sh601066",
    "sh601138",
    "sh601216",
    "sh601318",
]

# AKShare Chinese column name mapping
_AKSHARE_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "涨跌幅": "pct_chg",
}


def _get_tushare_token() -> str:
    """Read tushare API token from pass store."""
    r = subprocess.run(
        ["pass", "show", "api/tushare"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return r.stdout.strip()


def _tushare_code_to_qlib(ts_code: str) -> str:
    """Convert tushare code to lowercase qlib symbol.

    '000001.SZ' -> 'sz000001', '600000.SH' -> 'sh600000'
    """
    code, exchange = ts_code.split(".")
    return f"{exchange.lower()}{code}"


def fetch_today_data(
    trade_date: str,
    provider_uri: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all stocks via tushare pro.daily() for one trading day.

    Args:
        trade_date: Date string, accepts 'YYYY-MM-DD' or 'YYYYMMDD'.

    Returns:
        Dict mapping qlib symbol (e.g. 'sz000001') to single-row DataFrame
        with columns matching _CSV_COLUMNS.

    Raises:
        RuntimeError: If tushare returns empty or None.
    """
    token = _get_tushare_token()
    pro = ts.pro_api(token)
    date_compact = trade_date.replace("-", "")
    df = pro.daily(trade_date=date_compact)

    if df is None or df.empty:
        raise RuntimeError(f"tushare pro.daily returned empty for {trade_date}")

    # Normalize trade_date to YYYY-MM-DD for CSV output
    if "-" in trade_date:
        date_str = trade_date
    else:
        date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

    result: dict[str, pd.DataFrame] = {}
    for _, row in df.iterrows():
        symbol = _tushare_code_to_qlib(row["ts_code"])
        result[symbol] = pd.DataFrame(
            [
                {
                    "date": date_str,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["vol"]),
                    "factor": 1.0,
                    "change": float(row["pct_chg"]) / 100.0,
                    # Restated previous close: its ratio to yesterday's raw
                    # close is exactly the corporate action the factor absorbs.
                    "pre_close": float(row["pre_close"]),
                }
            ]
        )
    return _apply_store_basis(result, date_str, provider_uri)


def _apply_store_basis(
    rows: dict[str, pd.DataFrame],
    trade_date: str,
    provider_uri: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Put today's raw prices on the adjustment basis the store already uses.

    ``pro.daily`` returns unadjusted prices. The store holds forward-adjusted
    ones (``close = raw * factor``, factor carried forward from the last dump),
    so appending raw rows with ``factor`` 1.0 splits one series into two bases.
    Yesterday's close is scaled by the cumulative factor and today's is not; the
    gap is a corporate action read as a price move, and a stock with a factor of
    0.07 posts a 1,300% day.

    Each row's factor is the last stored factor times the corporate action
    implied by ``pre_close``.

    Raises:
        RuntimeError: when a symbol has history that cannot be read, when a
            symbol's factor cannot be computed, or when the batch looks
            systematically wrong (see
            :func:`ashare_lab.data.adjust.systemic_break`). Nothing is written in
            those cases: a row on the wrong basis is worse than no row, because
            the next run cannot tell it from a real price.
    """
    import datetime as dt
    import math

    from ashare_lab.data.adjust import (
        BREAK_RETURN,
        adjusted_return,
        next_factor,
        scale_prices,
        systemic_break,
    )
    from ashare_lab.data.calendar import previous_trading_day
    from ashare_lab.data.store_state import calendar, feature_dir, last_state
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI

    uri = Path(provider_uri) if provider_uri is not None else DEFAULT_PROVIDER_URI
    out: dict[str, pd.DataFrame] = {}
    returns: dict[str, float] = {}
    skipped: list[str] = []

    # The session the store must already reach. Taken from the exchange, not
    # from the store calendar: on the first run of a day the calendar gains the
    # date only later, in _dump_bin_update, so a store-calendar comparison would
    # never fire on the path it is meant to guard.
    prev_session = previous_trading_day(dt.date.fromisoformat(trade_date)).isoformat()
    stored = calendar(uri)
    if stored and stored[-1] < prev_session:
        raise RuntimeError(
            f"store calendar ends {stored[-1]}, before {prev_session} (the session "
            f"preceding {trade_date}); backfill the hole before appending"
        )

    for symbol, df in rows.items():
        row = df.iloc[0].to_dict()
        state = last_state(uri, symbol)
        if state is None:
            if feature_dir(uri, symbol).is_dir():
                # Files exist but no readable row: the tail is unwritten or
                # corrupt. Guessing factor 1.0 here is how the defect returns.
                raise RuntimeError(
                    f"{symbol}: feature files exist but hold no readable row; "
                    "refusing to append on an unknown basis"
                )
            # A symbol with no directory at all is new: its series starts today,
            # so factor 1.0 is its own basis rather than a contradiction.
            out[symbol] = df
            continue

        prev, prev_date = state
        if prev_date >= trade_date:
            # The day is already stored for this symbol. Dropping it keeps the
            # writer from touching a row the store already owns.
            log.debug("%s already stored at %s; skipping", symbol, prev_date)
            continue

        # A symbol may lag the calendar because it was suspended -- its
        # pre_close is then the last price it traded, so the ratio is one -- or
        # because its own rows are missing for sessions it traded, in which case
        # the ratio carries the missed move and would be written as an action.
        # The two are indistinguishable from one day's data, so an unexplained
        # gap is skipped and reported rather than absorbed.
        if prev_date < prev_session:
            drift = abs(prev.raw_close / float(row["pre_close"]) - 1.0)
            if drift > 1e-3:
                log.warning(
                    "%s: stored history ends %s, before %s; skipping this row "
                    "(implied move %.2f%% is not a corporate action)",
                    symbol, prev_date, prev_session, drift * 100,
                )
                skipped.append(symbol)
                continue

        prices = {k: float(row[k]) for k in ("open", "high", "low", "close")}
        for field, value in prices.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{symbol}: {field} is not a usable price ({value!r})")

        factor = next_factor(prev, float(row["pre_close"]))
        scaled = scale_prices(prices, factor)
        new_row = dict(row)
        new_row.update(scaled)
        new_row["factor"] = factor
        out[symbol] = pd.DataFrame([new_row])
        returns[symbol] = adjusted_return(prev, scaled["close"])

    considered = len(returns) + len(skipped)
    if considered and len(skipped) / considered > 0.05:
        raise RuntimeError(
            f"{len(skipped)} symbols lag the calendar by more than one session on "
            f"{trade_date}; that is a store hole, not a suspension. Backfill first. "
            f"First offenders: {', '.join(skipped[:5])}"
        )

    broken = systemic_break(returns)
    if broken:
        raise RuntimeError(
            f"{len(broken)} of {len(returns)} symbols move more than "
            f"{BREAK_RETURN:.0%} after adjustment on {trade_date}; refusing to write. "
            f"First offenders: {', '.join(broken[:5])}"
        )

    # pre_close is scaffolding for the adjustment above; the frame that leaves
    # this function keeps the CSV contract every caller already expects.
    for symbol, df in out.items():
        out[symbol] = df[_CSV_COLUMNS]
    return out


def fetch_cross_validation_sample(
    trade_date: str, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    """Fetch individual stocks via AKShare for cross-validation.

    Imports akshare lazily to avoid numpy version conflicts on hosts
    where pyqlib pins numpy<2.0.

    Args:
        trade_date: Date string (YYYY-MM-DD or YYYYMMDD).
        symbols: Qlib-format symbols (e.g. ['sz000002', 'sh601318']).

    Returns:
        Dict mapping symbol to single-row DataFrame with _CSV_COLUMNS.
        Empty dict if akshare unavailable or all symbols fail.
    """
    try:
        import akshare as ak
    except ImportError:
        log.warning("akshare not available, skipping cross-validation")
        return {}

    date_compact = trade_date.replace("-", "")
    if "-" in trade_date:
        date_str = trade_date
    else:
        date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

    results: dict[str, pd.DataFrame] = {}
    for sym in symbols[:10]:
        code = sym[2:]  # strip sh/sz/bj prefix
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, start_date=date_compact, end_date=date_compact
            )
            if df is None or df.empty:
                log.warning("AKShare returned no data for %s", sym)
                continue

            row = df.iloc[0]
            results[sym] = pd.DataFrame(
                [
                    {
                        "date": date_str,
                        "open": float(row["开盘"]),
                        "high": float(row["最高"]),
                        "low": float(row["最低"]),
                        "close": float(row["收盘"]),
                        "volume": float(row["成交量"]),
                        "factor": 1.0,
                        "change": float(row["涨跌幅"]) / 100.0,
                    }
                ]
            )
        except Exception as exc:
            log.warning("AKShare cross-validation failed for %s: %s", sym, exc)

    return results


def refresh_stock_names_cache(cache_path: Path) -> dict[str, str]:
    """Fetch all stock names via tushare stock_basic(), save as CSV.

    Args:
        cache_path: Where to write the CSV (columns: symbol, name).

    Returns:
        Dict mapping qlib symbol to Chinese stock name.
    """
    token = _get_tushare_token()
    pro = ts.pro_api(token)
    df = pro.stock_basic(fields="ts_code,name")

    names: dict[str, str] = {}
    for _, row in df.iterrows():
        qlib_sym = _tushare_code_to_qlib(row["ts_code"])
        names[qlib_sym] = row["name"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        list(names.items()), columns=["symbol", "name"]
    ).to_csv(cache_path, index=False)
    log.info("stock name cache written: %s (%d entries)", cache_path, len(names))
    return names
