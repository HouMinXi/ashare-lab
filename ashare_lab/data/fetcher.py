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


def fetch_today_data(trade_date: str) -> dict[str, pd.DataFrame]:
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
                }
            ]
        )
    return result


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
        import akshare as ak  # noqa: PLC0415
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
