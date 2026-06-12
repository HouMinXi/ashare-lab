"""Cross-source data validation for qlib cn_data."""

from __future__ import annotations

import logging
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from ashare_lab.data.calendar import latest_trading_day, trading_days_between
from ashare_lab.data.update import DEFAULT_PROVIDER_URI

log = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["SH600000", "SZ000001", "SH601318"]
RAW_TOLERANCE = 0.01
RETURN_SIGN_THRESHOLD = 0.95


def _qlib_code_to_baostock(code: str) -> str:
    return f"{code[:2].lower()}.{code[2:]}"


def _load_qlib_feature(
    provider_uri: Path, symbol: str, field: str
) -> pd.Series | None:
    bin_path = provider_uri / "features" / symbol.lower() / f"{field}.day.bin"
    if not bin_path.exists():
        return None
    data = np.fromfile(bin_path, dtype="<f")
    if len(data) < 2:
        return None
    start_idx = int(data[0])
    values = data[1:]

    cal_file = provider_uri / "calendars" / "day.txt"
    if not cal_file.exists():
        return None
    dates = cal_file.read_text().strip().splitlines()
    end_idx = start_idx + len(values)
    if end_idx > len(dates):
        end_idx = len(dates)
        values = values[: end_idx - start_idx]
    idx = pd.DatetimeIndex(dates[start_idx:end_idx])
    return pd.Series(values, index=idx, name=field)


def _fetch_baostock_raw(
    symbol: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    bs_code = _qlib_code_to_baostock(symbol)
    lg = bs.login()
    if lg.error_code != "0":
        log.warning("baostock login failed: %s", lg.error_msg)
        return None
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    finally:
        bs.logout()


def spot_check_raw(
    symbols: list[str] | None = None,
    lookback: int = 5,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> dict:
    """Compare raw close prices between qlib and baostock (should be identical)."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    ltd = latest_trading_day()
    days = trading_days_between(
        ltd.replace(day=max(1, ltd.day - lookback * 2)), ltd
    )
    if len(days) > lookback:
        days = days[-lookback:]
    start_str = days[0].isoformat()
    end_str = days[-1].isoformat()

    results = {}
    for sym in symbols:
        qlib_close = _load_qlib_feature(provider_uri, sym, "close")
        qlib_factor = _load_qlib_feature(provider_uri, sym, "factor")
        if qlib_close is None or qlib_factor is None:
            results[sym] = {"status": "SKIP", "reason": "no qlib data"}
            continue

        bs_df = _fetch_baostock_raw(sym, start_str, end_str)
        if bs_df is None or bs_df.empty:
            results[sym] = {"status": "SKIP", "reason": "no baostock data"}
            continue

        qlib_raw = qlib_close / qlib_factor

        common = qlib_raw.index.intersection(bs_df.index)
        if len(common) == 0:
            results[sym] = {"status": "SKIP", "reason": "no overlapping dates"}
            continue

        qlib_vals = qlib_raw.loc[common]
        bs_vals = bs_df.loc[common, "close"]
        diff = (qlib_vals - bs_vals).abs()
        max_diff = diff.max()

        if max_diff > RAW_TOLERANCE:
            results[sym] = {
                "status": "FAIL",
                "max_diff": float(max_diff),
                "dates_checked": len(common),
            }
            log.warning("%s raw price mismatch: max_diff=%.4f", sym, max_diff)
        else:
            results[sym] = {
                "status": "PASS",
                "max_diff": float(max_diff),
                "dates_checked": len(common),
            }
            log.info("%s raw price check PASS (%d dates)", sym, len(common))

    return results


def check_return_consistency(
    symbols: list[str] | None = None,
    lookback: int = 20,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> dict:
    """Check that daily returns have consistent sign across sources."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    ltd = latest_trading_day()
    days = trading_days_between(
        ltd.replace(day=max(1, ltd.day - lookback * 2)), ltd
    )
    start_str = days[0].isoformat() if days else ltd.isoformat()
    end_str = days[-1].isoformat() if days else ltd.isoformat()

    results = {}
    for sym in symbols:
        qlib_close = _load_qlib_feature(provider_uri, sym, "close")
        qlib_factor = _load_qlib_feature(provider_uri, sym, "factor")
        bs_df = _fetch_baostock_raw(sym, start_str, end_str)
        if qlib_close is None or qlib_factor is None or bs_df is None or bs_df.empty:
            results[sym] = {"status": "SKIP"}
            continue

        qlib_raw = qlib_close / qlib_factor

        common = qlib_raw.index.intersection(bs_df.index)
        if len(common) < 3:
            results[sym] = {"status": "SKIP", "reason": "insufficient overlap"}
            continue

        q_ret = qlib_raw.loc[common].pct_change().dropna()
        b_ret = bs_df.loc[common, "close"].pct_change().dropna()
        overlap = q_ret.index.intersection(b_ret.index)
        if len(overlap) == 0:
            results[sym] = {"status": "SKIP"}
            continue

        sign_match = (np.sign(q_ret.loc[overlap]) == np.sign(b_ret.loc[overlap])).mean()
        results[sym] = {
            "status": "PASS" if sign_match >= RETURN_SIGN_THRESHOLD else "WARN",
            "sign_match_rate": float(sign_match),
            "dates_checked": len(overlap),
        }
    return results


def validate_instruments(
    market: str = "csi300",
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> dict:
    """Check instruments file has expected entries and current date ranges."""
    inst_file = provider_uri / "instruments" / f"{market}.txt"
    if not inst_file.exists():
        return {"status": "FAIL", "reason": f"{inst_file} not found"}

    df = pd.read_csv(inst_file, sep="\t", header=None, names=["symbol", "start", "end"])
    total_rows = len(df)

    df["end"] = pd.to_datetime(df["end"])
    ltd = pd.Timestamp(latest_trading_day())
    current = df[df["end"] >= ltd - pd.Timedelta(days=10)]
    active_count = current["symbol"].nunique()

    if active_count < 250 or active_count > 350:
        return {
            "status": "WARN",
            "reason": f"expected ~300 active constituents, got {active_count}",
            "total_rows": total_rows,
            "active_count": active_count,
        }

    return {
        "status": "PASS",
        "total_rows": total_rows,
        "active_count": active_count,
        "latest_end": str(df["end"].max().date()),
    }
