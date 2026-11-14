"""Rebuild every price series whose tail left the store's adjustment basis.

The store holds forward-adjusted prices: ``close = raw * factor``, where factor
is carried by the last good row. Daily rows appended by the writer hold the raw
close with factor 1.0, so each affected symbol has a seam and everything after it
is on a different basis. Some rows also carry another day's data entirely.

Inferring the correction from the stored ``change`` reproduces that damage, so
this rebuilds from the source of record:

    factor_i = factor_(i-1) * raw_(i-1) / pre_close_i
    close_i  = raw_i * factor_i        (open/high/low the same)

Stocks come from ``pro.daily``; indices are absent from that endpoint and come
from ``pro.index_daily``. A slot with no source row is written as NaN rather than
carried: no source means no claim, and qlib reads NaN as no data. Suspended tails
therefore stop claiming prices they cannot support.

Usage: python3 repair_price_basis.py [--apply]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_lab.data.fetcher import _get_tushare_token

STORE = Path.home() / ".qlib" / "qlib_data" / "cn_data"
CACHE = Path("/tmp/tsdaily")
FIELDS = ("open", "high", "low", "close")
INDEX_SYMBOLS = {"SH000300", "SH000852", "SH000905", "SH000906", "SZ399300"}


def read_bin(path: Path) -> tuple[int, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 8 or (len(raw) - 4) % 4:
        raise ValueError(f"malformed bin: {path}")
    return int(struct.unpack("<f", raw[:4])[0]), np.frombuffer(raw[4:], dtype="<f4")


def write_bin(path: Path, first: int, values: np.ndarray) -> None:
    path.write_bytes(struct.pack("<f", float(first)) + values.astype("<f4").tobytes())


def find_seam(factor: np.ndarray) -> int | None:
    for i in range(1, len(factor)):
        if abs(factor[i] - 1.0) < 1e-6 and abs(factor[i - 1] - 1.0) > 1e-3:
            return i
    return None


def ts_symbol(symbol: str) -> str:
    return f"{symbol[2:]}.{symbol[:2]}"


def load_stock_days(dates: list[str], pro) -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in dates:
        path = CACHE / f"{d}.parquet"
        if path.exists():
            out[d] = pd.read_parquet(path)
            continue
        df = pro.daily(trade_date=d.replace("-", ""))
        if df is None or df.empty:
            continue
        df = df.set_index(df["ts_code"].map(lambda c: c.split(".")[1] + c.split(".")[0]))
        df.to_parquet(path)
        out[d] = df
    return out


def main() -> int:
    apply = "--apply" in sys.argv
    cal = [
        line.strip()
        for line in (STORE / "calendars" / "day.txt").read_text().splitlines()
        if line.strip()
    ]

    broken: dict[str, tuple[Path, int, int]] = {}
    for d in sorted((STORE / "features").iterdir()):
        if not d.is_dir() or not (d / "factor.day.bin").exists():
            continue
        first, factor = read_bin(d / "factor.day.bin")
        factor = factor.astype(np.float64)
        if len(factor) < 2:
            continue
        seam = find_seam(factor)
        if seam is not None:
            broken[d.name.upper()] = (d, first, seam)

    if not broken:
        print("nothing to repair: no seam found")
        return 0

    start_date = min(cal[first + seam] for _, first, seam in broken.values())
    dates = [d for d in cal if d >= start_date]
    print(f"broken symbols: {len(broken)}; window {start_date}..{dates[-1]}")

    pro = __import__("tushare").pro_api(_get_tushare_token())
    stock_days = load_stock_days(dates, pro)
    print(f"source: {len(stock_days)} stock days")

    stats = {"rewritten": 0, "nan_rows": 0, "untouched": 0, "nan_filled": 0}
    samples = []

    for symbol, (sym_dir, first, seam) in sorted(broken.items()):
        factor = read_bin(sym_dir / "factor.day.bin")[1].astype(np.float64)
        close = read_bin(sym_dir / "close.day.bin")[1].astype(np.float64)
        prices = {f: read_bin(sym_dir / f"{f}.day.bin")[1].astype(np.float64) for f in FIELDS}
        change = read_bin(sym_dir / "change.day.bin")[1].astype(np.float64)
        n = len(close)

        # An index has no corporate action: its factor is constant, so the
        # repair is a rescale by the factor the series already carried.
        if symbol in INDEX_SYMBOLS:
            anchor = factor[seam - 1]
            for i in range(seam, n):
                for f in FIELDS:
                    if np.isfinite(prices[f][i]):
                        prices[f][i] = prices[f][i] * anchor
                factor[i] = anchor
            if apply:
                for f in FIELDS:
                    write_bin(sym_dir / f"{f}.day.bin", first, prices[f])
                write_bin(sym_dir / "factor.day.bin", first, factor)
            stats["rewritten"] += 1
            if len(samples) < 4:
                samples.append((symbol, cal[first + seam], anchor, anchor))
            continue

        anchor_factor = factor[seam - 1]
        prev_factor = anchor_factor
        prev_raw = close[seam - 1] / anchor_factor
        touched = 0

        for i in range(seam, n):
            date = cal[first + i]
            row = None
            if date in stock_days and symbol in stock_days[date].index:
                row = stock_days[date].loc[symbol]
            if row is None:
                for f in FIELDS:
                    prices[f][i] = np.nan
                change[i] = np.nan
                factor[i] = prev_factor
                stats["nan_filled"] += 1
                continue

            raw_close = float(row["close"])
            pre_close = float(row["pre_close"])
            if not np.isfinite(raw_close) or not np.isfinite(pre_close) or pre_close <= 0:
                for f in FIELDS:
                    prices[f][i] = np.nan
                factor[i] = prev_factor
                stats["nan_filled"] += 1
                continue

            prev_factor = prev_factor * (prev_raw / pre_close)
            factor[i] = prev_factor
            for f in FIELDS:
                if f in row.index:
                    prices[f][i] = float(row[f]) * prev_factor
            if "pct_chg" in row.index:
                change[i] = float(row["pct_chg"]) / 100.0
            prev_raw = raw_close
            touched += 1

        # A tail with no source row still needs writing: leaving it in place
        # would keep prices the store cannot support, on the wrong basis.
        filled = int(np.isnan(prices["close"][seam:]).sum())
        if touched or filled:
            stats["rewritten"] += 1
            if apply:
                for f in FIELDS:
                    write_bin(sym_dir / f"{f}.day.bin", first, prices[f])
                write_bin(sym_dir / "factor.day.bin", first, factor)
                write_bin(sym_dir / "change.day.bin", first, change)
            if len(samples) < 4:
                samples.append((symbol, cal[first + seam], anchor_factor, float(factor[-1])))
        else:
            stats["untouched"] += 1

    print("mode:", "APPLY" if apply else "dry-run", stats)
    for symbol, date, a, last in samples:
        print(f"  {symbol} seam {date}: factor {a:.6f} -> {last:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# Run this when a store carries rows on the wrong basis (see docs in the module
# docstring):  python3 scripts/repair_price_basis.py [--apply]
# It needs a tushare token, exactly like the daily fetch, and reads the store at
# ~/.qlib/qlib_data/cn_data. Dry-run first: it reports how many symbols and slots
# it would rewrite before touching anything.
