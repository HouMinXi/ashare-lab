"""Baostock-based gap-fill for qlib cn_data when chenditc release is stale.

Entry point: ``gap_fill(from_date, to_date)`` -- identifies trading dates
absent from the qlib calendar, fetches OHLCV + adjustfactor (qfq) from
baostock, and writes the data into the binary feature store via
``qlib.scripts.dump_bin update``.

The baostock TCP session expires after ~30 s; the session is refreshed
every ``fallback.max_symbols_per_session`` symbols (from pipeline.yaml).
"""

from __future__ import annotations

import csv as csv_mod
import datetime as dt
import logging
import os
import struct
import tempfile
from pathlib import Path

import pandas as pd

from ashare_lab.data.calendar import trading_days_between
from ashare_lab.data.update import DEFAULT_PROVIDER_URI

log = logging.getLogger(__name__)

# Fallback default when pipeline.yaml is unavailable
_DEFAULT_SESSION_REFRESH_N = 50

# baostock adjustflag=1 means qfq (forward-adjusted prices)
_BAOSTOCK_ADJUST_FLAG = "1"
_BAOSTOCK_FIELDS = "date,open,high,low,close,volume,pctChg,adjustfactor"
_BAOSTOCK_FREQ = "d"

# Columns written to per-symbol CSVs (consumed by qlib dump_bin update)
_CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "factor", "change"]


# ---------------------------------------------------------------------------
# Symbol format helpers
# ---------------------------------------------------------------------------


def _bs_to_qlib(bs_code: str) -> str:
    """Convert baostock code to qlib symbol. 'sz.000001' -> 'SZ000001'."""
    parts = bs_code.split(".", 1)
    if len(parts) != 2:
        return bs_code
    return f"{parts[0].upper()}{parts[1]}"


def _qlib_to_bs(qlib_code: str) -> str:
    """Convert qlib symbol to baostock code. 'SZ000001' -> 'sz.000001'."""
    if len(qlib_code) < 3:
        return qlib_code
    return f"{qlib_code[:2].lower()}.{qlib_code[2:]}"


# ---------------------------------------------------------------------------
# Local data inspection
# ---------------------------------------------------------------------------


def _calendar_dates(provider_uri: Path) -> set[str]:
    """Return all YYYY-MM-DD dates present in the qlib calendar file."""
    cal = provider_uri / "calendars" / "day.txt"
    if not cal.exists():
        return set()
    return {line.strip() for line in cal.read_text().splitlines() if line.strip()}


def _all_instruments(provider_uri: Path) -> list[str]:
    """Collect all unique qlib-format symbols from ``instruments/*.txt``."""
    inst_dir = provider_uri / "instruments"
    if not inst_dir.exists():
        return []
    seen: set[str] = set()
    for f in inst_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                seen.add(parts[0])
    return sorted(seen)


def find_missing_dates(
    from_date: str,
    to_date: str,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> list[str]:
    """Return trading dates in [from_date, to_date] absent from the qlib calendar.

    Args:
        from_date: Inclusive start date (YYYY-MM-DD).
        to_date: Inclusive end date (YYYY-MM-DD).
        provider_uri: qlib data root; must contain ``calendars/day.txt``.

    Returns:
        Sorted list of YYYY-MM-DD strings for missing trading days.
    """
    existing = _calendar_dates(provider_uri)
    start = dt.date.fromisoformat(from_date)
    end = dt.date.fromisoformat(to_date)
    return [
        d.isoformat()
        for d in trading_days_between(start, end)
        if d.isoformat() not in existing
    ]


# ---------------------------------------------------------------------------
# baostock fetch
# ---------------------------------------------------------------------------


def _fetch_symbol(
    bs,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch OHLCV + adjustfactor for one symbol from baostock.

    Args:
        bs: pre-imported and logged-in baostock module.
        symbol: qlib-format symbol (e.g. 'SZ000001').
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.

    Returns:
        List of row dicts with keys: date, open, high, low, close, volume,
        factor, change. Empty list if no data or all days suspended.

    Suspended days (empty or zero close) are skipped silently.
    Rows with unparseable numeric fields are skipped with a DEBUG log.
    """
    rs = bs.query_history_k_data_plus(
        _qlib_to_bs(symbol),
        fields=_BAOSTOCK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency=_BAOSTOCK_FREQ,
        adjustflag=_BAOSTOCK_ADJUST_FLAG,
    )
    rows: list[dict] = []
    while rs.error_code == "0" and rs.next():
        try:
            date_, open_, high_, low_, close_, vol_, pct_, adj_ = rs.get_row_data()
        except (ValueError, TypeError) as exc:
            log.debug("skipping %s row: field unpack error %s", symbol, exc)
            continue
        # Skip suspended / missing-data bars
        if not close_ or close_ in ("", "0"):
            continue
        try:
            rows.append(
                {
                    "date": date_,
                    "open": float(open_),
                    "high": float(high_),
                    "low": float(low_),
                    "close": float(close_),
                    "volume": float(vol_),
                    # adjustfactor=1.0 when unavailable (e.g. first listing day)
                    "factor": float(adj_) if adj_ else 1.0,
                    # pctChg is percentage; qlib 'change' is fractional return
                    "change": float(pct_) / 100.0 if pct_ else 0.0,
                }
            )
        except (ValueError, TypeError) as exc:
            log.debug("skipping %s %s: parse error %s", symbol, date_, exc)
    return rows


def _fetch_all(
    symbols: list[str],
    start_date: str,
    end_date: str,
    session_refresh_n: int,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV+factor for all symbols, refreshing the baostock session.

    The baostock TCP session expires after ~30 s of inactivity.  We refresh
    every ``session_refresh_n`` symbols to stay within the timeout budget.

    Args:
        symbols: qlib-format symbol list (e.g. ['SH600000', 'SZ000001']).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        session_refresh_n: Refresh login every this many symbols.

    Returns:
        Dict mapping symbol -> DataFrame with columns matching ``_CSV_COLUMNS``.
        Symbols with no data (all days suspended) are omitted.

    Raises:
        ImportError:  baostock not installed.
        RuntimeError: baostock login fails.
    """
    try:
        import baostock as bs  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "baostock is not installed; add it via: uv add baostock"
        ) from exc

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login_result.error_msg}")

    results: dict[str, pd.DataFrame] = {}
    try:
        for i, sym in enumerate(symbols):
            if i > 0 and i % session_refresh_n == 0:
                bs.logout()
                login_result = bs.login()
                if login_result.error_code != "0":
                    raise RuntimeError(
                        f"baostock re-login failed at symbol {i}: "
                        f"{login_result.error_msg}"
                    )
                log.debug(
                    "baostock session refreshed at symbol %d/%d",
                    i + 1,
                    len(symbols),
                )
            rows = _fetch_symbol(bs, sym, start_date, end_date)
            if rows:
                results[sym] = pd.DataFrame(rows)
    finally:
        bs.logout()

    return results


# ---------------------------------------------------------------------------
# CSV serialisation and qlib dump_bin
# ---------------------------------------------------------------------------


def _write_csvs(data: dict[str, pd.DataFrame], dest: Path) -> int:
    """Write per-symbol CSV files in the format expected by qlib dump_bin.

    Each file is named ``{SYMBOL}.csv`` (e.g. ``SH600000.csv``) and contains
    columns: date, open, high, low, close, volume, factor, change -- sorted
    chronologically.

    Args:
        data: Dict of symbol -> DataFrame (from ``_fetch_all``).
        dest: Output directory; created if absent.

    Returns:
        Number of CSV files written.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for symbol, df in data.items():
        out = dest / f"{symbol}.csv"
        df.sort_values("date")[_CSV_COLUMNS].to_csv(out, index=False)
        count += 1
    return count


def _extend_instrument_end_dates(
    inst_dir: Path, traded_syms: set[str], latest: str
) -> None:
    """Extend open membership intervals so the calendar can advance.

    An *open interval* is any row whose end_date equals the file's
    current MAX end_date at call time AND whose symbol traded on
    *latest*.  Only open intervals are rolled forward; closed
    intervals (end < max_end) are never touched.

    The traded_syms filter prevents universe inflation: a stock
    removed on the bundle date (end == max_end) is suspended and
    does not appear in traded_syms, so it is not extended.

    Applies to all instrument files (all.txt + index membership files).
    """
    for inst_file in sorted(inst_dir.glob("*.txt")):
        _extend_open_rows_in_file(inst_file, traded_syms, latest)


def _extend_open_rows_in_file(
    inst_file: Path, traded_syms: set[str], latest: str
) -> None:
    """Extend rows where end == file's max_end to *latest*.

    Only extends rows whose symbol is in *traded_syms*.  This prevents
    extending a stock that was removed on the bundle date (end ==
    max_end but suspended, so not in traded_syms).

    Empty end_date means "still active" in qlib convention and is
    treated as an open interval -- also extended.
    """
    if not inst_file.exists():
        return
    lines = inst_file.read_text(encoding="utf-8").splitlines()
    # Pass 1: find max_end (ignoring empty end dates).
    max_end = ""
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] and parts[2] > max_end:
            max_end = parts[2]
    if max_end and max_end >= latest:
        return  # already current
    # Pass 2: extend open rows (end == max_end or empty) that traded.
    new_lines: list[str] = []
    changed = False
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 3 and (parts[2] == max_end or not parts[2]):
            if parts[0] in traded_syms:
                parts[2] = latest
                changed = True
        new_lines.append("\t".join(parts))
    if changed:
        fd, tmp_path = tempfile.mkstemp(
            dir=inst_file.parent, suffix=".tmp"
        )
        try:
            os.close(fd)
            Path(tmp_path).write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8"
            )
            os.chmod(tmp_path, inst_file.stat().st_mode & 0o7777)
            os.replace(tmp_path, inst_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise


def check_index_membership(
    inst_dir: Path,
    trade_date: str,
    index: str = "csi1000",
    expected: int = 1000,
    tolerance: int = 10,
) -> None:
    """Verify index member count is within tolerance of expected.

    Reads instruments/{index}.txt and counts symbols active on trade_date
    (start <= trade_date, end >= trade_date or empty). Raises ValueError
    if count is outside [expected - tolerance, expected + tolerance].

    Default values (expected=1000, tolerance=10) match
    baseline.yaml universe.data_quality.csi1000_member_count.
    Tight tolerance (1%) because chenditc bundles have exact counts.
    Callers may override if config is available.

    Silently skips when the index file does not exist (unit-test fixtures
    build minimal instrument dirs without all index files).

    Called at the end of _dump_bin_update so every data-write path
    (fetch-today, backfill, pipeline baostock fallback) inherits the
    gate. On violation the caller fails (fetch-today -> rc!=0 ->
    ashare-data-update.sh alerts; pipeline fallback -> gap_fill rc!=0 ->
    existing fallback failure path).
    """
    index_file = inst_dir / f"{index}.txt"
    if not index_file.exists():
        log.debug("membership gate: %s not found, skipping", index_file.name)
        return

    active = 0
    for line in index_file.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        start = parts[1]
        end = parts[2] if len(parts) >= 3 else ""
        if start <= trade_date and (not end or end >= trade_date):
            active += 1

    lo, hi = expected - tolerance, expected + tolerance
    if not (lo <= active <= hi):
        raise ValueError(
            f"{index} membership gate FAILED: {active} active on {trade_date} "
            f"(expected [{lo}, {hi}]). Possible universe inflation."
        )
    log.info("membership gate: %s %d active on %s (OK)", index, active, trade_date)


def _dump_bin_update(csv_dir: Path, provider_uri: Path) -> None:
    """Append daily CSV data to qlib binary feature files.

    Directly writes little-endian float32 values into the binary store,
    appends new dates to the calendar, and extends instrument end-dates.
    Replaces the previous ``qlib.scripts.dump_bin update`` subprocess
    which silently produced no output on the copied qlib.scripts module.

    Args:
        csv_dir: Directory containing per-symbol ``{SYMBOL}.csv`` files.
        provider_uri: qlib data root to update in-place.

    Raises:
        RuntimeError: No CSV files found or no instruments updated.
    """
    feat_dir = provider_uri / "features"
    cal_path = provider_uri / "calendars" / "day.txt"
    inst_dir = provider_uri / "instruments"

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"no CSV files in {csv_dir}")

    fields = [c for c in _CSV_COLUMNS if c != "date"]

    # Discover new dates from one sample CSV (all share the same dates).
    with open(csv_files[0], encoding="utf-8") as f:
        new_dates = sorted({row["date"] for row in csv_mod.DictReader(f)})

    # Append missing dates to calendar.
    existing_dates: set[str] = set()
    if cal_path.exists():
        existing_dates = set(cal_path.read_text(encoding="utf-8").splitlines())
    dates_to_add = [d for d in new_dates if d not in existing_dates]
    if not dates_to_add:
        log.info("dump_bin update: all dates already in calendar, skipping")
        # Gate: extend open rows + check membership even when no new dates.
        latest_date = (
            max(new_dates) if new_dates
            else (max(existing_dates) if existing_dates else None)
        )
        if latest_date:
            _traded = {cf.stem.upper() for cf in csv_files}
            _extend_instrument_end_dates(inst_dir, _traded, latest_date)
            check_index_membership(inst_dir, latest_date)
        return
    with open(cal_path, "a", encoding="utf-8") as f:
        for d in dates_to_add:
            f.write(d + "\n")

    # Append binary data per instrument.
    count = 0
    traded_syms: set[str] = set()
    for csv_file in csv_files:
        sym = csv_file.stem
        traded_syms.add(sym.upper())
        sym_dir = feat_dir / sym
        if not sym_dir.exists():
            continue

        with open(csv_file, encoding="utf-8") as f:
            rows = sorted(csv_mod.DictReader(f), key=lambda r: r["date"])

        for row in rows:
            for field in fields:
                bin_path = sym_dir / f"{field}.day.bin"
                if not bin_path.exists():
                    continue
                with open(bin_path, "ab") as bf:
                    bf.write(struct.pack("<f", float(row[field])))
        count += 1

    if not count:
        raise RuntimeError("no instruments updated (feature dirs missing?)")

    if new_dates:
        _extend_instrument_end_dates(inst_dir, traded_syms, max(new_dates))

    # Gate: always check after data write, even if no new dates
    # (the caller may have written data on a previous partial run).
    latest_date = (
        max(new_dates) if new_dates
        else max(existing_dates) if existing_dates else None
    )
    if latest_date:
        check_index_membership(inst_dir, latest_date)

    log.info(
        "dump_bin update: %d instruments, %d new dates", count, len(dates_to_add)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gap_fill(
    from_date: str,
    to_date: str,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> int:
    """Fill trading dates missing from the qlib store using baostock (qfq).

    Workflow:
      1. Find trading dates in [from_date, to_date] absent from the qlib
         calendar (``calendars/day.txt``).
      2. Fetch OHLCV + adjustfactor for all instruments via baostock.
      3. Write per-symbol CSVs; call ``qlib.scripts.dump_bin update`` to
         append data to the binary feature store.
      4. Verify: re-check missing dates; return 2 if any remain.

    Args:
        from_date: Inclusive start date (YYYY-MM-DD).
        to_date: Inclusive end date (YYYY-MM-DD).
        provider_uri: qlib data root (default: ``~/.qlib/qlib_data/cn_data``).

    Returns:
        0  success  -- all missing dates filled.
        1  no-op    -- no gap detected in [from_date, to_date].
        2  partial  -- some dates still missing after update.

    Raises:
        ImportError:  baostock not installed.
        RuntimeError: no instruments found, baostock login fails, or
                      dump_bin update fails.
    """
    from ashare_lab.config import load_config  # noqa: PLC0415

    cfg = load_config()
    session_refresh_n: int = int(
        cfg.get("fallback", {}).get("max_symbols_per_session", _DEFAULT_SESSION_REFRESH_N)
    )

    missing = find_missing_dates(from_date, to_date, provider_uri)
    if not missing:
        log.info("no missing dates in [%s, %s]; nothing to fill", from_date, to_date)
        return 1

    log.info(
        "gap-fill: %d missing date(s) %s..%s",
        len(missing),
        missing[0],
        missing[-1],
    )

    symbols = _all_instruments(provider_uri)
    if not symbols:
        raise RuntimeError(
            f"no instruments found in {provider_uri}/instruments/; "
            "run bootstrap first"
        )

    log.info(
        "fetching %d symbol(s) for %d date(s) from baostock (refresh every %d)",
        len(symbols),
        len(missing),
        session_refresh_n,
    )

    data = _fetch_all(symbols, missing[0], missing[-1], session_refresh_n)

    if not data:
        log.warning(
            "baostock returned no data for %s..%s", missing[0], missing[-1]
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="ashare_gapfill_") as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        n_written = _write_csvs(data, csv_dir)
        log.info("wrote %d symbol CSV(s)", n_written)
        _dump_bin_update(csv_dir, provider_uri)

    still_missing = find_missing_dates(from_date, to_date, provider_uri)
    if still_missing:
        log.warning(
            "gap-fill partial: %d date(s) still missing: %s",
            len(still_missing),
            still_missing,
        )
        return 2

    log.info("gap-fill complete: %d date(s) filled", len(missing))
    return 0
