"""Baostock-based gap-fill for qlib cn_data when chenditc release is stale.

Entry point: ``gap_fill(from_date, to_date)`` -- identifies trading dates
absent from the qlib calendar, fetches OHLCV + adjustfactor (qfq) from
baostock, and writes the data into the binary feature store via
``qlib.scripts.dump_bin update``.

The baostock TCP session expires after ~30 s; the session is refreshed
every ``fallback.max_symbols_per_session`` symbols (from pipeline.yaml).
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
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


def _dump_bin_update(csv_dir: Path, provider_uri: Path) -> None:
    """Invoke qlib's dump_bin update to append CSV data to binary feature files.

    Runs ``python -m qlib.scripts.dump_bin update`` as a subprocess so that
    qlib's internal cached provider state does not interfere with the current
    process.

    Args:
        csv_dir: Directory containing per-symbol ``{SYMBOL}.csv`` files.
        provider_uri: qlib data root to update in-place.

    Raises:
        RuntimeError: dump_bin exits non-zero.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qlib.scripts.dump_bin",
            "update",
            "--csv_path",
            str(csv_dir),
            "--qlib_dir",
            str(provider_uri),
            "--freq",
            "day",
            "--include_fields",
            ",".join(c for c in _CSV_COLUMNS if c != "date"),
            "--date_col_name",
            "date",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dump_bin update failed (rc={result.returncode})\n"
            f"stderr[-2000:]: {result.stderr[-2000:]}\n"
            f"stdout[-2000:]: {result.stdout[-2000:]}"
        )
    log.info("dump_bin update succeeded")


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
