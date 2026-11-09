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

from ashare_lab.data.adjust import PrevState, next_factor, scale_prices
from ashare_lab.data.calendar import trading_days_between
from ashare_lab.data.store_state import last_state
from ashare_lab.data.update import DEFAULT_PROVIDER_URI

log = logging.getLogger(__name__)

# Fallback default when pipeline.yaml is unavailable
_DEFAULT_SESSION_REFRESH_N = 50

# baostock adjustflag=3 is the unadjusted price, and that is what this path
# wants: the store keeps front-adjusted values, and the factor that turns one
# into the other is derived from the previous stored row (see _to_store_basis).
# Asking for adjustflag=1 returns 后复权, which is a different series, and asking
# for the field `adjustfactor` is rejected outright (10004012).
_BAOSTOCK_ADJUST_FLAG = "3"
_BAOSTOCK_FIELDS = "date,open,high,low,close,volume,pctChg,preclose"
# An index has no adjustment factor and baostock rejects the field for one.
_INDEX_FIELDS = "date,open,high,low,close,volume,pctChg"
_BAOSTOCK_FREQ = "d"

# Columns written to per-symbol CSVs (consumed by qlib dump_bin update)
_CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "factor", "change"]

# A slot a symbol never traded: qlib reads NaN as no data, which is exactly a
# suspended session.
NAN_BYTES = struct.pack("<f", float("nan"))

# Index instruments only refresh when the full package is replaced, so a
# daily incremental run leaves them behind the stocks.
INDEX_SYMBOLS = ("SH000300", "SH000852", "SH000905", "SH000906", "SZ399300")


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
    *,
    index: bool = False,
) -> list[dict]:
    """Fetch OHLCV + adjustfactor for one symbol from baostock.

    Args:
        bs: pre-imported and logged-in baostock module.
        symbol: qlib-format symbol (e.g. 'SZ000001').
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        index: fetch an index, which carries no adjustment factor.

    Returns:
        List of row dicts with keys: date, open, high, low, close, volume,
        factor, change. Empty list if no data or all days suspended.

    Suspended days (empty or zero close) are skipped silently.
    Rows with unparseable numeric fields are skipped with a DEBUG log.
    """
    query = {
        "fields": _INDEX_FIELDS if index else _BAOSTOCK_FIELDS,
        "start_date": start_date,
        "end_date": end_date,
        "frequency": _BAOSTOCK_FREQ,
    }
    if not index:
        query["adjustflag"] = _BAOSTOCK_ADJUST_FLAG
    rs = bs.query_history_k_data_plus(_qlib_to_bs(symbol), **query)
    rows: list[dict] = []
    while rs.error_code == "0" and rs.next():
        try:
            row = rs.get_row_data()
            if index:
                date_, open_, high_, low_, close_, vol_, pct_ = row
                pre_ = ""
            else:
                date_, open_, high_, low_, close_, vol_, pct_, pre_ = row
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
                    # Restated previous close: the anchor a corporate action is
                    # measured against. None on a first listing day, which _to_store_basis
                    # refuses rather than guessing a factor.
                    "pre_close": float(pre_) if pre_ else None,
                    # pctChg is percentage; qlib 'change' is fractional return
                    "change": float(pct_) / 100.0 if pct_ else 0.0,
                }
            )
            if index:
                # An index has no corporate actions, so its basis is the price
                # itself. Stocks get theirs from _to_store_basis instead.
                rows[-1]["factor"] = 1.0
        except (ValueError, TypeError) as exc:
            log.debug("skipping %s %s: parse error %s", symbol, date_, exc)
    return rows


def _fetch_all(
    symbols: list[str],
    start_date: str,
    end_date: str,
    session_refresh_n: int,
    *,
    index: bool = False,
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
        import baostock as bs
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
            rows = _fetch_symbol(bs, sym, start_date, end_date, index=index)
            if rows:
                results[sym] = pd.DataFrame(rows)
    finally:
        bs.logout()

    return results


# ---------------------------------------------------------------------------
# CSV serialisation and qlib dump_bin
# ---------------------------------------------------------------------------


def _to_store_basis(df: pd.DataFrame, prev: PrevState) -> pd.DataFrame:
    """Restate raw baostock rows on the store's front-adjusted basis.

    The store holds ``close = raw * factor`` and the factor must stay continuous
    with the row before it: ``factor_t = factor_(t-1) * raw_(t-1) / pre_close_t``
    (see ``adjust.next_factor``). baostock gives raw prices and the restated
    previous close, which is exactly the pair that formula needs. A row without
    a previous close is refused -- a first listing day has no anchor, and
    inventing ``factor = 1`` there would write a level break into the series.
    """
    rows: list[dict] = []
    state = prev
    for rec in df.sort_values("date").to_dict("records"):
        pre_close = rec.get("pre_close")
        if pre_close is None:
            raise ValueError(f"{rec.get('date')}: row carries no previous close")
        pre_close = float(pre_close)
        if not (pre_close > 0):
            raise ValueError(f"{rec.get('date')}: previous close is {pre_close}")
        factor = next_factor(state, pre_close)
        prices = scale_prices(
            {
                "open": float(rec["open"]),
                "high": float(rec["high"]),
                "low": float(rec["low"]),
                "close": float(rec["close"]),
            },
            factor,
        )
        rows.append(
            {
                "date": rec["date"],
                "open": prices["open"],
                "high": prices["high"],
                "low": prices["low"],
                "close": prices["close"],
                "volume": float(rec["volume"]),
                "factor": factor,
                # Raw close against the restated previous close: both plain
                # prices, so a split cannot show up here as a return.
                "change": float(rec["close"]) / pre_close - 1.0,
            }
        )
        state = PrevState(close=prices["close"], factor=factor)
    return pd.DataFrame(rows)


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


def _symbol_trails(sym_dir: Path, wanted: set[int], fields: list[str]) -> bool:
    """Whether any maintained field of a symbol has a free slot in ``wanted``.

    A field is waiting for a row when its next free slot is at or before the
    furthest date this batch carries: a slot that falls short is padded with
    NaN up to the row's own date, which is the whole point of tracking the
    calendar index. Only the fields this writer maintains count: adjclose,
    amount and vwap arrive with the full package and are never written here,
    so they sit weeks behind and would make every symbol look stale.

    Args:
        sym_dir: that symbol's feature directory.
        wanted: calendar indices the available rows could fill.
        fields: the field names this writer maintains.

    Returns:
        True when at least one of those fields is waiting for such a row.
    """
    if not wanted:
        return False
    furthest = max(wanted)
    return any(
        (slot := _bin_next_slot(sym_dir / f"{field}.day.bin")) is not None and slot <= furthest
        for field in fields
    )


def _feature_dir(feat_root: Path, symbol: str) -> Path:
    """Feature directory holding one symbol's bin files.

    ``instruments/all.txt`` names symbols in upper case while the feature
    directories are lower case, so every lookup has to normalise or it
    silently misses the files.

    Args:
        feat_root: the ``features`` directory.
        symbol: symbol in either case.

    Returns:
        Path to that symbol's directory, which may not exist.
    """
    return feat_root / symbol.lower()


def _bin_next_slot(bin_path: Path) -> int | None:
    """Calendar index the next point appended to ``bin_path`` would occupy.

    A qlib bin file opens with a float32 holding the calendar index of its
    first point, so the next free slot sits at that index plus the number of
    points stored.

    Args:
        bin_path: one ``<field>.day.bin`` file.

    Returns:
        The index, or None when the file is absent or too short to carry a
        header.
    """
    if not bin_path.exists():
        return None
    raw = bin_path.read_bytes()
    if len(raw) < 4:
        return None
    if (len(raw) - 4) % 4:
        # An interrupted append left a partial float. Rounding the slot down
        # would file the next row over the fragment and misalign the series.
        raise RuntimeError(f"truncated feature file: {bin_path}")
    start = int(struct.unpack("<f", raw[:4])[0])
    return start + (len(raw) - 4) // 4


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
    calendar: list[str] = []
    if cal_path.exists():
        # positions here must line up with the ones qlib built, and
        # qlib skips blank lines
        calendar = [
            ln.strip()
            for ln in cal_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    existing_dates: set[str] = set(calendar)
    dates_to_add = [d for d in new_dates if d not in existing_dates]

    # A symbol can trail the calendar it shares with its peers: an index
    # only refreshes when the full package is unpacked, so the stocks move
    # the calendar forward without it. Those rows still need writing even
    # though the calendar already knows every date.
    cal_pos = {d: i for i, d in enumerate(calendar)}
    wanted = {cal_pos[d] for d in new_dates if d in cal_pos}
    maintained = [c for c in _CSV_COLUMNS if c != "date"]
    trailing = bool(wanted) and any(
        _symbol_trails(_feature_dir(feat_dir, cf.stem), wanted, maintained)
        for cf in csv_files
    )

    if not dates_to_add and not trailing:
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
    # The calendar is appended after the bins, not before. A process that dies
    # between the two would otherwise leave a date whose symbols have no rows,
    # and every reader treats a calendar date as data that exists. If the append
    # fails instead, the bins carry a tail the calendar does not name yet, which
    # qlib ignores; the next successful append restores the alignment.
    calendar = calendar + dates_to_add
    cal_pos = {d: i for i, d in enumerate(calendar)}

    # Append binary data per instrument.
    count = 0
    traded_syms: set[str] = set()
    for csv_file in csv_files:
        sym = csv_file.stem
        sym_dir = _feature_dir(feat_dir, sym)
        if not sym_dir.exists():
            # Not traded, as far as the provider is concerned: nothing was
            # written for it, so its instrument end date must not move either.
            continue
        traded_syms.add(sym.upper())

        with open(csv_file, encoding="utf-8") as f:
            rows = sorted(csv_mod.DictReader(f), key=lambda r: r["date"])

        # A bin carries no dates, only a position. Fields drift apart, so a
        # field that already holds a date must be left alone: appending to it
        # again would shift every later value onto the wrong day.
        next_slot = {
            field: _bin_next_slot(sym_dir / f"{field}.day.bin")
            for field in fields
        }
        wrote = False
        for row in rows:
            idx = cal_pos.get(row["date"])
            if idx is None:
                continue
            for field in fields:
                slot = next_slot.get(field)
                bin_path = sym_dir / f"{field}.day.bin"
                if not bin_path.exists():
                    continue
                if slot is None:
                    # An empty file has no header to read a slot from. Starting
                    # the field at this row's own date is the only truthful
                    # choice: later rows get padded from here.
                    if bin_path.stat().st_size == 0:
                        with open(bin_path, "wb") as bf:
                            bf.write(struct.pack("<f", float(idx)))
                            bf.write(struct.pack("<f", float(row[field])))
                        next_slot[field] = idx + 1
                        wrote = True
                    continue
                if idx < slot:
                    # The field already holds this date.
                    continue
                with open(bin_path, "ab") as bf:
                    if idx > slot:
                        # The symbol was absent for (idx - slot) sessions. A
                        # bin carries positions, not dates: appending here
                        # would file this row under the first missed session
                        # and shift every later value one slot early, which
                        # the factor then hides as a corporate action.
                        bf.write(NAN_BYTES * (idx - slot))
                        slot = idx
                    bf.write(struct.pack("<f", float(row[field])))
                next_slot[field] = slot + 1
                wrote = True
        if wrote:
            count += 1

    if not count:
        raise RuntimeError("no instruments updated (feature dirs missing?)")

    if dates_to_add:
        with open(cal_path, "a", encoding="utf-8") as f:
            for d in dates_to_add:
                f.write(d + "\n")

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
    from ashare_lab.config import load_config

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

    # Every row has to land on the store's own basis before it is written.
    anchored: dict[str, pd.DataFrame] = {}
    unanchored: list[str] = []
    for symbol, frame in data.items():
        state = last_state(provider_uri, symbol.lower())
        if state is None:
            unanchored.append(symbol)
            continue
        anchored[symbol] = _to_store_basis(frame, state[0])
    if unanchored:
        log.warning(
            "gap-fill: %d symbol(s) have no stored row to anchor a factor on, skipped: %s",
            len(unanchored),
            unanchored[:5],
        )
    data = anchored

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


def backfill_missing_dates(
    start: dt.date,
    end: dt.date,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> list[str]:
    """Fill trading days absent from the qlib calendar using tushare.

    A failed day is logged and skipped; days already filled stay filled.
    The calendar is rewritten in date order afterwards, because qlib drops
    dates that were appended out of order.

    Returns the dates that were filled.
    """
    from ashare_lab.data.fetcher import fetch_today_data

    missing = find_missing_dates(start.isoformat(), end.isoformat(), provider_uri)
    filled: list[str] = []
    for day in missing:
        try:
            data = fetch_today_data(day, provider_uri=provider_uri)
        except Exception as exc:
            log.warning("backfill %s: fetch failed: %s", day, exc)
            continue
        if not data:
            log.warning("backfill %s: no rows", day)
            continue
        with tempfile.TemporaryDirectory(prefix="ashare_backfill_") as tmp:
            csv_dir = Path(tmp) / "csv"
            _write_csvs(data, csv_dir)
            _dump_bin_update(csv_dir, provider_uri)
        filled.append(day)

    cal_path = provider_uri / "calendars" / "day.txt"
    if cal_path.exists():
        dates = sorted({ln.strip() for ln in cal_path.read_text(encoding="utf-8").splitlines() if ln.strip()})
        fd, tmp_path = tempfile.mkstemp(dir=cal_path.parent, suffix=".tmp")
        try:
            os.close(fd)
            Path(tmp_path).write_text("\n".join(dates) + "\n", encoding="utf-8")
            os.replace(tmp_path, cal_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    return filled


def _instrument_end_dates(provider_uri: Path = DEFAULT_PROVIDER_URI) -> dict[str, str]:
    """Collect ``symbol -> end_date`` from ``instruments/all.txt``.

    Args:
        provider_uri: qlib data root.

    Returns:
        Mapping of symbol to its last stored date, empty when the file is
        absent.
    """
    path = Path(provider_uri) / "instruments" / "all.txt"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[parts[0]] = parts[2]
    return out


def backfill_stale_indexes(
    trade_date: str,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
) -> list[str]:
    """Fill index rows that lag behind ``trade_date``.

    The daily incremental fetch only covers stocks, so every index keeps the
    tail it had when the full package was last unpacked. Readers of a stale
    index get no error, just an older last date, which silently turns a
    date-keyed lookup into a miss.

    Args:
        trade_date: YYYY-MM-DD, the date the stocks were just written up to.
        provider_uri: qlib data root.

    Returns:
        Symbols that were written, empty when every index is current.
    """
    root = Path(provider_uri)
    end_dates = _instrument_end_dates(root)
    target = str(trade_date)
    stale = {
        sym: end_dates[sym]
        for sym in INDEX_SYMBOLS
        if sym in end_dates and end_dates[sym] < target
    }
    if not stale:
        return []

    earliest = min(stale.values())
    start = (dt.date.fromisoformat(earliest) + dt.timedelta(days=1)).isoformat()
    log.info(
        "index backfill: %d stale, fetching %s..%s", len(stale), start, target
    )
    # Measured: one index over a two-week gap takes under a second, and the
    # whole set about two, inside the ~30s baostock idle timeout. One login
    # covers the batch.
    symbols = sorted(stale)
    data = _fetch_all(symbols, start, target, len(symbols), index=True)
    if not data:
        log.warning("index backfill: no rows returned for %s", sorted(stale))
        return []

    with tempfile.TemporaryDirectory(prefix="ashare_index_") as tmp:
        csv_dir = Path(tmp)
        _write_csvs(data, csv_dir)
        _dump_bin_update(csv_dir, root)
    filled = sorted(data)
    log.info("index backfill: wrote %s", filled)
    return filled
