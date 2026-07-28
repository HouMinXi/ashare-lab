"""Daily pipeline orchestrator for the A-share paper trading engine.

Sequences all post-close steps: data update, adjustfactor audit,
CSI1000 exit detection, settlement, risk checks, signal generation,
IPO processing, NAV recording, and backup.

The engine never loads models or imports torch -- it reads the daily
prediction file (predictions/{trade_date}.parquet) produced upstream
by research/predict.py (Option 3 decouple).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import sys
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ashare_lab.config import PREDICTIONS_DIR, PROJECT_ROOT, load_config
from ashare_lab.data.calendar import (
    latest_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)
from ashare_lab.paper.adjust import check_and_apply_adjustfactor
from ashare_lab.paper.engine import (
    get_limit_threshold,
    round_lots,
    settle_day,
)
from ashare_lab.paper.hedge import (
    _fetch_hedge_prices,
    _load_hedge_config,
    compute_hedge_state,
    generate_hedge_orders,
    load_hedge_state,
    save_hedge_state,
    update_peak_nav,
)
from ashare_lab.paper.ipo import (
    check_ipo_subscription,
    detect_board_type,
    determine_ipo_sell_date,
)
from ashare_lab.paper.ledger import (
    bump_carry_days,
    bump_suspension_carry_days,
    cleanup_old_backups,
    compute_nav,
    create_golden_backup,
    delete_expired_cooldowns,
    force_reset_day,
    get_connection,
    get_cooldowns,
    get_latest_cash,
    get_latest_positions,
    get_positions_for_date,
    hot_backup_with_integrity,
    init_schema,
    insert_order,
    is_day_settled,
    log_settle_change,
    record_nav,
    record_run,
    open_pipeline_run,
    close_pipeline_run,
    set_cooldown,
    snapshot_positions,
    update_order,
)
from ashare_lab.paper.risk import (
    manage_trailing_cooldown,
    run_all_risk_checks,
)
from ashare_lab.paper.signal import (
    filter_candidates,
    generate_signals,
    topk_dropout_orders,
)

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BS_CACHE_DIR = PROJECT_ROOT / "data" / "baostock_cache"


# -- baostock CSV cache helpers (cache-first, live-API fallback) --

def _bs_code_to_universe(bs_code: str, symbols: set[str]) -> str | None:
    """Map baostock 'sh.600006' to a matching universe symbol."""
    num = bs_code.split(".")[-1] if "." in bs_code else bs_code[-6:]
    for sym in symbols:
        if sym.endswith(num):
            return sym
    return None


def _load_st_cache(
    trade_date: str, symbols: set[str],
) -> set[str] | None:
    """Load ST names from cache CSV.

    Tries exact date match first; falls back to the latest available
    date (ST status is stable across dates, and an approximate answer
    beats hanging on a cross-Pacific baostock query).
    """
    path = _BS_CACHE_DIR / "st_status.csv"
    if not path.exists():
        return None
    rows_by_date: dict[str, list] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_date.setdefault(row["date"], []).append(row)
    use_date = trade_date if trade_date in rows_by_date else None
    if use_date is None and rows_by_date:
        use_date = max(rows_by_date.keys())
    if use_date is None:
        return None
    st: set[str] = set()
    for row in rows_by_date[use_date]:
        if row["is_st"] == "1":
            sym = _bs_code_to_universe(row["code"], symbols)
            if sym:
                st.add(sym)
    return st


def _load_stock_names_cache(symbols: set[str]) -> dict[str, str] | None:
    """Load stock names from cache CSV.

    Prefers tushare-based cache (symbol,name columns) written by
    fetcher.refresh_stock_names_cache. Falls back to legacy baostock
    cache (code,code_name columns) for backward compatibility.
    """
    # Tushare cache: symbol column already in qlib format (e.g. sz000001)
    ts_path = PROJECT_ROOT / "data" / "stock_names_cache.csv"
    if ts_path.exists():
        result: dict[str, str] = {}
        with open(ts_path, newline="") as f:
            for row in csv.DictReader(f):
                # F3 fix: handle None from missing CSV cells
                raw = row.get("symbol") or ""
                sym = raw.upper()
                if sym in symbols:
                    result[sym] = row.get("name", "")
        if result:
            return result

    # Baostock fallback: code column uses baostock format (sh.600006)
    bs_path = _BS_CACHE_DIR / "stock_names.csv"
    if not bs_path.exists():
        return None
    # Integrity check: file too small is likely corrupt
    if bs_path.stat().st_size < 100:
        logger.warning("stock_names.csv too small (%d bytes), likely corrupt", bs_path.stat().st_size)
        return None
    result = {}
    with open(bs_path, newline="") as f:
        for row in csv.DictReader(f):
            sym = _bs_code_to_universe(row["code"], symbols)
            if sym:
                result[sym] = row["code_name"]
    if not result:
        logger.warning("stock_names.csv parsed to empty dict, likely corrupt")
        return None
    return result


def _load_benchmark_cache(trade_date: str) -> dict[str, float] | None:
    """Load benchmark closes from cache CSV.

    Benchmark prices are date-sensitive (absolute levels differ across
    years), so only exact date match is valid.  Returns None on miss;
    the pipeline gracefully handles missing benchmarks.
    """
    path = _BS_CACHE_DIR / "benchmark.csv"
    if not path.exists():
        return None
    result: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["date"] == trade_date:
                try:
                    result[row["index"]] = float(row["close"])
                except (ValueError, TypeError):
                    pass
    return result if result else None


def _load_regime_cache(trade_date: str) -> list[float] | None:
    """Load CSI1000 regime closes from cache CSV.

    Regime data is date-sensitive (price levels differ across years),
    so only exact date match is valid.  Returns None on miss, which
    makes check_market_regime return False (insufficient data) rather
    than halting buying on stale prices from a different year.
    """
    path = _BS_CACHE_DIR / "csi1000_regime.csv"
    if not path.exists():
        return None
    closes: list[float] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["query_date"] == trade_date:
                try:
                    closes.append(float(row["close"]))
                except (ValueError, TypeError):
                    pass
    if not closes:
        return None
    return closes[-11:] if len(closes) >= 11 else closes


def _load_industry_cache(
    trade_date: str, symbols: set[str],
) -> dict[str, str] | None:
    """Load industry map from cache CSV.

    Falls back to latest available date (industry classification is
    stable across years).
    """
    path = _BS_CACHE_DIR / "industry.csv"
    if not path.exists():
        return None
    rows_by_date: dict[str, list] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_date.setdefault(row["date"], []).append(row)
    use_date = trade_date if trade_date in rows_by_date else None
    if use_date is None and rows_by_date:
        use_date = max(rows_by_date.keys())
    if use_date is None:
        return None
    result: dict[str, str] = {}
    for row in rows_by_date[use_date]:
        sym = _bs_code_to_universe(row["code"], symbols)
        if sym and row["industry"]:
            result[sym] = row["industry"]
    return result


def _save_industry_cache(
    trade_date: str, industry_map: dict[str, str],
) -> None:
    """Append new industry data to cache CSV (idempotent by date)."""
    path = _BS_CACHE_DIR / "industry.csv"
    _BS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing to avoid duplicates
    existing: set[str] = set()
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["date"] == trade_date:
                    existing.add(row["code"])

    # Map universe symbols back to baostock codes
    def _to_bs_code(sym: str) -> str:
        s = sym.upper()
        if s.startswith(("SH", "SZ")):
            return s[:2].lower() + "." + s[2:]
        num = s[-6:]
        pfx = "sh" if s.startswith(("SH", "6")) else "sz"
        return pfx + "." + num

    new_rows = []
    for sym, ind in industry_map.items():
        code = _to_bs_code(sym)
        if code not in existing:
            new_rows.append({"date": trade_date, "code": code, "industry": ind})

    if not new_rows:
        return

    try:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "code", "industry"])
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        logger.info("industry cache: saved %d new entries for %s", len(new_rows), trade_date)
    except OSError as exc:
        logger.warning("industry cache save failed: %s", exc)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _fetch_ipo_calendar(listing_date: str) -> list[dict]:
    """Fetch IPO calendar from eastmoney for the given date.

    Returns a list of dicts with keys: symbol, listing_date,
    issue_price, board, ceiling_lots, win_rate.

    On HTTP or parse failure: log warning, return [] (IPO step
    becomes a no-op).
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.warning("requests not installed, IPO calendar unavailable")
        return []

    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get"
        "?sortColumns=LISTING_DATE"
        "&sortTypes=-1"
        "&pageSize=50"
        "&pageNumber=1"
        "&reportName=RPTA_APP_IPOAPPLY"
        "&columns=ALL"
        f"&filter=(LISTING_DATE='{listing_date}')"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning(
            "IPO calendar fetch failed for %s", listing_date, exc_info=True
        )
        return []

    result_data = data.get("result", {})
    if result_data is None:
        return []
    rows = result_data.get("data")
    if not rows:
        return []

    calendar: list[dict] = []
    for row in rows:
        try:
            symbol_raw = row.get("SECURITY_CODE", "")
            listing_dt = row.get("LISTING_DATE", "")
            if listing_dt and "T" in listing_dt:
                listing_dt = listing_dt.split("T")[0]
            issue_price = float(row.get("ISSUE_PRICE", 0) or 0)
            ceiling_lots = int(row.get("APPLY_UPPER_LIMIT", 0) or 0)
            win_rate_raw = row.get("LOTTERY_RATE")
            win_rate = float(win_rate_raw) if win_rate_raw else 0.0

            board = detect_board_type(symbol_raw)
            calendar.append({
                "symbol": symbol_raw,
                "listing_date": listing_dt,
                "issue_price": issue_price,
                "board": board,
                "ceiling_lots": ceiling_lots,
                "win_rate": win_rate,
            })
        except (TypeError, ValueError):
            continue

    return calendar


def _fetch_benchmark_closes(trade_date: str) -> dict[str, float]:
    """Fetch benchmark closes: cache first, baostock subprocess fallback.

    Returns {"csi300": float, "csi1000": float}.
    On failure returns zeros with warning log.
    """
    cached = _load_benchmark_cache(trade_date)
    if cached is not None:
        logger.debug("benchmark from cache for %s", trade_date)
        return {"csi300": cached.get("csi300", 0.0),
                "csi1000": cached.get("csi1000", 0.0)}

    # [R9] subprocess timeout -- baostock hangs cross-Pacific
    _bench_script = """
import baostock as bs, json, sys, os, socket
socket.setdefaulttimeout(30)
td = sys.argv[1]
_orig_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
bs.login()
sys.stdout.close()
sys.stdout = _orig_stdout
result = {"csi300": 0.0, "csi1000": 0.0}
for code, key in [("sh.000300","csi300"),("sh.000852","csi1000")]:
    rs = bs.query_history_k_data_plus(code, "close", start_date=td, end_date=td, frequency="d")
    while rs.error_code == "0" and rs.next():
        try: result[key] = float(rs.get_row_data()[0])
        except: pass
_null = open(os.devnull, 'w')
sys.stdout = _null
bs.logout()
_null.close()
sys.stdout = _orig_stdout
print(json.dumps(result))
"""
    try:
        _r = subprocess.run([sys.executable, "-c", _bench_script, trade_date],
                            capture_output=True, text=True, timeout=60,
                            cwd=str(PROJECT_ROOT))
        if _r.returncode == 0 and _r.stdout.strip():
            result = json.loads(_r.stdout.strip())
            logger.info("benchmark via subprocess: %s", result)
            return result
        logger.warning("benchmark subprocess failed: rc=%d", _r.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("benchmark subprocess timed out after 30s")
    except Exception as e:
        logger.warning("benchmark subprocess error: %s", e)
    return {"csi300": 0.0, "csi1000": 0.0}


def _resolve_prediction_file(date_str: str) -> Path | None:
    """Return the prediction parquet path for the given date, or None."""
    path = PREDICTIONS_DIR / f"{date_str}.parquet"
    if path.exists():
        return path
    logger.warning(
        "Prediction file not found: %s "
        "(run the producer or sync predictions/)",
        path,
    )
    return None


# ------------------------------------------------------------------
# Main pipeline -- context object + step helpers
# ------------------------------------------------------------------


@dataclass
class DailyRunContext:
    """Mutable state threaded through run_daily step helpers.

    Each field is set by the step that produces it and consumed by later
    steps.  Using a single context object avoids a long argument list
    while keeping all state explicit and traceable.
    """
    # Inputs resolved in step 0/1
    trade_date: str
    force: bool
    steps: set[str] | None
    pred_path: "Path | None"
    start_time: float
    predictions_date_str: str
    pipeline_run_id: int | None = None

    # Step 1 outputs
    config: dict = field(default_factory=dict)
    paper_cfg: dict = field(default_factory=dict)
    risk_cfg: dict = field(default_factory=dict)
    db_path: "Path | None" = None
    conn: "object | None" = None

    # Step 4 outputs
    current_positions: dict = field(default_factory=dict)
    cash: float = 0.0
    cooldown_state: dict = field(default_factory=dict)
    is_soft_reduced: bool = False
    risk_state: "object | None" = None
    lockdown_enter_date: str | None = None

    # Step 5 outputs
    universe_symbols: list = field(default_factory=list)
    ipo_won: list = field(default_factory=list)
    ipo_listing_syms: set = field(default_factory=set)
    fetch_symbols: set = field(default_factory=set)
    st_names: set = field(default_factory=set)
    prices: dict = field(default_factory=dict)
    benchmarks: dict = field(default_factory=dict)
    csi1000_closes_11d: list = field(default_factory=list)
    industry_map: dict = field(default_factory=dict)
    market_data: dict = field(default_factory=dict)

    # Step 8 output
    settle_result: "object | None" = None

    # Step 9 outputs
    total_nav: float = 0.0
    risk_result: "object | None" = None

    # Step 9b outputs
    hedge_state: "object | None" = None
    hedge_symbols: set = field(default_factory=set)

    # Step 10 output
    signals_raw: dict = field(default_factory=dict)

    # Step 11 output
    ipo_cash_changed: bool = False

    # Sentiment veto status (set by step 10 on failure)
    sentiment_veto_skipped: bool = False


def _step1_init(ctx: DailyRunContext) -> None:
    """Load config, open DB, init qlib."""
    ctx.config = load_config()
    ctx.paper_cfg = ctx.config["paper"]
    ctx.risk_cfg = ctx.paper_cfg["risk"]
    ctx.db_path = PROJECT_ROOT / ctx.paper_cfg["db_path"]
    ctx.conn = get_connection(ctx.db_path)
    init_schema(ctx.conn)
    try:
        import qlib  # noqa: PLC0415
        from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415
        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
    except Exception:
        pass  # qlib not installed or already initialized


def _step2_idempotency(ctx: DailyRunContext) -> int:
    """Return 0 to continue, 0 to skip (already settled), 2 on force conflict."""
    if not ctx.force and is_day_settled(ctx.conn, ctx.trade_date):
        logger.info("Already settled: %s", ctx.trade_date)
        return 0  # sentinel: caller returns 0
    if not ctx.force:
        row = ctx.conn.execute(
            "SELECT status FROM runs WHERE trade_date = ?", (ctx.trade_date,)
        ).fetchone()
        if row is not None and row["status"] == "ledger_error":
            # A prior run's settle identity check failed and left this
            # date's cash and positions out of sync.  Auto-retrying would
            # open the next day on a book that pairs an errored day's cash
            # with a stale day's positions -- a human needs to inspect and
            # repair the ledger first, so only an explicit --force may
            # proceed past it.
            logger.error(
                "%s is marked ledger_error (settle identity check failed); "
                "refusing automatic retry. Investigate the drift, repair "
                "the ledger, then re-run with --force.",
                ctx.trade_date,
            )
            _close_pipeline_run(
                ctx, "error",
                f"{ctx.trade_date} marked ledger_error, refusing auto-retry",
            )
            ctx.conn.commit()
            return 2
    if ctx.force:
        latest_settled = ctx.conn.execute(
            "SELECT MAX(trade_date) FROM runs WHERE status='settled'"
        ).fetchone()[0]
        if latest_settled is not None and ctx.trade_date < latest_settled:
            logger.error(
                "force-reset of %s leaves settled days > it stale; "
                "run `paper backfill %s %s --force` to replay the "
                "coupled range",
                ctx.trade_date, ctx.trade_date, latest_settled,
            )
            return 2
        force_reset_day(ctx.conn, ctx.trade_date)
    return -1  # sentinel: continue


def _try_baostock_fallback(ctx: DailyRunContext, chenditc_exc: Exception) -> int:
    """Try baostock gap_fill when chenditc download fails.

    Returns 0 on success, 1 on stale (no gap), 2 on failure.
    """
    try:
        from ashare_lab.data.fallback import gap_fill  # noqa: PLC0415
        from ashare_lab.data.calendar import previous_trading_day  # noqa: PLC0415
        start_date = previous_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()
        rc = gap_fill(start_date, ctx.trade_date)
        if rc == 0:
            logger.info("baostock fallback succeeded for %s", ctx.trade_date)
            return 0
        if rc == 1:
            logger.info("baostock fallback: no gap detected")
            return 0
        logger.error("baostock fallback partial (rc=%d)", rc)
        record_run(ctx.conn, ctx.trade_date, "error")
        _close_pipeline_run(ctx, "error", f"chenditc+baostock both partial: {chenditc_exc}")
        ctx.conn.commit()
        return 2
    except Exception as fb_exc:
        logger.error("baostock fallback also failed: %s", fb_exc, exc_info=True)
        record_run(ctx.conn, ctx.trade_date, "error")
        _close_pipeline_run(ctx, "error", f"chenditc: {chenditc_exc}; baostock: {fb_exc}")
        ctx.conn.commit()
        return 2


def _step3_data_update(ctx: DailyRunContext) -> int:
    """Refresh qlib data feed; return 1 if stale, 2 on error, -1 to continue."""
    if ctx.steps == {"signal"}:
        return -1
    td = dt.date.fromisoformat(ctx.trade_date)
    recent_cutoff = dt.date.today() - dt.timedelta(days=10)
    if td < recent_cutoff:
        return -1

    # If fetch-today already populated qlib data for this trade_date,
    # skip the chenditc refresh entirely.
    from ashare_lab.data.update import _read_calendar_last_date, DEFAULT_PROVIDER_URI, daily_refresh  # noqa: PLC0415
    cal_last = _read_calendar_last_date(DEFAULT_PROVIDER_URI)
    if cal_last and cal_last >= ctx.trade_date:
        logger.info("qlib data already covers %s (cal=%s), skipping chenditc", ctx.trade_date, cal_last)
        return -1

    try:
        stale = daily_refresh()
    except Exception as exc:
        stale = _try_baostock_fallback(ctx, exc)
        if stale == 2:
            return 2
    if stale == 1:
        record_run(ctx.conn, ctx.trade_date, "skipped_stale")
        ctx.conn.commit()
        logger.warning("Data stale for %s, skipping", ctx.trade_date)
        return 1
    return -1


def _gate_data_completeness(ctx: DailyRunContext) -> None:
    """Halt if qlib calendar missing expected trading days since last run."""
    if ctx.steps == {"signal"}:
        return
    last_run = ctx.conn.execute(
        "SELECT MAX(trade_date) FROM pipeline_runs WHERE status='success'"
    ).fetchone()[0]
    if last_run is None or last_run >= ctx.trade_date:
        return  # first run or backfill
    from ashare_lab.data.update import _read_calendar_last_date, DEFAULT_PROVIDER_URI  # noqa: PLC0415
    from ashare_lab.data.calendar import trading_days_between  # noqa: PLC0415
    cal_last = _read_calendar_last_date(DEFAULT_PROVIDER_URI)
    expected = trading_days_between(
        dt.date.fromisoformat(last_run) + dt.timedelta(days=1),
        dt.date.fromisoformat(ctx.trade_date),
    )
    missing = [d.isoformat() for d in expected if cal_last is None or d.isoformat() > cal_last]
    if missing:
        raise RuntimeError(
            f"Data completeness gate: qlib missing {len(missing)} trading days: {missing[:5]}. "
            f"Run fetch-today or update qlib data before pipeline."
        )


def _step4_load_state(ctx: DailyRunContext) -> None:
    """Load positions, cash, cooldowns, risk state from DB."""
    ctx.current_positions = get_latest_positions(ctx.conn)
    ctx.cash = get_latest_cash(ctx.conn, ctx.paper_cfg["initial_cash"])
    ctx.cooldown_state = get_cooldowns(ctx.conn)
    ctx.cooldown_state = manage_trailing_cooldown(ctx.cooldown_state, ctx.trade_date)
    delete_expired_cooldowns(ctx.conn, ctx.trade_date)
    ctx.cooldown_state = {
        s: cd for s, cd in ctx.cooldown_state.items()
        if cd["cooldown_until"] >= ctx.trade_date
    }

    # Load persistent risk state (state machine)
    from ashare_lab.paper.risk_state import (  # noqa: PLC0415
        RiskState,
        ensure_shadow_log_table,
        load_risk_state,
        migrate_from_soft_reduced,
    )
    ensure_shadow_log_table(ctx.conn)
    ctx.risk_state, ctx.lockdown_enter_date = load_risk_state(ctx.conn)
    if ctx.risk_state == RiskState.NORMAL and ctx.lockdown_enter_date is None:
        # Check if migration is needed (risk_state key absent)
        row = ctx.conn.execute(
            "SELECT value FROM paper_state WHERE key = 'risk_state'"
        ).fetchone()
        if row is None:
            # First run with state machine -- migrate from is_soft_reduced
            # Compute current dd for initial state derivation
            nav_rows = ctx.conn.execute(
                "SELECT trade_date, total_nav FROM nav ORDER BY trade_date ASC"
            ).fetchall()
            nav_history = [dict(r) for r in nav_rows]
            if nav_history:
                peak = max(r["total_nav"] for r in nav_history)
                current = nav_history[-1]["total_nav"]
                dd = (peak - current) / peak if peak > 0 else 0.0
            else:
                dd = 0.0
            ctx.risk_state = migrate_from_soft_reduced(
                ctx.conn, dd, ctx.trade_date,
            )
            if ctx.risk_state == RiskState.LIQUIDATED:
                ctx.lockdown_enter_date = ctx.trade_date
            ctx.conn.commit()

    # Backward compat: load is_soft_reduced (derived from risk_state)
    row = ctx.conn.execute(
        "SELECT value FROM paper_state WHERE key='is_soft_reduced'"
    ).fetchone()
    if row is not None:
        ctx.is_soft_reduced = row["value"] == "true"
    else:
        ctx.is_soft_reduced = False


def _step5_fetch_prices_and_universe(ctx: DailyRunContext) -> int:
    """Fetch universe, prices, benchmarks, industry map.  Return 2 on qlib import failure."""
    try:
        from qlib.data import D  # noqa: PLC0415
    except ImportError as exc:
        logger.error("qlib not available, cannot run pipeline")
        record_run(ctx.conn, ctx.trade_date, "error")
        _close_pipeline_run(ctx, "error", str(exc))
        ctx.conn.commit()
        return 2

    # 5a. universe
    ctx.universe_symbols = D.list_instruments(
        D.instruments("csi1000"),
        start_time=ctx.trade_date,
        end_time=ctx.trade_date,
        as_list=True,
    )

    # 5b. IPO calendar
    next_td = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()
    ipo_calendar = _fetch_ipo_calendar(next_td)
    ctx.ipo_won = [
        row for row in ipo_calendar
        if check_ipo_subscription(row["ceiling_lots"], row["win_rate"])[0]
    ]
    ctx.ipo_listing_syms = {
        row["symbol"] for row in ctx.ipo_won
        if row["listing_date"] == ctx.trade_date
    }

    order_syms = {
        r[0] for r in ctx.conn.execute(
            "SELECT DISTINCT symbol FROM orders WHERE status IN ('pending','carry')"
        )
    }
    ctx.fetch_symbols = (
        set(ctx.universe_symbols)
        | set(ctx.current_positions.keys())
        | order_syms
        | ctx.ipo_listing_syms
    )

    # settle-only fast path: skip regime, industry, market_data (signal-path only).
    # settle needs prices + benchmarks + ST names; risk checks degrade gracefully
    # on empty csi1000_closes / industry_map / market_data.
    settle_only = ctx.steps == {"settle"}

    # 5c. ST names -- always runs (local CSV cache, no network).
    # ST status determines limit threshold (5% vs 10%) used by settle_day.
    from ashare_lab.data.validator import _ST_PATTERN  # noqa: PLC0415
    st_cached = _load_st_cache(ctx.trade_date, ctx.fetch_symbols)
    if st_cached is not None:
        ctx.st_names = st_cached
        logger.debug("ST names from cache for %s", ctx.trade_date)
    else:
        ctx.st_names = set()
        names = _load_stock_names_cache(ctx.fetch_symbols)
        if names:
            for sym, name in names.items():
                if _ST_PATTERN.match(name or ""):
                    ctx.st_names.add(sym)
            logger.info("ST detection from stock_names_cache: %d ST stocks", len(ctx.st_names))
        else:
            logger.warning("no stock name cache available, ST detection skipped")

    # Prices from qlib (subprocess with 90s timeout -- qlib client mode
    # hangs if server is down; SIGALRM gets swallowed by joblib/qlib internals)
    ctx.prices = {}
    if ctx.fetch_symbols:
        # subprocess and json imported at module level
        _fetch_script = """
import qlib, json, sys
from pathlib import Path
from ashare_lab.data.update import DEFAULT_PROVIDER_URI
qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
from qlib.data import D
syms = json.loads(sys.argv[1])
td = sys.argv[2]
raw = D.features(instruments=syms, fields=["$close","$change","$volume","$factor"],
                 start_time=td, end_time=td)
if raw is not None and not raw.empty:
    out = {}
    for idx, row in raw.iterrows():
        inst = idx[0] if isinstance(idx, tuple) else str(idx)
        out[str(inst)] = {"close": float(row.get("$close",0)), "change": float(row.get("$change",0)),
                          "volume": float(row.get("$volume",0)), "factor": float(row.get("$factor",1.0))}
    print(json.dumps(out))
else:
    print("{}")
"""
        try:
            _result = subprocess.run(
                [sys.executable, "-c", _fetch_script,
                 json.dumps(list(ctx.fetch_symbols)), ctx.trade_date],
                capture_output=True, text=True, timeout=120,
                cwd=str(PROJECT_ROOT),
            )
            if _result.returncode == 0 and _result.stdout.strip():
                _price_data = json.loads(_result.stdout.strip())
                for sym, pdata in _price_data.items():
                    # match symbol to fetch_symbols (suffix match)
                    matched = sym
                    for fs in ctx.fetch_symbols:
                        if fs.endswith(sym[-6:]) if len(sym) > 6 else fs == sym:
                            matched = fs
                            break
                    # qlib $close is normalized (IPO day=1.0); divide by $factor
                    # to recover actual CNY price for NAV and order sizing.
                    _raw_factor = pdata.get("factor")
                    if _raw_factor is None:
                        # Factor missing: price may be normalized but we
                        # cannot recover real CNY. Log warning, do NOT set
                        # adjusted=True so the sanity gate can flag it.
                        logger.warning(
                            "Factor missing for %s, using raw qlib close=%.4f",
                            matched, pdata["close"],
                        )
                        _factor = 1.0
                        _adjusted = False
                    else:
                        _factor = max(_raw_factor, 1e-8)
                        _adjusted = True
                    ctx.prices[matched] = {
                        "close": pdata["close"] / _factor,
                        "change": pdata["change"],
                        "volume": pdata["volume"],
                        "factor": _raw_factor if _raw_factor is not None else 1.0,
                        "adjusted": _adjusted,
                        "threshold": get_limit_threshold(matched, ctx.st_names),
                    }
                logger.info("fetched %d prices via subprocess", len(ctx.prices))
            else:
                logger.warning("D.features subprocess failed: rc=%d stderr=%s",
                               _result.returncode, _result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("D.features subprocess timed out after 90s, prices unavailable")
        except Exception as e:
            logger.warning("D.features subprocess error: %s", e)

    # 5d. benchmarks
    ctx.benchmarks = _fetch_benchmark_closes(ctx.trade_date)

    # 5e-5g settle fast path: use local cache only (no network fallback).
    # Risk checks (step 9) persist cooldowns and soft-reduce state, so regime
    # and industry data must be populated when cached. market_data is signal-
    # path only (filter_candidates) and safe to skip.
    if settle_only:
        # 5e. regime: cache-only, no qlib D.features fallback
        regime_cached = _load_regime_cache(ctx.trade_date)
        ctx.csi1000_closes_11d = regime_cached if regime_cached is not None else []
        if not ctx.csi1000_closes_11d:
            logger.warning("settle: no cached regime data for %s, market regime check disabled", ctx.trade_date)

        # 5f. industry: cache-only, no baostock fallback
        ind_cached = _load_industry_cache(ctx.trade_date, ctx.fetch_symbols)
        ctx.industry_map = ind_cached if ind_cached is not None else {}
        if not ctx.industry_map:
            logger.warning("settle: no cached industry data for %s, concentration check disabled", ctx.trade_date)

        # 5g. market_data: signal-path only, not needed for settle
        ctx.market_data = {}
        return -1

    # 5e. CSI1000 11-day closes for regime check (qlib, no baostock)
    regime_cached = _load_regime_cache(ctx.trade_date)
    if regime_cached is not None:
        ctx.csi1000_closes_11d = regime_cached
        logger.debug("CSI1000 regime from cache for %s", ctx.trade_date)
    else:
        ctx.csi1000_closes_11d = []
        start_60d = (dt.date.fromisoformat(ctx.trade_date) - dt.timedelta(days=60)).isoformat()
        try:
            # [R8] subprocess timeout -- same pattern as prices fetch
            _regime_script = """
import qlib, json, sys
from ashare_lab.data.update import DEFAULT_PROVIDER_URI
qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
from qlib.data import D
start_d, end_d = sys.argv[1], sys.argv[2]
raw = D.features(instruments=["SH000852"], fields=["$close"],
                 start_time=start_d, end_time=end_d)
if raw is not None and not raw.empty:
    closes = [float(v) for v in raw["$close"].dropna().values]
    print(json.dumps(closes[-11:] if len(closes) >= 11 else closes))
else:
    print("[]")
"""
            _regime_result = subprocess.run(
                [sys.executable, "-c", _regime_script, start_60d, ctx.trade_date],
                capture_output=True, text=True, timeout=60,
                cwd=str(PROJECT_ROOT),
            )
            if _regime_result.returncode == 0 and _regime_result.stdout.strip():
                ctx.csi1000_closes_11d = json.loads(_regime_result.stdout.strip())
                logger.info("regime data fetched via subprocess: %d closes", len(ctx.csi1000_closes_11d))
        except subprocess.TimeoutExpired:
            logger.warning("regime subprocess timed out, market regime check disabled")
        except Exception as exc:
            logger.warning("CSI1000 regime query failed: %s", exc)

    # 5f. industry map (cache + incremental subprocess)
    ind_cached = _load_industry_cache(ctx.trade_date, ctx.fetch_symbols)
    if ind_cached is not None:
        ctx.industry_map = dict(ind_cached)
        logger.debug("industry_map from cache for %s (%d entries)",
                     ctx.trade_date, len(ctx.industry_map))
    else:
        ctx.industry_map = {}

    # Query only symbols missing from cache
    _missing = [s for s in ctx.fetch_symbols if s not in ctx.industry_map]
    if _missing:
        _ind_script = """
import baostock as bs, json, sys, os, socket
socket.setdefaulttimeout(30)
from concurrent.futures import ThreadPoolExecutor, as_completed
td = sys.argv[1]
syms = [s for s in json.loads(sys.argv[2]) if not s.upper().startswith("BJ")]
_orig = sys.stdout
sys.stdout = open(os.devnull, 'w')
bs.login()
sys.stdout.close()
sys.stdout = _orig

def _query_one(sym):
    code = sym.lower()
    if len(code) == 6:
        pfx = "sh" if code.startswith("6") else "sz"
        code = pfx + "." + code
    elif not code.startswith(("sh.", "sz.")):
        pfx = "sh" if sym.startswith(("SH", "6")) else "sz"
        code = pfx + "." + sym[-6:]
    rs = bs.query_stock_industry(code=code, date=td)
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        if len(row) > 3 and row[3]:
            return (sym, row[3])
    return None

result = {}
with ThreadPoolExecutor(max_workers=10) as pool:
    futs = {pool.submit(_query_one, s): s for s in syms}
    for fut in as_completed(futs):
        r = fut.result()
        if r:
            result[r[0]] = r[1]

_null = open(os.devnull, 'w')
sys.stdout = _null
bs.logout()
_null.close()
sys.stdout = _orig
print(json.dumps(result))
"""
        try:
            _r = subprocess.run(
                [sys.executable, "-c", _ind_script,
                 ctx.trade_date, json.dumps(_missing)],
                capture_output=True, text=True, timeout=300,
                cwd=str(PROJECT_ROOT))
            if _r.returncode == 0 and _r.stdout.strip():
                _new = json.loads(_r.stdout.strip())
                ctx.industry_map.update(_new)
                _save_industry_cache(ctx.trade_date, _new)
                logger.info("industry subprocess: %d new entries (%d total)",
                            len(_new), len(ctx.industry_map))
            else:
                logger.warning("industry subprocess failed: rc=%d stderr=%s",
                               _r.returncode, _r.stderr[:200] if _r.stderr else "")
        except subprocess.TimeoutExpired:
            logger.warning(
                "industry subprocess timed out after 180s (%d missing, %d cached)",
                len(_missing), len(ctx.industry_map))
        except Exception as e:
            logger.warning("industry subprocess error: %s", e)
    else:
        logger.debug("industry_map fully cached (%d entries)", len(ctx.industry_map))

    # 5g. market_data for filter_candidates
    # Batch D.features: 2 calls instead of 2*N per-symbol calls.
    ctx.market_data = {}
    priced_symbols = [s for s in ctx.fetch_symbols if s in ctx.prices]
    if not priced_symbols:
        # ponytail: equivalent to old code's empty for-loop producing empty dict
        return -1

    # 5g-0. build suffix -> symbol reverse-lookup (O(1) matching, no nested loop)
    import pandas as pd  # noqa: PLC0415
    suffix_to_sym = {}
    for sym in priced_symbols:
        suffix = sym[-6:] if len(sym) > 6 else sym
        if suffix in suffix_to_sym:
            logger.warning("suffix collision: %s vs %s (suffix=%s)", sym, suffix_to_sym[suffix], suffix)
        suffix_to_sym[suffix] = sym

    def _match_inst(inst_key):
        """Map qlib instrument ID to our symbol via 6-digit suffix."""
        s = str(inst_key)
        suffix = s[-6:] if len(s) > 6 else s
        matched = suffix_to_sym.get(suffix)
        if matched is None:
            logger.debug("no symbol match for qlib instrument %s", s)
        return matched

    # 5g-1. listing_days: batch query first bar date for all symbols
    listing_days_map = {}
    try:
        first_bar_df = D.features(
            instruments=priced_symbols, fields=["$close"],
            start_time="2005-01-01", end_time=ctx.trade_date,
        )
        if first_bar_df is not None and not first_bar_df.empty:
            td = dt.date.fromisoformat(ctx.trade_date)
            for inst in first_bar_df.index.get_level_values(0).unique():
                sym_df = first_bar_df.loc[inst]
                # pandas .loc on MultiIndex returns Series for single-row results
                if isinstance(sym_df, pd.Series):
                    sym_df = sym_df.to_frame().T
                if sym_df.empty:
                    continue
                first_date = sym_df.index[0]
                first_bar_date = first_date.date() if hasattr(first_date, "date") else first_date
                matched = _match_inst(inst)
                if matched:
                    listing_days_map[matched] = len(
                        trading_days_between(first_bar_date, td)
                    )
            del first_bar_df  # free ~40MB before turnover query
    except Exception:
        logger.debug("batch listing_days query failed", exc_info=True)

    # 5g-2. turnover: batch query 40-day volume/close/factor for all symbols
    turnover_map = {}
    start_40d = (dt.date.fromisoformat(ctx.trade_date) - dt.timedelta(days=40)).isoformat()
    try:
        turnover_batch = D.features(
            instruments=priced_symbols,
            fields=["$volume", "$close", "$factor"],
            start_time=start_40d, end_time=ctx.trade_date,
        )
        if turnover_batch is not None and not turnover_batch.empty:
            for inst in turnover_batch.index.get_level_values(0).unique():
                sym_df = turnover_batch.loc[inst]
                if isinstance(sym_df, pd.Series):
                    sym_df = sym_df.to_frame().T
                if sym_df.empty:
                    continue
                # qlib $close is forward-adjusted; divide by $factor to recover actual CNY price
                factor = sym_df["$factor"].replace(0, 1)
                unadj_close = sym_df["$close"] / factor
                turnover_vals = (sym_df["$volume"] * unadj_close).tail(20)
                if len(turnover_vals) > 0:
                    matched = _match_inst(inst)
                    if matched:
                        turnover_map[matched] = float(turnover_vals.mean())
            del turnover_batch
    except Exception:
        logger.debug("batch turnover query failed", exc_info=True)

    # 5g-3. assemble market_data (both maps keyed by our symbol, independent lookups)
    for sym in priced_symbols:
        ctx.market_data[sym] = {
            "close": ctx.prices[sym]["close"],
            "listing_days": listing_days_map.get(sym, 999),
            "avg_turnover_20d": turnover_map.get(sym, 0.0),
        }

    return -1  # continue


def _is_normalized_price(close: float, factor: float | None, adjusted: bool = False) -> bool:
    """Detect qlib normalized $close (IPO day=1.0) vs real CNY price.

    If `adjusted` is True, the price was already divided by factor — not normalized.
    Otherwise: factor < 0.1 indicates high-IPO-price stock where normalized close
    could be >= 1.0 (e.g., IPO at 500 yuan, now at 600 -> normalized close=1.2).
    Missing factor: flag if close < 5 (normalized closes are typically < 2.0,
    while most A-share stocks trade above 5 CNY).
    """
    if adjusted:
        return False
    if factor is not None:
        return factor < 0.1
    return close < 5


def _gate_price_sanity(ctx: DailyRunContext) -> None:
    """Halt if prices look like normalized (qlib $close) or wrong-year data."""
    if ctx.steps == {"signal"}:
        return
    import math  # noqa: PLC0415
    suspicious = 0
    for sym, pdata in ctx.prices.items():
        close = pdata.get("close")
        if close is None or (isinstance(close, float) and (math.isnan(close) or close <= 0)):
            continue
        if _is_normalized_price(close, pdata.get("factor"), pdata.get("adjusted", False)):
            suspicious += 1
            if suspicious <= 3:
                f = pdata.get("factor")
                logger.error("Price anomaly: %s close=%.6f factor=%s (likely normalized $close)",
                             sym, close, f"{f:.6f}" if f is not None else "MISSING")
    if suspicious > len(ctx.prices) * 0.1:
        raise RuntimeError(
            f"Price sanity gate: {suspicious}/{len(ctx.prices)} suspect prices. "
            f"Likely qlib $close not divided by $factor. Halting."
        )
    if suspicious > 0:
        logger.warning("Price sanity: %d prices < 1.0 (below threshold, continuing)", suspicious)


def _step6_adjustfactor(ctx: DailyRunContext) -> None:
    """Audit adjustfactor changes and update positions in-memory."""
    previous_factors = {s: pos["factor"] for s, pos in ctx.current_positions.items()}
    current_factors = {
        s: ctx.prices[s]["factor"]
        for s in ctx.current_positions if s in ctx.prices
    }
    ctx.current_positions, adj_records = check_and_apply_adjustfactor(
        ctx.current_positions, previous_factors, current_factors
    )
    if adj_records:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"adjust_{ctx.trade_date}.jsonl"
        with log_path.open("a") as f:
            for rec in adj_records:
                f.write(json.dumps(rec) + "\n")
        logger.warning(
            "Adjustfactor changes detected for %d symbols on %s",
            len(adj_records), ctx.trade_date,
        )


def _step7_csi1000_exits(ctx: DailyRunContext) -> None:
    """Insert sell orders for positions that have left the CSI1000 universe.

    Requires consecutive absence to handle qlib one-day lag: a stock
    must be absent for 2+ consecutive days before exit sell is created.

    Persistence strategy: instead of an in-memory counter (resets each
    run), use the orders table as the counter.  On first absence, insert
    a pending exit sell for next_td.  On the next run, if the stock is
    still absent and already has a pending/carry sell, that confirms
    2+ consecutive absence -- let the order proceed.  If the stock
    returns to the universe, cancel the pending exit sell.
    """
    universe_set = set(ctx.universe_symbols)
    next_td_date = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()
    for sym, pos in ctx.current_positions.items():
        if sym in universe_set:
            # Stock is in universe -- cancel any pending exit sell
            # created by this mechanism (identified by cancel_reason).
            pending_exit = ctx.conn.execute(
                "SELECT id FROM orders WHERE symbol=? AND side='sell' "
                "AND status IN ('pending','carry') "
                "AND cancel_reason = 'csi1000_exit'",
                (sym,),
            ).fetchone()
            if pending_exit is not None:
                update_order(ctx.conn, pending_exit["id"], status="cancelled",
                             cancel_reason="csi1000_returned")
                logger.info("CSI1000 exit: %s returned to universe, exit sell cancelled", sym)
            continue
        # Stock not in universe -- check for existing pending exit sell
        existing_exit = ctx.conn.execute(
            "SELECT id FROM orders WHERE symbol=? AND side='sell' "
            "AND status IN ('pending','carry') "
            "AND cancel_reason = 'csi1000_exit'",
            (sym,),
        ).fetchone()
        if existing_exit is not None:
            # Already has a pending exit sell from yesterday's absence.
            # This confirms 2+ consecutive absence. Let the order
            # proceed to settlement.
            logger.info("CSI1000 exit: %s confirmed absent (pending sell exists), proceeding", sym)
            continue
        # First absence -- insert pending exit sell for next_td
        qty = pos["qty"]
        insert_order(ctx.conn, next_td_date, sym, "sell", qty, None, "pending", 0, ctx.trade_date,
                     cancel_reason="csi1000_exit")
        logger.info("CSI1000 exit: %s absent from universe, exit sell for %s created", sym, next_td_date)


def _prefetch_hedge_prices_for_settle(ctx: DailyRunContext) -> None:
    """Fetch hedge leg prices before settle so yesterday's hedge orders can fill.

    step9b_hedge_sleeve creates orders for next_td.  On the next day,
    step8_settle needs those ETF prices in ctx.prices.  Without this
    prefetch, step5 only fetches universe stocks and hedge legs get
    volume=0.0 -> _is_suspended -> orders carried indefinitely.
    """
    hedge_cfg_dict = ctx.config.get("paper", {}).get("hedge", {})
    if not hedge_cfg_dict.get("enabled", False):
        return
    from ashare_lab.paper.hedge import _load_hedge_config
    hedge_cfg = _load_hedge_config(hedge_cfg_dict)
    leg_symbols = [leg.symbol for leg in hedge_cfg.legs]
    # Only fetch for legs not already in ctx.prices (step5 may have some)
    missing = [s for s in leg_symbols if s not in ctx.prices]
    if missing:
        _fetch_hedge_prices(ctx, missing)
        logger.info("hedge pre-fetch: got prices for %d legs before settle", len(missing))


def _step8_settle(ctx: DailyRunContext) -> int | None:
    """Settle pending orders; skip for signal-only runs.

    Returns 2 if the post-settle ledger identity check fails (caller
    must halt), None otherwise.
    """
    if ctx.steps == {"signal"}:
        return None
    active_cooldowns = {
        s for s, cd in ctx.cooldown_state.items()
        if cd["cooldown_until"] >= ctx.trade_date
    }
    prev_date = previous_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()
    topk_rows = ctx.conn.execute(
        "SELECT symbol FROM signals WHERE trade_date = ?", (prev_date,)
    ).fetchall()
    if not topk_rows:
        topk_rows = ctx.conn.execute(
            "SELECT symbol FROM signals WHERE trade_date = "
            "(SELECT MAX(trade_date) FROM signals WHERE trade_date <= ?)",
            (prev_date,),
        ).fetchall()
    topk_symbols = {r["symbol"] for r in topk_rows}

    pending_orders = [
        dict(r) for r in ctx.conn.execute(
            "SELECT id, symbol, side, target_qty, carry_day, reset_count, "
            "suspension_carry_day FROM orders "
            "WHERE status IN ('pending','carry') AND trade_date <= ?",
            (ctx.trade_date,),
        ).fetchall()
    ]
    all_orders = list(pending_orders)

    # carry cooldown cancel (two-pass)
    orders_to_cancel = [
        o for o in all_orders
        if o["side"] == "buy" and o["carry_day"] > 0 and o["symbol"] in active_cooldowns
    ]
    cancel_ids: set[int] = set()
    for o in orders_to_cancel:
        log_settle_change(ctx.conn, ctx.trade_date, o["id"])
        update_order(ctx.conn, o["id"], status="cancelled")
        cancel_ids.add(o["id"])
    all_orders = [o for o in all_orders if o["id"] not in cancel_ids]

    ctx.settle_result = settle_day(
        ctx.conn, ctx.trade_date, all_orders, ctx.prices,
        ctx.current_positions, ctx.cash, topk_symbols, ctx.benchmarks, ctx.paper_cfg,
    )
    # Ledger identity check: the positions settle_day just committed to the
    # DB, plus the settled cash, must reconstruct the post-trade NAV that
    # settle_day itself computed.  This runs before the four lines below
    # (carry-day bumps, positions/cash reload) on purpose: a violation
    # means a human has to repair the ledger, and the abort path must not
    # leave orders with bumped carry counts or leave ctx holding cash/
    # positions read back from a day that never really finished settling.
    # mv_db and drift depend only on ctx.settle_result (set by settle_day
    # above) and the positions table settle_day already wrote -- neither
    # needs bump_carry_days/bump_suspension_carry_days/current_positions/
    # ctx.cash, so moving the gate ahead of them changes nothing it reads.
    mv_db = ctx.conn.execute(
        "SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE trade_date = ?",
        (ctx.trade_date,),
    ).fetchone()[0]
    drift = mv_db + ctx.settle_result.cash - ctx.settle_result.post_trade_nav
    # 1.0 CNY is an absolute threshold, not relative to NAV: float64 carries
    # about 15-17 significant digits, so even a million-CNY book leaves a
    # ~1e-7 CNY rounding floor -- seven orders of magnitude below this gate.
    # A relative threshold would only make the gate laxer on a large book,
    # which is exactly backwards for a correctness check.
    if abs(drift) >= 1.0:
        logger.error(
            "Settle identity check failed on %s: positions market_value=%.2f "
            "+ cash=%.2f != post_trade_nav=%.2f (drift=%.2f)",
            ctx.trade_date, mv_db, ctx.settle_result.cash,
            ctx.settle_result.post_trade_nav, drift,
        )
        # runs.status is free text (no CHECK constraint) so it can carry a
        # status finer-grained than pipeline_runs' fixed enum: 'ledger_error'
        # marks this date as needing manual repair, not just a transient
        # failure -- see the idempotency check in _step2_idempotency.
        record_run(ctx.conn, ctx.trade_date, "ledger_error")
        _close_pipeline_run(
            ctx, "error",
            f"settle identity check failed: drift={drift:.2f}",
        )
        ctx.conn.commit()
        return 2

    bump_carry_days(ctx.conn, [o["order_id"] for o in ctx.settle_result.carries_to_bump])
    bump_suspension_carry_days(ctx.conn, [o["order_id"] for o in ctx.settle_result.carries_suspended])
    # Exact-date read, not get_latest_positions: record_run hasn't marked
    # today 'settled' yet at this point in the pipeline, so the "latest
    # settled day" anchor would still resolve to yesterday.
    ctx.current_positions = get_positions_for_date(ctx.conn, ctx.trade_date)
    ctx.cash = ctx.settle_result.cash


def _step9_risk_checks(ctx: DailyRunContext) -> None:
    """Compute NAV and run all risk checks; persist cooldown entries."""
    ctx.total_nav = compute_nav(ctx.current_positions, ctx.prices, ctx.cash)
    nav_rows = ctx.conn.execute(
        "SELECT trade_date, total_nav FROM nav ORDER BY trade_date ASC"
    ).fetchall()
    nav_history = [dict(r) for r in nav_rows]
    yesterday_nav = float(nav_history[-2]["total_nav"]) if len(nav_history) >= 2 else ctx.total_nav

    is_shadow = ctx.risk_cfg.get("state_machine", "shadow") == "shadow"
    ctx.risk_result = run_all_risk_checks(
        nav_history, yesterday_nav, ctx.current_positions, ctx.prices,
        ctx.csi1000_closes_11d, ctx.industry_map, ctx.cooldown_state, ctx.cash,
        ctx.is_soft_reduced, ctx.risk_cfg, ctx.trade_date,
        pred_date=ctx.predictions_date_str,
        risk_state=ctx.risk_state,
        lockdown_enter_date=ctx.lockdown_enter_date,
        is_shadow=is_shadow,
    )
    ctx.is_soft_reduced = ctx.risk_result.topk_override is not None
    ctx.conn.execute(
        "INSERT OR REPLACE INTO paper_state VALUES ('is_soft_reduced', ?)",
        ("true" if ctx.is_soft_reduced else "false",),
    )
    for symbol, entry in ctx.risk_result.cooldown_entries.items():
        set_cooldown(ctx.conn, symbol, entry["cooldown_until"], entry["holding_high"])

    # Persist shadow log if present
    if ctx.risk_result.shadow_log is not None:
        from ashare_lab.paper.risk_state import log_shadow_transition  # noqa: PLC0415
        log_shadow_transition(
            ctx.conn, ctx.trade_date,
            ctx.risk_result.shadow_log.get("old_flags", {}),
            ctx.risk_result.shadow_log.get("shadow_state", "normal"),
            ctx.risk_result.shadow_log.get("would_do", {}),
        )

    # Persist new risk state (F1: save every run, both modes)
    if ctx.risk_result.shadow_log is not None and "new_state" in ctx.risk_result.shadow_log:
        from ashare_lab.paper.risk_state import RiskState, save_risk_state  # noqa: PLC0415
        new_state = RiskState(ctx.risk_result.shadow_log["new_state"])
        new_lockdown = ctx.risk_result.shadow_log.get("new_lockdown_enter_date")
        ctx.risk_state = new_state
        ctx.lockdown_enter_date = new_lockdown
        save_risk_state(
            ctx.conn, new_state, new_lockdown,
            derive_soft_reduced=not is_shadow,  # F2: only enforce mode
        )


def _step9b_hedge_sleeve(ctx: DailyRunContext) -> None:
    """Adjust equity/hedge allocation based on drawdown."""
    hedge_cfg_dict = ctx.config.get("paper", {}).get("hedge", {})
    if not hedge_cfg_dict.get("enabled", False):
        return

    hedge_cfg = _load_hedge_config(hedge_cfg_dict)
    leg_symbols = [leg.symbol for leg in hedge_cfg.legs]
    _fetch_hedge_prices(ctx, leg_symbols)

    prev_td = previous_trading_day(dt.date.fromisoformat(ctx.trade_date))
    prev = load_hedge_state(ctx.conn, prev_td.isoformat()) if prev_td else None

    peak = update_peak_nav(ctx.total_nav, prev.peak_nav if prev else 0.0)
    dd = (peak - ctx.total_nav) / peak if peak > 0 else 0.0
    prev_active = prev.active if prev else False

    if prev_active:
        days = prev.days_in_hedge + 1
    else:
        days = 1 if dd >= hedge_cfg.activate_dd else 0

    prev_recovery = prev.days_in_recovery if prev else 0
    state = compute_hedge_state(
        ctx.total_nav, peak, days, hedge_cfg,
        prev_active=prev_active, prev_days_in_recovery=prev_recovery,
    )
    ctx.hedge_state = state
    ctx.hedge_symbols = set(leg_symbols)

    save_hedge_state(ctx.conn, ctx.trade_date, state)

    next_td_str = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()

  
    ctx.conn.execute(
        "DELETE FROM orders WHERE trade_date=? AND source='hedge' AND status='pending'",
        (next_td_str,),
    )

    orders = generate_hedge_orders(
        ctx.current_positions,
        state,
        ctx.total_nav,
        ctx.prices,
        hedge_cfg,
        buying_halted=ctx.risk_result.buying_halted if ctx.risk_result else False,
    )
    for order in orders:
        insert_order(
            ctx.conn,
            next_td_str,
            order.symbol,
            order.side,
            order.target_qty,
            None,
            "pending",
            0,
            ctx.trade_date,
            source=order.source,
        )


def _step9c_nav_hedge_split(ctx: DailyRunContext) -> None:
    """Overwrite NAV row with hedge/equity split (runs after step 9b)."""
    if not ctx.hedge_symbols:
        return
    hedge_val = sum(
        ctx.current_positions.get(s, {}).get("market_value") or 0
        for s in ctx.hedge_symbols
    )
    equity_val = ctx.total_nav - ctx.cash - hedge_val
    record_nav(
        ctx.conn,
        ctx.trade_date,
        ctx.cash,
        ctx.total_nav - ctx.cash,
        ctx.total_nav,
        ctx.total_nav,
        ctx.total_nav,
        ctx.benchmarks.get("csi300"),
        ctx.benchmarks.get("csi1000"),
        hedge_value=hedge_val,
        equity_value=equity_val,
    )


def timing_multiplier(trade_date: str, model: str = "none") -> float:
    """Return the timing multiplier m for the given trade date and model.

    m scales top-k target weights at order generation (cash allocation),
    not signal scores.  Supported values: {0, 0.5, 1.0}.

    Supported models:
      "none" = always 1.0 (Book A behavior)
      "n1"   = CSI1000 20d MA trend filter
      "n2"   = 20d realized volatility percentile
      "n3"   = drawdown state map

    Args:
        trade_date: ISO date string (YYYY-MM-DD).
        model: Model identifier.

    Returns:
        Float multiplier in {0, 0.5, 1.0}.
    """
    if model == "none":
        return 1.0

    if model not in ("n1", "n2", "n3"):
        logger.warning("Unknown timing model %r, defaulting to m=1.0", model)
        return 1.0

    try:
        closes = _fetch_csi1000_closes_for_timing(trade_date)
        if closes is None or closes.empty:
            logger.warning("[timing] No CSI1000 data for %s, m=1.0", trade_date)
            return 1.0

        from ashare_lab.research.timing_nulls import (  # noqa: PLC0415
            ma_trend_multiplier,
            vol_percentile_multiplier,
            drawdown_state_multiplier,
        )

        fn = {
            "n1": ma_trend_multiplier,
            "n2": vol_percentile_multiplier,
            "n3": drawdown_state_multiplier,
        }[model]
        m_series = fn(closes)

        if trade_date not in m_series.index:
            logger.warning(
                "[timing] trade_date %s not in m series (model=%s), m=1.0",
                trade_date, model,
            )
            return 1.0

        m = float(m_series.loc[trade_date])
        logger.info("[timing] %s on %s: m=%.1f", model, trade_date, m)
        return m

    except Exception:
        logger.warning("[timing] %s failed for %s, m=1.0", model, trade_date, exc_info=True)
        return 1.0


def _fetch_csi1000_closes_for_timing(trade_date: str):
    """Fetch CSI1000 close series for timing null models.

    Returns Series indexed by trading date with enough history for all
    nulls (N2/N3 need 252 days).  Uses qlib subprocess, same source
    as regime data fetch.
    """
    import pandas as pd  # noqa: PLC0415

    # Need 300 trading days (~420 calendar days) for N2/N3 warmup
    start_date = (dt.date.fromisoformat(trade_date) - dt.timedelta(days=500)).isoformat()

    _script = """
import qlib, json, sys
from ashare_lab.data.update import DEFAULT_PROVIDER_URI
qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
from qlib.data import D
start_d, end_d = sys.argv[1], sys.argv[2]
raw = D.features(instruments=["SH000852"], fields=["$close"],
                 start_time=start_d, end_time=end_d)
if raw is not None and not raw.empty:
    result = {}
    for idx, val in zip(raw.index, raw["$close"].values):
        date_str = str(idx[1].date()) if hasattr(idx[1], "date") else str(idx[1])
        result[date_str] = float(val)
    print(json.dumps(result))
else:
    print("{}")
"""
    try:
        _r = subprocess.run(
            [sys.executable, "-c", _script, start_date, trade_date],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if _r.returncode == 0 and _r.stdout.strip():
            data = json.loads(_r.stdout.strip())
            if data:
                closes = pd.Series(data, dtype=float)
                closes.index = pd.to_datetime(closes.index)
                closes = closes.sort_index()
                logger.info("[timing] fetched %d CSI1000 closes for %s", len(closes), trade_date)
                return closes
    except subprocess.TimeoutExpired:
        logger.warning("[timing] CSI1000 fetch timed out")
    except Exception as exc:
        logger.warning("[timing] CSI1000 fetch error: %s", exc)

    return None


def _step10_signal_generation(ctx: DailyRunContext) -> int:
    """Generate signals, filter candidates, insert buy/sell orders.  Return 2 on error."""
    if ctx.steps == {"settle"}:
        return -1

    # Resolve prediction file
    if ctx.pred_path is None:
        ctx.pred_path = PREDICTIONS_DIR / f"{ctx.trade_date}.parquet"
        if not ctx.pred_path.exists():
            logger.error(
                "Prediction file not found: %s (run the producer or sync predictions/)",
                ctx.pred_path,
            )
            record_run(ctx.conn, ctx.trade_date, "error")
            _close_pipeline_run(ctx, "error", "prediction file not found")
            ctx.conn.commit()
            return 2

    m = re.search(r"(\d{4}-\d{2}-\d{2})", ctx.pred_path.name)
    if m:
        ctx.predictions_date_str = m.group(1)

    try:
        ctx.signals_raw = generate_signals(
            ctx.trade_date, conn=ctx.conn, pred_path=ctx.pred_path,
            topk=ctx.paper_cfg["topk"],
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Signal generation failed for %s: %s", ctx.trade_date, e)
        record_run(ctx.conn, ctx.trade_date, "error")
        _close_pipeline_run(ctx, "error", str(e))
        ctx.conn.commit()
        return 2

    # Provenance log + model staleness + IC check
    # Derive meta.json path from pred_path (which may have been
    # overridden by the caller) so the companion .meta.json stays
    # paired with the actual prediction file used.
    meta_path = ctx.pred_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with meta_path.open() as f:
                meta = json.load(f)
            logger.info(
                "Prediction provenance: model=%s window=%s",
                meta.get("model"), meta.get("window_id"),
            )
            model_age = meta.get("model_age_days")
            if model_age is not None and model_age > 7:
                logger.warning(
                    "Pipeline: model %s is %.0f days old (threshold: 7). "
                    "Consider retraining.",
                    meta.get("model"), model_age,
                )
            # IC=nan detection: the TRA model's internal IC metric is
            # NaN when predicting beyond the training window (no labels
            # to compare).  The blend uses a fixed 60/40 ratio so
            # predictions are unaffected, but we log and flag it for
            # the report.
            ic_val = meta.get("ic")
            import math as _math  # noqa: PLC0415
            if ic_val is None or (isinstance(ic_val, float) and _math.isnan(ic_val)):
                logger.warning(
                    "Pipeline: model IC is unavailable (nan) for %s. "
                    "Predictions use fixed 60/40 blend, unaffected.",
                    ctx.trade_date,
                )
                # Store flag for report annotation
                ctx.config["_ic_nan"] = True
        except Exception:
            logger.warning("Failed to read meta.json: %s", meta_path)
    else:
        logger.debug(
            "No meta.json found for prediction file: %s "
            "(staleness info unavailable)",
            ctx.pred_path,
        )

    # Track IC history for event-driven retraining (outside the
    # meta.json try-block so failures log their own message, not
    # the misleading "Failed to read meta.json").
    append_ic_history(ctx.pred_path, ctx.trade_date)

    candidate_syms = [s for s in ctx.signals_raw if s in ctx.market_data]
    filtered_syms = filter_candidates(
        candidate_syms, ctx.market_data,
        ctx.paper_cfg["listing_min_days"],
        ctx.paper_cfg["liquidity_min_turnover"],
        ctx.config.get("universe", {}).get("exclude_close_above_cny", 300.0),
    )
    filtered_signals = {s: ctx.signals_raw[s] for s in filtered_syms}

    effective_topk = ctx.risk_result.topk_override or ctx.paper_cfg["topk"]

    ipo_held = {
        s for s in ctx.current_positions
        if ctx.market_data.get(s, {}).get("listing_days", 999)
        < ctx.paper_cfg.get("listing_min_days", 60)
    }
    held_set = set(ctx.current_positions.keys()) - ipo_held - ctx.hedge_symbols
    topk_for_dropout = max(0, effective_topk - len(ipo_held))
    sell_syms, buy_syms = topk_dropout_orders(
        filtered_signals, held_set, topk_for_dropout, ctx.paper_cfg.get("n_drop", 1),
    )

    buy_syms = [
        s for s in buy_syms
        if ctx.industry_map.get(s) not in ctx.risk_result.blocked_industries
        and s not in ctx.risk_result.blocked_rebuys
    ]

    if ctx.config.get("paper", {}).get("sentiment", {}).get("enabled", False):
        try:
            from ashare_lab.paper.sentiment import run_sentiment_veto
            veto_result = run_sentiment_veto(
                buy_syms, ctx.trade_date, ctx.industry_map, ctx.conn, ctx.config,
            )
            buy_syms = [s for s in buy_syms if s not in veto_result.vetoed_stocks]
            if veto_result.global_halted:
                ctx.risk_result = replace(ctx.risk_result, buying_halted=True)
            logger.info(
                "sentiment veto: %d stocks vetoed, %d industries, global=%s",
                len(veto_result.vetoed_stocks), len(veto_result.vetoed_industries),
                "halted" if veto_result.global_halted else "ok",
            )
        except Exception:
            ctx.sentiment_veto_skipped = True
            logger.warning("Sentiment veto failed, continuing without veto", exc_info=True)

    # T+1 sell guard
    sell_syms = [
        s for s in sell_syms
        if ctx.current_positions.get(s, {}).get("buy_date") != ctx.trade_date
    ]

    if ctx.risk_result.buying_halted:
        buy_syms = []

    if effective_topk <= 0:
        target_value = 0.0
        buy_syms = []
    else:
        if ctx.hedge_state and ctx.hedge_state.active:
            equity_nav = ctx.total_nav * ctx.hedge_state.equity_target_pct
        else:
            equity_nav = ctx.total_nav
        target_value = (equity_nav * ctx.config["cost_model"]["risk_degree"]) / effective_topk

    forced_sells = ctx.risk_result.forced_sells
    sell_set = set(sell_syms) | set(forced_sells.keys())
    desired_sell_qty: dict[str, int] = {}
    for s in sell_set:
        if s in sell_syms:
            desired_sell_qty[s] = ctx.current_positions.get(s, {}).get("qty", 0)
        else:
            desired_sell_qty[s] = forced_sells[s]

    next_td_str = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()
    for s in sell_set:
        already = ctx.conn.execute(
            "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
            "WHERE symbol=? AND side='sell' AND status IN ('pending','carry')",
            (s,),
        ).fetchone()[0]
        to_insert = desired_sell_qty[s] - already
        if to_insert > 0:
            insert_order(ctx.conn, next_td_str, s, "sell", to_insert, None, "pending", 0, ctx.trade_date)

    for s in buy_syms:
        close_price = ctx.prices.get(s, {}).get("close")
        if (
            not close_price
            or close_price <= 0
            or (isinstance(close_price, float) and close_price != close_price)
        ):
            logger.warning("Skip buy %s: invalid close price %s", s, close_price)
            continue
        target_qty = round_lots(target_value / close_price, "buy")
        if target_qty <= 0:
            continue
        insert_order(ctx.conn, next_td_str, s, "buy", target_qty, None, "pending", 0, ctx.trade_date)

    return -1


def _step11_ipo_processing(ctx: DailyRunContext) -> None:
    """Process IPO listings and re-evaluate held IPO positions."""
    ctx.ipo_cash_changed = False
    next_td_str = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()

    for ipo_row in ctx.ipo_won:
        if ipo_row["listing_date"] != ctx.trade_date:
            continue
        symbol = ipo_row["symbol"]
        won, shares = check_ipo_subscription(ipo_row["ceiling_lots"], ipo_row["win_rate"])
        if not won:
            continue
        new_pos = {
            "qty": shares,
            "avg_cost": ipo_row["issue_price"],
            "market_value": shares * ctx.prices.get(symbol, {}).get("close", ipo_row["issue_price"]),
            "buy_date": ipo_row["listing_date"],
            "holding_high": ctx.prices.get(symbol, {}).get("close", ipo_row["issue_price"]),
            "factor": ctx.prices.get(symbol, {}).get("factor", 1.0),
        }
        ctx.current_positions[symbol] = new_pos
        ctx.cash -= shares * ipo_row["issue_price"]
        ctx.ipo_cash_changed = True
        snapshot_positions(ctx.conn, ctx.trade_date, ctx.current_positions)
        if symbol not in ctx.prices:
            continue
        sell_date = determine_ipo_sell_date(
            symbol, ipo_row["listing_date"],
            [(ipo_row["listing_date"], ctx.prices[symbol]["change"])],
        )
        if sell_date is None:
            continue
        order_date = sell_date if sell_date > ctx.trade_date else next_td_str
        already = ctx.conn.execute(
            "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
            "WHERE symbol=? AND side='sell' AND status IN ('pending','carry')",
            (symbol,),
        ).fetchone()[0]
        to_insert = shares - already
        if to_insert > 0:
            insert_order(ctx.conn, order_date, symbol, "sell", to_insert, None, "pending", 0, ctx.trade_date)

    # Held IPO re-evaluation
    for sym, pos in list(ctx.current_positions.items()):
        if ctx.market_data.get(sym, {}).get("listing_days", 999) >= ctx.paper_cfg.get("listing_min_days", 60):
            continue
        if pos.get("buy_date", "") == ctx.trade_date:
            continue
        try:
            from qlib.data import D as D_ipo  # noqa: PLC0415
            chg_df = D_ipo.features(
                instruments=[sym], fields=["$change"],
                start_time=pos["buy_date"], end_time=ctx.trade_date,
            )
            if chg_df is not None and not chg_df.empty:
                daily_changes = [
                    (str(d.date()) if hasattr(d, "date") else str(d), float(c))
                    for d, c in zip(
                        chg_df.index.get_level_values(0)
                        if isinstance(chg_df.index, type(chg_df.index))
                        else chg_df.index,
                        chg_df["$change"],
                    )
                ]
            else:
                continue
        except Exception:
            continue
        sell_date = determine_ipo_sell_date(sym, pos["buy_date"], daily_changes)
        if sell_date is None or sell_date > ctx.trade_date:
            continue
        order_date = next_td_str
        already = ctx.conn.execute(
            "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
            "WHERE symbol=? AND side='sell' AND status IN ('pending','carry')",
            (sym,),
        ).fetchone()[0]
        to_insert = pos["qty"] - already
        if to_insert > 0:
            insert_order(ctx.conn, order_date, sym, "sell", to_insert, None, "pending", 0, ctx.trade_date)
        snapshot_positions(ctx.conn, ctx.trade_date, ctx.current_positions)

    # 11d. persist post-IPO nav if cash changed
    if ctx.ipo_cash_changed and ctx.settle_result is not None:
        market_value = sum(
            pos["qty"] * ctx.prices.get(s, {}).get("close", pos["avg_cost"])
            for s, pos in ctx.current_positions.items()
            if pos["qty"] > 0
        )
        new_total_nav = market_value + ctx.cash
        hedge_val = sum(
            ctx.current_positions.get(s, {}).get("market_value") or 0
            for s in ctx.hedge_symbols
        )
        equity_val = new_total_nav - ctx.cash - hedge_val
        record_nav(
            ctx.conn, ctx.trade_date, ctx.cash, market_value, new_total_nav,
            ctx.settle_result.pre_trade_nav, ctx.settle_result.post_trade_nav,
            ctx.benchmarks["csi300"], ctx.benchmarks["csi1000"],
            hedge_value=hedge_val, equity_value=equity_val,
        )


def _step12_backup_and_finalize(ctx: DailyRunContext) -> None:
    """Commit, integrity-gated backup, golden monthly backup, cleanup, record settled."""
    # Pre-commit NAV gate: if NAV is anomalous (>20% change), delete
    # today's NAV row before commit to prevent corrupted data from
    # persisting. Step 13 report gate also catches this, but the
    # corrupted row would still be in DB for next day's risk engine.
    try:
        nav_rows = ctx.conn.execute(
            "SELECT trade_date, total_nav FROM nav ORDER BY trade_date DESC LIMIT 2"
        ).fetchall()
        if len(nav_rows) >= 2:
            today_nav = float(nav_rows[0]["total_nav"])
            prev_nav = float(nav_rows[1]["total_nav"])
            if prev_nav > 0:
                nav_change = (today_nav - prev_nav) / prev_nav
                # Only delete on suspicious GAINS (likely data error).
                # Preserve real crashes so step13 crash alert can fire.
                if nav_change > 0.20:
                    logger.error(
                        "[pre-commit] NAV anomalous gain: +%.0f%% (%.0f -> %.0f), "
                        "removing today's NAV row to prevent corrupted persistence",
                        nav_change * 100, prev_nav, today_nav,
                    )
                    ctx.conn.execute(
                        "DELETE FROM nav WHERE trade_date = ?",
                        (ctx.trade_date,),
                    )
    except Exception:
        logger.warning("[pre-commit] NAV gate check failed", exc_info=True)

    ctx.conn.commit()

    # Integrity gate: check DB health before backup
    backup_dir = PROJECT_ROOT / "backups"
    backup_path = backup_dir / f"paper_{ctx.trade_date}.db"

    backup_result = hot_backup_with_integrity(ctx.db_path, backup_path)
    if not backup_result.success:
        logger.critical(
            "Backup failed integrity check for %s -- no backup created. "
            "Operator must investigate database health before next run. "
            "Pipeline result still recorded (pipeline did execute).",
            ctx.trade_date,
        )

    cleanup_old_backups(backup_dir, ctx.paper_cfg["backup_retention_days"])

    # Monthly golden backup (immutable, never cleaned up)
    create_golden_backup(ctx.db_path, backup_dir)

    record_run(ctx.conn, ctx.trade_date, "settled")
    ctx.conn.commit()


def _step12_book_b(ctx: DailyRunContext) -> None:
    """Run Book B shadow ledger (fail-open, never affects production).

    Book B is an in-process shadow of Book A: shares ctx.prices,
    prediction parquet, universe, industry_map, benchmarks.  Owns only
    its book state (positions, cash, orders, nav) in paper_b.db.

    The timing multiplier m scales target weights at order generation.
    With m=1.0, Book B must reproduce Book A to the cent (R3).
    """
    try:
        _run_book_b(ctx)
    except Exception:
        logger.warning("[book_b] Book B failed for %s, continuing", ctx.trade_date, exc_info=True)


def _run_book_b(ctx: DailyRunContext) -> None:
    """Book B implementation.  Isolated for testability."""
    book_id = "none"  # future: "n1", "n2", "n3"
    m = timing_multiplier(ctx.trade_date, model=book_id)

    # Open Book B database
    db_dir = ctx.db_path.parent if ctx.db_path else Path("data")
    book_b_path = db_dir / f"paper_b_{book_id}.db"
    if not book_b_path.exists():
        # Day 0 bootstrap: copy production DB
        _bootstrap_book_b(ctx.db_path, book_b_path)

    from ashare_lab.paper.ledger import get_connection  # noqa: PLC0415
    book_conn = get_connection(book_b_path)
    try:
        # Load Book B state
        from ashare_lab.paper.ledger import (  # noqa: PLC0415
            get_latest_positions, get_latest_cash, get_cooldowns,
            delete_expired_cooldowns,
        )
        from ashare_lab.paper.risk import manage_trailing_cooldown  # noqa: PLC0415
        book_positions = get_latest_positions(book_conn)
        book_cash = get_latest_cash(book_conn, ctx.paper_cfg["initial_cash"])
        book_cooldown = get_cooldowns(book_conn)
        book_cooldown = manage_trailing_cooldown(book_cooldown, ctx.trade_date)
        delete_expired_cooldowns(book_conn, ctx.trade_date)

        # PHASE 1: Settle yesterday's Book B orders (T+1)
        # Must happen BEFORE computing NAV/signals/target_value,
        # mirroring Book A's flow: _step8_settle -> _step10_signals.
        from ashare_lab.paper.engine import round_lots, settle_day  # noqa: PLC0415

        pending_orders = [
            dict(r) for r in book_conn.execute(
                "SELECT id, symbol, side, target_qty, carry_day, reset_count, "
                "suspension_carry_day FROM orders "
                "WHERE status IN ('pending','carry') AND trade_date <= ?",
                (ctx.trade_date,),
            ).fetchall()
        ]

        if pending_orders:
            settle_day(
                book_conn, ctx.trade_date, pending_orders, ctx.prices,
                book_positions, book_cash, set(), ctx.benchmarks, ctx.config,
            )
            # Reload post-settle state
            book_positions = get_latest_positions(book_conn)
            book_cash = get_latest_cash(book_conn, ctx.paper_cfg["initial_cash"])

        # Compute Book B NAV from post-settle state
        book_total_nav = sum(
            pos["market_value"] for pos in book_positions.values()
        ) + book_cash

        # Generate Book B orders (same signals, m-scaled target)
        from ashare_lab.paper.signal import (  # noqa: PLC0415
            generate_signals, filter_candidates, topk_dropout_orders,
        )

        signals = generate_signals(ctx.trade_date, pred_path=ctx.pred_path)
        candidate_syms = [s for s in signals if s in ctx.universe_symbols and s not in ctx.ipo_listing_syms and s not in ctx.st_names]
        filtered_syms = filter_candidates(
            candidate_syms, ctx.market_data,
            ctx.config.get("universe", {}).get("listing_min_days", 60),
            ctx.config.get("universe", {}).get("min_avg_turnover_20d", 0.0),
            ctx.config.get("universe", {}).get("exclude_close_above_cny", 300.0),
        )
        filtered_signals = {s: signals[s] for s in filtered_syms}

        effective_topk = ctx.risk_result.topk_override or ctx.paper_cfg["topk"]
        ipo_held = {
            s for s in book_positions
            if ctx.market_data.get(s, {}).get("listing_days", 999)
            < ctx.paper_cfg.get("listing_min_days", 60)
        }
        held_set = set(book_positions.keys()) - ipo_held
        topk_for_dropout = max(0, effective_topk - len(ipo_held))
        sell_syms, buy_syms = topk_dropout_orders(
            filtered_signals, held_set, topk_for_dropout, ctx.paper_cfg.get("n_drop", 1),
        )

        # Apply risk filters (same as Book A)
        buy_syms = [
            s for s in buy_syms
            if ctx.industry_map.get(s) not in ctx.risk_result.blocked_industries
            and s not in ctx.risk_result.blocked_rebuys
        ]
        if ctx.risk_result.buying_halted:
            buy_syms = []

        # Compute m-scaled target value from post-settle NAV
        if effective_topk <= 0:
            target_value = 0.0
            buy_syms = []
        else:
            equity_nav = book_total_nav
            target_value = (equity_nav * ctx.config["cost_model"]["risk_degree"] * m) / effective_topk

        # PHASE 2: Generate next-day orders (not settled today)
        # These orders have trade_date=next_td and settle on the NEXT run.
        next_td_str = next_trading_day(dt.date.fromisoformat(ctx.trade_date)).isoformat()

        forced_sells = ctx.risk_result.forced_sells
        sell_set = set(sell_syms) | set(forced_sells.keys())
        desired_sell_qty: dict[str, int] = {}
        for s in sell_set:
            if s in sell_syms:
                desired_sell_qty[s] = book_positions.get(s, {}).get("qty", 0)
            else:
                desired_sell_qty[s] = forced_sells[s]

        for s in sell_set:
            to_insert = desired_sell_qty[s]
            if to_insert > 0:
                insert_order(book_conn, next_td_str, s, "sell", to_insert, None, "pending", 0, ctx.trade_date)

        for s in buy_syms:
            close_price = ctx.prices.get(s, {}).get("close")
            if not close_price or close_price <= 0:
                continue
            target_qty = round_lots(target_value / close_price, "buy")
            if target_qty <= 0:
                continue
            insert_order(book_conn, next_td_str, s, "buy", target_qty, None, "pending", 0, ctx.trade_date)

        # Write A/B/C artifact
        _write_abc_artifact(ctx, book_conn, book_id, m)

        book_conn.commit()
        logger.info("[book_b] completed for %s (m=%.1f, book_id=%s)", ctx.trade_date, m, book_id)
    finally:
        book_conn.close()


def _bootstrap_book_b(prod_db_path: Path, book_b_path: Path) -> None:
    """Day 0 bootstrap: copy production DB to Book B DB."""
    import shutil  # noqa: PLC0415
    shutil.copy2(prod_db_path, book_b_path)
    logger.info("[book_b] bootstrapped %s -> %s", prod_db_path, book_b_path)


def _write_abc_artifact(ctx: DailyRunContext, book_conn, book_id: str, m: float) -> None:
    """Write daily A/B/C comparison artifact to experiments/control_books/."""
    from ashare_lab.paper.ledger import get_latest_positions, get_latest_cash  # noqa: PLC0415

    # Book A NAV (from production)
    nav_a = ctx.total_nav

    # Book B NAV
    book_positions = get_latest_positions(book_conn)
    book_cash = get_latest_cash(book_conn, ctx.paper_cfg["initial_cash"])
    nav_b = sum(pos["market_value"] for pos in book_positions.values()) + book_cash

    # Book C (benchmark CSI1000)
    nav_c = ctx.benchmarks.get("csi1000", 0.0)

    # Excess returns
    excess_a = nav_a / nav_c - 1.0 if nav_c > 0 else 0.0
    excess_b = nav_b / nav_c - 1.0 if nav_c > 0 else 0.0

    # Write artifact
    artifact_dir = Path("experiments/control_books")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "date": ctx.trade_date,
        "book_id": book_id,
        "nav_a": round(nav_a, 2),
        "nav_b": round(nav_b, 2),
        "nav_c": round(nav_c, 2),
        "excess_a": round(excess_a, 6),
        "excess_b": round(excess_b, 6),
        "m": m,
    }
    artifact_path = artifact_dir / f"{ctx.trade_date}_{book_id}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[book_b] A/B/C artifact written: %s", artifact_path)


def _step13_report(ctx: DailyRunContext) -> None:
    """Deliver WeChat report (non-blocking)."""
    logger.info("[report] entered for %s (steps=%s)", ctx.trade_date, ctx.steps)
    try:
        stale_flag = os.environ.get("ASHARE_USE_STALE", "")
        if stale_flag and stale_flag != "1":
            logger.warning("[report] ASHARE_USE_STALE='%s' (expected '1' or empty), treating as stale", stale_flag)
        if stale_flag == "1":
            logger.info("[report] skipping: stale predictions (GPU inference failed)")
            return
        # Report quality gate: NAV reasonableness
        nav_rows = ctx.conn.execute(
            "SELECT trade_date, total_nav FROM nav ORDER BY trade_date DESC LIMIT 2"
        ).fetchall()
        if len(nav_rows) >= 2:
            today_nav = float(nav_rows[0]["total_nav"])
            prev_nav = float(nav_rows[1]["total_nav"])
            if prev_nav > 0:
                nav_change = (today_nav - prev_nav) / prev_nav
                if nav_change > 0.20:
                    logger.error(
                        "[report] gate: NAV jumped %.0f%% (%.0f -> %.0f), "
                        "likely data error, skipping report",
                        nav_change * 100, prev_nav, today_nav)
                    return
                if nav_change < -0.20:
                    logger.critical(
                        "[report] CRASH ALERT: NAV dropped %.0f%% (%.0f -> %.0f)!",
                        abs(nav_change) * 100, prev_nav, today_nav)
                    try:
                        subprocess.run(
                            [sys.executable,
                             str(PROJECT_ROOT / "scripts" / "alert.py"),
                             "1", "crash_alert", "0"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=30,
                        )
                    except Exception:
                        pass  # alert is best-effort
            else:
                logger.info("[report] NAV gate: prev_nav=%.0f <= 0, skipping NAV check", prev_nav)
        else:
            logger.info("[report] NAV gate: only %d rows, skipping NAV check", len(nav_rows))
        # Report quality gate: normalized price detection
        for sym, pdata in ctx.prices.items():
            close = pdata.get("close")
            if close is None or close <= 0:
                continue
            if _is_normalized_price(close, pdata.get("factor"), pdata.get("adjusted", False)):
                logger.error("[report] gate: normalized price (%s=%.4f factor=%s), skip report",
                             sym, close, pdata.get("factor"))
                return
        if ctx.steps is not None and "report" not in ctx.steps:
            logger.info("[report] skipping: steps=%s does not include 'report'", ctx.steps)
            return
        logger.info("[report] calling generate_and_send_report for %s", ctx.trade_date)
        from ashare_lab.paper.report import generate_and_send_report
        report_rc = generate_and_send_report(ctx.trade_date, ctx.conn, ctx.config)
        if report_rc != 0:
            logger.warning("[report] delivery failed for %s (rc=%d)", ctx.trade_date, report_rc)
        else:
            logger.info("[report] delivery succeeded for %s", ctx.trade_date)
    except Exception as exc:
        logger.warning("[report] step failed for %s: %s", ctx.trade_date, exc, exc_info=True)
    finally:
        ctx.conn.commit()


def _step14_record_pipeline_run(ctx: DailyRunContext) -> None:
    """Close the pipeline_runs row as success or stale."""
    stale_flag = os.environ.get("ASHARE_USE_STALE", "")
    status = "stale" if stale_flag == "1" else "success"
    _close_pipeline_run(ctx, status)
    ctx.conn.commit()


def _step15_graduation(ctx: DailyRunContext) -> None:
    """Check graduation gate (non-blocking, skip when stale)."""
    stale_flag = os.environ.get("ASHARE_USE_STALE", "")
    if stale_flag == "1":
        return
    try:
        from ashare_lab.paper.graduation import check_graduation, notify_graduation  # noqa: PLC0415
        passed, stats = check_graduation(ctx.conn)
        logger.info(
            "Graduation gate: passed=%s rate=%.1f%% days_remaining=%d",
            passed, stats["rate"] * 100, stats["days_remaining"],
        )
        if passed and stats.get("should_notify"):
            notify_graduation(ctx.conn, stats)
    except Exception:
        logger.warning("Graduation check failed", exc_info=True)


def _close_pipeline_run(ctx: DailyRunContext, status: str, error_msg: str | None = None) -> None:
    """Close the open pipeline_runs row to its final status.

    Sets pipeline_run_id to None after successful close so the generic
    uncaught-exception handler cannot overwrite a deliberate close
    (close-once semantics).
    """
    if ctx.pipeline_run_id is None:
        return
    try:
        duration_s = time.monotonic() - ctx.start_time
        close_pipeline_run(ctx.conn, ctx.pipeline_run_id, status, duration_s, error_msg)
        ctx.conn.commit()
        ctx.pipeline_run_id = None  # close-once: prevent overwrite
    except Exception:
        logger.warning("Failed to close pipeline_run id=%s", ctx.pipeline_run_id, exc_info=True)


def run_daily(
    trade_date: str | None = None,
    force: bool = False,
    steps: set[str] | None = None,
    pred_path: Path | None = None,
) -> int:
    """Run the daily post-close pipeline.

    Parameters
    ----------
    trade_date : str or None
        ISO date (YYYY-MM-DD).  None = latest trading day.
    force : bool
        If True, reset the day and re-run.
    steps : set or None
        Subset of steps to run: {"settle"} or {"signal"}.
        None = all steps.
    pred_path : Path or None
        Explicit prediction file.  None = resolve from PREDICTIONS_DIR.

    Returns
    -------
    int
        0 = success, 1 = skipped (stale/idempotent), 2 = error.
    """
    # Step 0: resolve and validate trade_date
    if trade_date is None:
        trade_date = latest_trading_day().isoformat()
    if not _DATE_RE.match(trade_date):
        logger.error("Invalid date format: %s", trade_date)
        return 2

    ctx = DailyRunContext(
        trade_date=trade_date,
        force=force,
        steps=steps,
        pred_path=pred_path,
        start_time=time.monotonic(),
        predictions_date_str=trade_date,
    )

    _step1_init(ctx)

    # Open pipeline_runs row (fail-open: if insert fails, continue without recording)
    try:
        ctx.pipeline_run_id = open_pipeline_run(
            ctx.conn, ctx.trade_date, ctx.predictions_date_str,
        )
    except Exception:
        logger.warning("Failed to open pipeline_run, continuing without recording", exc_info=True)

    try:
        rc = _step2_idempotency(ctx)
        if rc == 0:   # already settled -- close as success
            _close_pipeline_run(ctx, "success")
            return 0
        if rc == 2:
            return 2

        rc = _step3_data_update(ctx)
        if rc == 2:
            return 2
        if rc == 1:
            _close_pipeline_run(ctx, "stale", f"data stale for {ctx.trade_date}, skipping")
            return 1

        try:
            _gate_data_completeness(ctx)
        except RuntimeError as exc:
            _close_pipeline_run(ctx, "halted", str(exc))
            raise

        _step4_load_state(ctx)

        rc = _step5_fetch_prices_and_universe(ctx)
        if rc == 2:
            return 2

        try:
            _gate_price_sanity(ctx)
        except RuntimeError as exc:
            _close_pipeline_run(ctx, "halted", str(exc))
            raise

        _step6_adjustfactor(ctx)
        _step7_csi1000_exits(ctx)
        _prefetch_hedge_prices_for_settle(ctx)
        rc = _step8_settle(ctx)
        if rc == 2:
            _close_pipeline_run(ctx, "error", "settle failed")
            return 2
        _step9_risk_checks(ctx)
        _step9b_hedge_sleeve(ctx)
        _step9c_nav_hedge_split(ctx)

        rc = _step10_signal_generation(ctx)
        if rc == 2:
            return 2

        _step11_ipo_processing(ctx)
        _step12_backup_and_finalize(ctx)
        _step12_book_b(ctx)
        _step13_report(ctx)
        _step14_record_pipeline_run(ctx)
        _step15_graduation(ctx)

        logger.info("Pipeline completed for %s", trade_date)
        return 0

    except Exception as exc:
        _close_pipeline_run(ctx, "error", f"uncaught exception: {exc}")
        raise



# ------------------------------------------------------------------
# Backfill
# ------------------------------------------------------------------


def run_backfill(
    from_date: str, to_date: str, force: bool = False,
) -> int:
    """Replay the pipeline over a date range sequentially (D-15).

    Force mode tears down settled days newest-first, then replays
    forward.

    Returns 0 on success, 1 if any day skipped, 2 on error.
    """
    days = [
        d.isoformat()
        for d in trading_days_between(
            dt.date.fromisoformat(from_date),
            dt.date.fromisoformat(to_date),
        )
    ]

    if force:
        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]
        conn = get_connection(db_path)
        init_schema(conn)
        for d in reversed(days):
            force_reset_day(conn, d)
        conn.commit()

    had_skip = False
    for date_str in days:
        pred_path = _resolve_prediction_file(date_str)
        if pred_path is None:
            logger.warning(
                "Skipping %s: no prediction file", date_str,
            )
            had_skip = True
            continue
        rc = run_daily(date_str, force=False, steps={"settle", "signal"}, pred_path=pred_path)
        if rc == 2:
            return 2
        if rc == 1:
            had_skip = True
        logger.info("Backfill %s: rc=%d", date_str, rc)

    return 1 if had_skip else 0


# ------------------------------------------------------------------
# IC history tracking (for event-driven retraining)
# ------------------------------------------------------------------


def append_ic_history(pred_path: Path, trade_date: str) -> None:
    """Append today's IC value to data/ic_history.tsv.

    Reads meta.json next to the prediction file, extracts the ``ic``
    field (may be ``None``), and appends ``{date}\\t{ic}`` to the
    TSV file.  Deduplicates by date and keeps only the last 30 rows.
    """
    meta_path = pred_path.with_suffix(".meta.json")
    if not meta_path.exists():
        logger.debug("IC history: no meta.json at %s, skipping", meta_path)
        return

    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except Exception:
        logger.debug("IC history: failed to read %s", meta_path)
        return

    ic_val = meta.get("ic")
    # Normalize NaN to empty string for the TSV
    import math as _math  # noqa: PLC0415
    if ic_val is not None and isinstance(ic_val, float) and _math.isnan(ic_val):
        ic_val = None

    history_path = PROJECT_ROOT / "data" / "ic_history.tsv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing entries
    entries: list[tuple[str, str]] = []
    if history_path.exists():
        with history_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    entries.append((parts[0], parts[1]))

    # Skip if today already recorded
    for d, _ in entries:
        if d == trade_date:
            logger.debug("IC history: %s already recorded, skipping", trade_date)
            return

    ic_str = str(ic_val) if ic_val is not None else ""
    entries.append((trade_date, ic_str))

    # Sort by date to handle backfills correctly
    entries.sort(key=lambda e: e[0])

    # Keep only last 30 entries (by date, not file order)
    entries = entries[-30:]

    # Atomic write: write to temp file then rename
    tmp_path = history_path.with_suffix(".tsv.tmp")
    with tmp_path.open("w") as f:
        for d, v in entries:
            f.write(f"{d}\t{v}\n")
    tmp_path.rename(history_path)

    logger.info("IC history: recorded %s -> %s", trade_date, ic_str or "null")


# ------------------------------------------------------------------
# Status query
# ------------------------------------------------------------------


def get_status() -> dict:
    """Return current ledger status for CLI display."""
    config = load_config()
    db_path = PROJECT_ROOT / config["paper"]["db_path"]
    conn = get_connection(db_path)
    init_schema(conn)

    last_run = conn.execute(
        "SELECT trade_date, status FROM runs "
        "ORDER BY trade_date DESC LIMIT 1"
    ).fetchone()

    nav_row = conn.execute(
        "SELECT total_nav, cash FROM nav "
        "ORDER BY trade_date DESC LIMIT 1"
    ).fetchone()

    pos_count = conn.execute(
        "SELECT COUNT(DISTINCT symbol) FROM positions "
        "WHERE trade_date = (SELECT MAX(trade_date) FROM positions)"
    ).fetchone()[0]

    pending_count = conn.execute(
        "SELECT COUNT(*) FROM orders "
        "WHERE status IN ('pending','carry')"
    ).fetchone()[0]

    return {
        "last_trade_date": last_run["trade_date"] if last_run else None,
        "run_status": last_run["status"] if last_run else None,
        "total_nav": float(nav_row["total_nav"]) if nav_row else None,
        "cash": float(nav_row["cash"]) if nav_row else None,
        "position_count": pos_count,
        "pending_orders": pending_count,
    }
