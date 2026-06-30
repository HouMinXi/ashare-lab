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
import logging
import os
import re
import time
from dataclasses import replace
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
from ashare_lab.paper.ipo import (
    check_ipo_subscription,
    detect_board_type,
    determine_ipo_sell_date,
)
from ashare_lab.paper.ledger import (
    bump_carry_days,
    cleanup_old_backups,
    compute_nav,
    delete_expired_cooldowns,
    force_reset_day,
    get_connection,
    get_cooldowns,
    get_latest_cash,
    get_latest_positions,
    hot_backup,
    init_schema,
    insert_order,
    is_day_settled,
    log_settle_change,
    record_nav,
    record_run,
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
                sym = row.get("symbol", "")
                if sym in symbols:
                    result[sym] = row.get("name", "")
        if result:
            return result

    # Baostock fallback: code column uses baostock format (sh.600006)
    bs_path = _BS_CACHE_DIR / "stock_names.csv"
    if not bs_path.exists():
        return None
    result = {}
    with open(bs_path, newline="") as f:
        for row in csv.DictReader(f):
            sym = _bs_code_to_universe(row["code"], symbols)
            if sym:
                result[sym] = row["code_name"]
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
        resp = requests.get(url, timeout=10)
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
    """Fetch benchmark closes: cache first, baostock fallback.

    Returns {"csi300": float, "csi1000": float}.
    On failure returns zeros with warning log.
    """
    cached = _load_benchmark_cache(trade_date)
    if cached is not None:
        logger.debug("benchmark from cache for %s", trade_date)
        return {"csi300": cached.get("csi300", 0.0),
                "csi1000": cached.get("csi1000", 0.0)}

    try:
        import baostock as bs  # noqa: PLC0415
    except ImportError:
        logger.warning("baostock not installed, benchmark unavailable")
        return {"csi300": 0.0, "csi1000": 0.0}

    result = {"csi300": 0.0, "csi1000": 0.0}
    indices = [("sh.000300", "csi300"), ("sh.000852", "csi1000")]

    login_result = bs.login()
    if login_result.error_code != "0":
        logger.warning("baostock login failed: %s", login_result.error_msg)
        return result

    try:
        for code, key in indices:
            rs = bs.query_history_k_data_plus(
                code,
                "close",
                start_date=trade_date,
                end_date=trade_date,
                frequency="d",
            )
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                try:
                    result[key] = float(row[0])
                except (IndexError, ValueError, TypeError):
                    pass
    finally:
        bs.logout()

    return result


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
# Main pipeline
# ------------------------------------------------------------------


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
    # -- Step 0: resolve and validate trade_date --
    if trade_date is None:
        trade_date = latest_trading_day().isoformat()
    if not _DATE_RE.match(trade_date):
        logger.error("Invalid date format: %s", trade_date)
        return 2

    start_time = time.monotonic()
    predictions_date_str = trade_date  # early default; overwritten when pred_path resolves

    # -- Step 1: init --
    config = load_config()
    paper_cfg = config["paper"]
    risk_cfg = paper_cfg["risk"]
    db_path = PROJECT_ROOT / paper_cfg["db_path"]
    conn = get_connection(db_path)
    init_schema(conn)

    # -- Step 2: idempotency (D-02) --
    if not force and is_day_settled(conn, trade_date):
        logger.info("Already settled: %s", trade_date)
        return 0

    if force:
        latest_settled = conn.execute(
            "SELECT MAX(trade_date) FROM runs WHERE status='settled'"
        ).fetchone()[0]
        if latest_settled is not None and trade_date < latest_settled:
            logger.error(
                "force-reset of %s leaves settled days > it stale; "
                "run `paper backfill %s %s --force` to replay the "
                "coupled range",
                trade_date, trade_date, latest_settled,
            )
            return 2
        force_reset_day(conn, trade_date)

    # -- Step 3: data update (skip for signal-only and historical dates) --
    # daily_refresh checks TODAY's feed freshness; for historical backfill
    # dates the data was collected long ago and staleness is irrelevant.
    # Only check when trade_date is within ~10 calendar days of today.
    if steps != {"signal"}:
        td = dt.date.fromisoformat(trade_date)
        recent_cutoff = dt.date.today() - dt.timedelta(days=10)
        if td >= recent_cutoff:
            try:
                from ashare_lab.data.update import daily_refresh  # noqa: PLC0415
                stale = daily_refresh()
            except Exception as exc:
                logger.error("data refresh failed", exc_info=True)
                record_run(conn, trade_date, "error")
                try:
                    from ashare_lab.paper.ledger import insert_pipeline_run  # noqa: PLC0415
                    insert_pipeline_run(conn, trade_date, "error",
                                       time.monotonic() - start_time,
                                       str(exc), predictions_date_str)
                except Exception:
                    logger.warning("Failed to record pipeline_run", exc_info=True)
                conn.commit()
                return 2
            if stale == 1:
                record_run(conn, trade_date, "skipped_stale")
                conn.commit()
                logger.warning("Data stale for %s, skipping", trade_date)
                return 1

    # -- Step 4: load state --
    current_positions = get_latest_positions(conn)
    cash = get_latest_cash(conn, paper_cfg["initial_cash"])
    cooldown_state = get_cooldowns(conn)
    cooldown_state = manage_trailing_cooldown(cooldown_state, trade_date)
    delete_expired_cooldowns(conn, trade_date)
    # Explicit in-memory guard: keep only active cooldowns (R-21, >= )
    cooldown_state = {
        s: cd for s, cd in cooldown_state.items()
        if cd["cooldown_until"] >= trade_date
    }

    # is_soft_reduced from paper_state
    row = conn.execute(
        "SELECT value FROM paper_state WHERE key='is_soft_reduced'"
    ).fetchone()
    if row is not None:
        is_soft_reduced = row["value"] == "true"
    else:
        is_soft_reduced = False
        conn.execute(
            "INSERT INTO paper_state (key, value) "
            "VALUES ('is_soft_reduced', 'false')"
        )
        conn.commit()

    # -- Step 5: fetch prices and universe (deferred imports) --
    try:
        from qlib.data import D  # noqa: PLC0415
    except ImportError as exc:
        logger.error("qlib not available, cannot run pipeline")
        record_run(conn, trade_date, "error")
        try:
            from ashare_lab.paper.ledger import insert_pipeline_run  # noqa: PLC0415
            insert_pipeline_run(conn, trade_date, "error",
                               time.monotonic() - start_time,
                               str(exc), predictions_date_str)
        except Exception:
            logger.warning("Failed to record pipeline_run", exc_info=True)
        conn.commit()
        return 2

    # 5a. universe (csi1000 from qlib)
    universe_symbols = D.list_instruments(
        D.instruments("csi1000"),
        start_time=trade_date,
        end_time=trade_date,
        as_list=True,
    )

    # 5b. IPO calendar -- single fetch and evaluate
    next_td = next_trading_day(
        dt.date.fromisoformat(trade_date)
    ).isoformat()
    ipo_calendar = _fetch_ipo_calendar(next_td)
    ipo_won = [
        row
        for row in ipo_calendar
        if check_ipo_subscription(row["ceiling_lots"], row["win_rate"])[0]
    ]
    ipo_listing_syms = {
        row["symbol"]
        for row in ipo_won
        if row["listing_date"] == trade_date
    }

    # order symbols needing price data
    order_syms = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM orders "
            "WHERE status IN ('pending','carry')"
        )
    }
    fetch_symbols = (
        set(universe_symbols)
        | set(current_positions.keys())
        | order_syms
        | ipo_listing_syms
    )

    # 5c. ST names + prices via cache / baostock + qlib
    st_cached = _load_st_cache(trade_date, fetch_symbols)
    if st_cached is not None:
        st_names = st_cached
        logger.debug("ST names from cache for %s", trade_date)
    else:
        st_names: set[str] = set()
        try:
            import baostock as bs  # noqa: PLC0415

            login_r = bs.login()
            if login_r.error_code == "0":
                for sym in fetch_symbols:
                    code = sym.lower()
                    if len(code) == 6:
                        prefix = "sh" if code.startswith("6") else "sz"
                        code = f"{prefix}.{code}"
                    elif not code.startswith(("sh.", "sz.")):
                        prefix = "sh" if sym.startswith(("SH", "6")) else "sz"
                        code = f"{prefix}.{sym[-6:]}"
                    rs = bs.query_stock_basic(code=code, code_name="")
                    while rs.error_code == "0" and rs.next():
                        row_data = rs.get_row_data()
                        if len(row_data) > 1 and "ST" in str(row_data[1]).upper():
                            st_names.add(sym)
                bs.logout()
        except ImportError:
            logger.warning("baostock not installed, ST detection unavailable")

    # Prices from qlib
    prices: dict[str, dict] = {}
    if fetch_symbols:
        raw = D.features(
            instruments=list(fetch_symbols),
            fields=["$close", "$change", "$volume", "$factor"],
            start_time=trade_date,
            end_time=trade_date,
        )
        if raw is not None and not raw.empty:
            for idx, row_s in raw.iterrows():
                # idx is (instrument, datetime) in qlib MultiIndex
                inst = idx[0] if isinstance(idx, tuple) else str(idx)
                sym = str(inst)[-6:] if len(str(inst)) > 6 else str(inst)
                # Normalize to match universe symbol format
                for fs in fetch_symbols:
                    if fs.endswith(sym):
                        sym = fs
                        break
                prices[sym] = {
                    "close": float(row_s.get("$close", 0)),
                    "change": float(row_s.get("$change", 0)),
                    "volume": float(row_s.get("$volume", 0)),
                    "factor": float(row_s.get("$factor", 1.0)),
                    "threshold": get_limit_threshold(sym, st_names),
                }

    # 5d. benchmark closes
    benchmarks = _fetch_benchmark_closes(trade_date)

    # 5e. CSI1000 11-day closes for regime check (cache / baostock)
    regime_cached = _load_regime_cache(trade_date)
    if regime_cached is not None:
        csi1000_closes_11d = regime_cached
        logger.debug("CSI1000 regime from cache for %s", trade_date)
    else:
        csi1000_closes_11d: list[float] = []
        start_60d = (
            dt.date.fromisoformat(trade_date) - dt.timedelta(days=60)
        ).isoformat()
        try:
            import baostock as bs  # noqa: PLC0415

            login_r = bs.login()
            if login_r.error_code == "0":
                rs = bs.query_history_k_data_plus(
                    "sh.000852",
                    "close",
                    start_date=start_60d,
                    end_date=trade_date,
                    frequency="d",
                )
                closes: list[float] = []
                while rs.error_code == "0" and rs.next():
                    row_data = rs.get_row_data()
                    try:
                        closes.append(float(row_data[0]))
                    except (IndexError, ValueError, TypeError):
                        pass
                csi1000_closes_11d = (
                    closes[-11:] if len(closes) >= 11 else closes
                )
                bs.logout()
        except ImportError:
            pass

    # 5f. industry_map via cache / baostock
    ind_cached = _load_industry_cache(trade_date, fetch_symbols)
    if ind_cached is not None:
        industry_map = ind_cached
        logger.debug("industry_map from cache for %s", trade_date)
    else:
        industry_map: dict[str, str] = {}
        try:
            import baostock as bs  # noqa: PLC0415

            login_r = bs.login()
            if login_r.error_code == "0":
                for sym in fetch_symbols:
                    code = sym.lower()
                    if len(code) == 6:
                        prefix = "sh" if code.startswith("6") else "sz"
                        code = f"{prefix}.{code}"
                    elif not code.startswith(("sh.", "sz.")):
                        prefix = "sh" if sym.startswith(("SH", "6")) else "sz"
                        code = f"{prefix}.{sym[-6:]}"
                    rs = bs.query_stock_industry(code=code, date=trade_date)
                    while rs.error_code == "0" and rs.next():
                        row_data = rs.get_row_data()
                        if len(row_data) > 3 and row_data[3]:
                            industry_map[sym] = row_data[3]
                bs.logout()
        except ImportError:
            pass

    # 5g. market_data for filter_candidates
    market_data: dict[str, dict] = {}
    for sym in fetch_symbols:
        p = prices.get(sym)
        if p is None:
            continue

        # listing_days: trading days since first qlib bar
        listing_days = 999
        try:
            first_bar_df = D.features(
                instruments=[sym],
                fields=["$close"],
                start_time="2005-01-01",
                end_time=trade_date,
            )
            if first_bar_df is not None and not first_bar_df.empty:
                first_date = first_bar_df.index[0]
                if isinstance(first_date, tuple):
                    first_date = first_date[0]
                first_bar_date = first_date.date() if hasattr(
                    first_date, "date"
                ) else first_date
                listing_days = len(
                    trading_days_between(first_bar_date, dt.date.fromisoformat(trade_date))
                )
        except Exception:
            pass

        # avg_turnover_20d
        avg_turnover_20d = 0.0
        try:
            turnover_df = D.features(
                instruments=[sym],
                fields=["$volume", "$close", "$factor"],
                start_time=(
                    dt.date.fromisoformat(trade_date) - dt.timedelta(days=40)
                ).isoformat(),
                end_time=trade_date,
            )
            if turnover_df is not None and not turnover_df.empty:
                # qlib $close is forward-adjusted; divide by $factor
                # to recover actual CNY price for turnover calculation.
                factor = turnover_df["$factor"].replace(0, 1)
                unadj_close = turnover_df["$close"] / factor
                turnover_vals = (
                    turnover_df["$volume"] * unadj_close
                ).tail(20)
                if len(turnover_vals) > 0:
                    avg_turnover_20d = float(turnover_vals.mean())
        except Exception:
            pass

        market_data[sym] = {
            "close": p["close"],
            "listing_days": listing_days,
            "avg_turnover_20d": avg_turnover_20d,
        }

    # -- Step 6: adjustfactor (D-47, audit-only) --
    previous_factors = {
        s: pos["factor"] for s, pos in current_positions.items()
    }
    current_factors = {
        s: prices[s]["factor"]
        for s in current_positions
        if s in prices
    }
    current_positions, adj_records = check_and_apply_adjustfactor(
        current_positions, previous_factors, current_factors
    )
    if adj_records:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"adjust_{trade_date}.jsonl"
        with log_path.open("a") as f:
            for rec in adj_records:
                f.write(json.dumps(rec) + "\n")
        logger.warning(
            "Adjustfactor changes detected for %d symbols on %s",
            len(adj_records), trade_date,
        )

    # -- Step 7: CSI1000 exit detection (D-34) --
    universe_set = set(universe_symbols)
    csi1000_exits = {
        s for s in current_positions if s not in universe_set
    }
    for s in csi1000_exits:
        existing = conn.execute(
            "SELECT id FROM orders WHERE symbol=? AND side='sell' "
            "AND status IN ('pending','carry') AND trade_date <= ?",
            (s, trade_date),
        ).fetchone()
        if existing is None:
            qty = current_positions[s]["qty"]
            next_td_date = next_trading_day(
                dt.date.fromisoformat(trade_date)
            ).isoformat()
            # CSI1000 exit: next-day enforcement (post-close pipeline)
            insert_order(
                conn, next_td_date, s, "sell", qty, None,
                "pending", 0, trade_date,
            )

    # -- Step 8: settle (skip for signal-only) --
    settle_result = None
    if steps != {"signal"}:
        # 8a-pre. active cooldowns for carry-cancel
        active_cooldowns = {
            s for s, cd in cooldown_state.items()
            if cd["cooldown_until"] >= trade_date
        }

        # 8a. yesterday's signals for TopK recheck
        prev_date = previous_trading_day(
            dt.date.fromisoformat(trade_date)
        ).isoformat()
        topk_rows = conn.execute(
            "SELECT symbol FROM signals WHERE trade_date = ?",
            (prev_date,),
        ).fetchall()
        if not topk_rows:
            topk_rows = conn.execute(
                "SELECT symbol FROM signals WHERE trade_date = "
                "(SELECT MAX(trade_date) FROM signals "
                "WHERE trade_date <= ?)",
                (prev_date,),
            ).fetchall()
        topk_symbols = {r["symbol"] for r in topk_rows}

        # 8b. pending orders
        pending_orders = [
            dict(r) for r in conn.execute(
                "SELECT id, symbol, side, target_qty, carry_day "
                "FROM orders "
                "WHERE status IN ('pending','carry') "
                "AND trade_date <= ?",
                (trade_date,),
            ).fetchall()
        ]
        all_orders = list(pending_orders)

        # 8c2. carry cooldown cancel (two-pass, no mutation during iteration)
        orders_to_cancel = [
            o for o in all_orders
            if o["side"] == "buy"
            and o["carry_day"] > 0
            and o["symbol"] in active_cooldowns
        ]
        cancel_ids: set[int] = set()
        for o in orders_to_cancel:
            log_settle_change(conn, trade_date, o["id"])
            update_order(conn, o["id"], status="cancelled")
            cancel_ids.add(o["id"])
        all_orders = [o for o in all_orders if o["id"] not in cancel_ids]

        # 8d. settle
        settle_result = settle_day(
            conn, trade_date, all_orders, prices,
            current_positions, cash, topk_symbols, benchmarks, paper_cfg,
        )

        # 8e. carry day bump (non-suspended only)
        bump_carry_days(
            conn,
            [o["order_id"] for o in settle_result.carries_to_bump],
        )

        # 8f. reload post-settle state
        current_positions = get_latest_positions(conn)
        cash = settle_result.cash

    # -- Step 9: risk checks (after settle, uses post-settle NAV) --
    # 9a. compute NAV
    total_nav = compute_nav(current_positions, prices, cash)

    # 9b. nav history + yesterday_nav
    nav_rows = conn.execute(
        "SELECT trade_date, total_nav FROM nav ORDER BY trade_date ASC"
    ).fetchall()
    nav_history = [dict(r) for r in nav_rows]
    if len(nav_history) >= 2:
        yesterday_nav = float(nav_history[-2]["total_nav"])
    else:
        yesterday_nav = total_nav

    # 9c. run all risk checks
    risk_result = run_all_risk_checks(
        nav_history, yesterday_nav, current_positions, prices,
        csi1000_closes_11d, industry_map, cooldown_state, cash,
        is_soft_reduced, risk_cfg, trade_date,
    )

    # 9e. update is_soft_reduced
    is_soft_reduced = risk_result.topk_override is not None
    conn.execute(
        "INSERT OR REPLACE INTO paper_state VALUES "
        "('is_soft_reduced', ?)",
        ("true" if is_soft_reduced else "false",),
    )

    # 9f. persist new cooldown entries
    for symbol, entry in risk_result.cooldown_entries.items():
        set_cooldown(conn, symbol, entry["cooldown_until"], entry["holding_high"])

    # -- Step 10: signal generation (skip for settle-only) --
    if steps != {"settle"}:
        # 10a. resolve prediction file
        if pred_path is None:
            pred_path = PREDICTIONS_DIR / f"{trade_date}.parquet"
            if not pred_path.exists():
                logger.error(
                    "Prediction file not found: %s "
                    "(run the producer or sync predictions/)",
                    pred_path,
                )
                record_run(conn, trade_date, "error")
                try:
                    from ashare_lab.paper.ledger import insert_pipeline_run  # noqa: PLC0415
                    insert_pipeline_run(conn, trade_date, "error",
                                       time.monotonic() - start_time,
                                       "prediction file not found",
                                       predictions_date_str)
                except Exception:
                    logger.warning("Failed to record pipeline_run", exc_info=True)
                conn.commit()
                return 2

        # Extract predictions_date from filename (e.g. "2026-01-15.parquet")
        m = re.search(r"(\d{4}-\d{2}-\d{2})", pred_path.name)
        if m:
            predictions_date_str = m.group(1)

        try:
            signals_raw = generate_signals(
                trade_date, conn=conn, pred_path=pred_path,
                topk=paper_cfg["topk"],
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(
                "Signal generation failed for %s: %s", trade_date, e
            )
            record_run(conn, trade_date, "error")
            try:
                from ashare_lab.paper.ledger import insert_pipeline_run  # noqa: PLC0415
                insert_pipeline_run(conn, trade_date, "error",
                                   time.monotonic() - start_time,
                                   str(e), predictions_date_str)
            except Exception:
                logger.warning("Failed to record pipeline_run", exc_info=True)
            conn.commit()
            return 2

        # Provenance log
        meta_path = PREDICTIONS_DIR / f"{trade_date}.meta.json"
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
                logger.info(
                    "Prediction provenance: model=%s window=%s",
                    meta.get("model"), meta.get("window_id"),
                )
            except Exception:
                pass

        # 10b. filter candidates
        candidate_syms = [s for s in signals_raw if s in market_data]
        filtered_syms = filter_candidates(
            candidate_syms, market_data,
            paper_cfg["listing_min_days"],
            paper_cfg["liquidity_min_turnover"],
            config.get("universe", {}).get(
                "exclude_close_above_cny", 300.0
            ),
        )
        filtered_signals = {s: signals_raw[s] for s in filtered_syms}

        # 10d. effective topk
        effective_topk = (
            risk_result.topk_override
            if risk_result.topk_override
            else paper_cfg["topk"]
        )

        # 10e. IPO-held exclusion + topk_dropout
        ipo_held = {
            s for s in current_positions
            if market_data.get(s, {}).get("listing_days", 999)
            < paper_cfg.get("listing_min_days", 60)
        }
        held_set = set(current_positions.keys()) - ipo_held
        topk_for_dropout = max(0, effective_topk - len(ipo_held))
        sell_syms, buy_syms = topk_dropout_orders(
            filtered_signals, held_set, topk_for_dropout,
            paper_cfg.get("n_drop", 1),
        )

        # 10e2. risk blocks on BUY side only
        buy_syms = [
            s for s in buy_syms
            if industry_map.get(s) not in risk_result.blocked_industries
            and s not in risk_result.blocked_rebuys
        ]

        # 10e3. sentiment veto (fail-open: errors skip veto, not block buys)
        if config.get("paper", {}).get("sentiment", {}).get("enabled", False):
            try:
                from ashare_lab.paper.sentiment import run_sentiment_veto
                veto_result = run_sentiment_veto(
                    buy_syms, trade_date, industry_map, conn, config,
                )
                buy_syms = [
                    s for s in buy_syms
                    if s not in veto_result.vetoed_stocks
                ]
                if veto_result.global_halted:
                    risk_result = replace(risk_result, buying_halted=True)
                logger.info(
                    "sentiment veto: %d stocks vetoed, %d industries, global=%s",
                    len(veto_result.vetoed_stocks),
                    len(veto_result.vetoed_industries),
                    "halted" if veto_result.global_halted else "ok",
                )
            except Exception:
                logger.warning(
                    "Sentiment veto failed, continuing without veto",
                    exc_info=True,
                )

        # 10f. T+1 sell guard
        sell_syms = [
            s for s in sell_syms
            if current_positions.get(s, {}).get("buy_date") != trade_date
        ]

        # 10g. circuit breaker halts all buys
        if risk_result.buying_halted:
            buy_syms = []

        # 10h. compute buy quantities
        if effective_topk <= 0:
            target_value = 0.0
            buy_syms = []
        else:
            target_value = (
                total_nav * config["cost_model"]["risk_degree"]
            ) / effective_topk

        # 10i. desired_sell_qty (unified rotation + forced)
        forced_sells = risk_result.forced_sells
        sell_set = set(sell_syms) | set(forced_sells.keys())
        desired_sell_qty: dict[str, int] = {}
        for s in sell_set:
            if s in sell_syms:
                # Rotation = full position exit
                desired_sell_qty[s] = current_positions.get(s, {}).get("qty", 0)
            else:
                # Forced-only (concentration/trailing stop partial)
                desired_sell_qty[s] = forced_sells[s]

        # 10j. insert sells -- quantity-aware dedup
        next_td_str = next_trading_day(
            dt.date.fromisoformat(trade_date)
        ).isoformat()
        for s in sell_set:
            already = conn.execute(
                "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
                "WHERE symbol=? AND side='sell' "
                "AND status IN ('pending','carry')",
                (s,),
            ).fetchone()[0]
            to_insert = desired_sell_qty[s] - already
            if to_insert > 0:
                # risk-forced sell: next-day enforcement (post-settle arch)
                insert_order(
                    conn, next_td_str, s, "sell", to_insert, None,
                    "pending", 0, trade_date,
                )

        # Insert buys
        for s in buy_syms:
            close_price = prices.get(s, {}).get("close")
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
            insert_order(
                conn, next_td_str, s, "buy", target_qty, None,
                "pending", 0, trade_date,
            )

    # -- Step 11: IPO processing (D-42/D-43) --
    ipo_cash_changed = False
    for ipo_row in ipo_won:
        if ipo_row["listing_date"] != trade_date:
            continue

        symbol = ipo_row["symbol"]
        won, shares = check_ipo_subscription(
            ipo_row["ceiling_lots"], ipo_row["win_rate"]
        )
        if not won:
            continue

        # 11b. create position
        new_pos = {
            "qty": shares,
            "avg_cost": ipo_row["issue_price"],
            "market_value": shares * prices.get(symbol, {}).get(
                "close", ipo_row["issue_price"]
            ),
            "buy_date": ipo_row["listing_date"],
            "holding_high": prices.get(symbol, {}).get(
                "close", ipo_row["issue_price"]
            ),
            "factor": prices.get(symbol, {}).get("factor", 1.0),
        }
        current_positions[symbol] = new_pos
        cash -= shares * ipo_row["issue_price"]
        ipo_cash_changed = True
        snapshot_positions(conn, trade_date, current_positions)

        # Determine sell date (guard: must have price data)
        if symbol not in prices:
            continue

        sell_date = determine_ipo_sell_date(
            symbol, ipo_row["listing_date"],
            [(ipo_row["listing_date"], prices[symbol]["change"])],
        )
        if sell_date is None:
            continue

        # Normalize order date
        if sell_date > trade_date:
            order_date = sell_date
        else:
            order_date = next_trading_day(
                dt.date.fromisoformat(trade_date)
            ).isoformat()

        # Quantity-aware dedup
        already = conn.execute(
            "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
            "WHERE symbol=? AND side='sell' "
            "AND status IN ('pending','carry')",
            (symbol,),
        ).fetchone()[0]
        to_insert = shares - already
        if to_insert > 0:
            insert_order(
                conn, order_date, symbol, "sell", to_insert, None,
                "pending", 0, trade_date,
            )

    # 11c. held IPO positions re-evaluation
    for sym, pos in list(current_positions.items()):
        if market_data.get(sym, {}).get("listing_days", 999) >= paper_cfg.get(
            "listing_min_days", 60
        ):
            continue
        if pos.get("buy_date", "") == trade_date:
            # Already handled in 11b above
            continue

        # Held IPO: get daily changes for determine_ipo_sell_date
        try:
            from qlib.data import D as D_ipo  # noqa: PLC0415

            chg_df = D_ipo.features(
                instruments=[sym],
                fields=["$change"],
                start_time=pos["buy_date"],
                end_time=trade_date,
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
        if sell_date is None:
            continue
        if sell_date > trade_date:
            continue

        # Sell date is today or in the past: insert sell for next day
        order_date = next_trading_day(
            dt.date.fromisoformat(trade_date)
        ).isoformat()
        already = conn.execute(
            "SELECT COALESCE(SUM(target_qty), 0) FROM orders "
            "WHERE symbol=? AND side='sell' "
            "AND status IN ('pending','carry')",
            (sym,),
        ).fetchone()[0]
        to_insert = pos["qty"] - already
        if to_insert > 0:
            insert_order(
                conn, order_date, sym, "sell", to_insert, None,
                "pending", 0, trade_date,
            )
        snapshot_positions(conn, trade_date, current_positions)

    # -- Step 11d: persist post-IPO nav if cash changed --
    if ipo_cash_changed and settle_result is not None:
        market_value = sum(
            pos["qty"] * prices.get(s, {}).get("close", pos["avg_cost"])
            for s, pos in current_positions.items()
            if pos["qty"] > 0
        )
        new_total_nav = market_value + cash
        record_nav(
            conn, trade_date, cash, market_value, new_total_nav,
            settle_result.pre_trade_nav, settle_result.post_trade_nav,
            benchmarks["csi300"], benchmarks["csi1000"],
        )

    # -- Step 12: backup and finalize --
    conn.commit()
    backup_path = PROJECT_ROOT / "backups" / f"paper_{trade_date}.db"
    hot_backup(db_path, backup_path)
    cleanup_old_backups(
        PROJECT_ROOT / "backups", paper_cfg["backup_retention_days"]
    )
    record_run(conn, trade_date, "settled")
    conn.commit()

    # -- Step 13: WeChat report delivery (non-blocking) --
    if steps is None or "report" in (steps or set()):
        try:
            from ashare_lab.paper.report import generate_and_send_report
            # generate_and_send_report opens a transaction, handles its own commits
            report_rc = generate_and_send_report(trade_date, conn, config)
            if report_rc != 0:
                logger.warning("Report delivery failed for %s (rc=%d)", trade_date, report_rc)
        except Exception as exc:
            logger.warning("Report step failed for %s: %s", trade_date, exc)
        finally:
            conn.commit()  # safety commit for report status persistence

    # -- Step 14: Record pipeline run (non-blocking) --
    use_stale = os.environ.get("ASHARE_USE_STALE") == "1"
    try:
        from ashare_lab.paper.ledger import insert_pipeline_run  # noqa: PLC0415
        duration_s = time.monotonic() - start_time
        status = "stale" if use_stale else "success"
        insert_pipeline_run(conn, trade_date, status, duration_s,
                            None, predictions_date_str)
    except Exception:
        logger.warning("Failed to record pipeline_run", exc_info=True)
    finally:
        conn.commit()

    # -- Step 15: Graduation check (non-blocking, skip when stale) --
    if not use_stale:
        try:
            from ashare_lab.paper.graduation import (  # noqa: PLC0415
                check_graduation,
                notify_graduation,
            )
            passed, stats = check_graduation(conn)
            logger.info(
                "Graduation gate: passed=%s rate=%.1f%% days_remaining=%d",
                passed, stats["rate"] * 100, stats["days_remaining"],
            )
            if passed and stats.get("should_notify"):
                notify_graduation(conn, stats)
        except Exception:
            logger.warning("Graduation check failed", exc_info=True)

    logger.info("Pipeline completed for %s", trade_date)
    return 0


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
        rc = run_daily(date_str, force=False, pred_path=pred_path)
        if rc == 2:
            return 2
        if rc == 1:
            had_skip = True
        logger.info("Backfill %s: rc=%d", date_str, rc)

    return 1 if had_skip else 0


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
