"""CLI entry point for ashare-lab data pipeline."""

from __future__ import annotations

# Global socket timeout: prevents qlib client-mode from hanging indefinitely
# when no qlib server is running. Must be set BEFORE any qlib import.
import socket as _socket
_socket.setdefaulttimeout(30)

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from ashare_lab.config import PROJECT_ROOT
from ashare_lab.data.fallback import gap_fill, _write_csvs, _dump_bin_update
from ashare_lab.data.update import bootstrap, daily_refresh, DEFAULT_PROVIDER_URI
from ashare_lab.data.validate import (
    check_return_consistency,
    spot_check_raw,
    validate_instruments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    try:
        bootstrap()
        return 0
    except Exception as exc:
        log.error("bootstrap failed: %s", exc)
        return 2


def cmd_update(args: argparse.Namespace) -> int:
    try:
        return daily_refresh()
    except Exception as exc:
        log.error("update failed: %s", exc)
        return 2


def cmd_fetch_today(args: argparse.Namespace) -> int:
    import datetime as dt  # noqa: PLC0415
    from ashare_lab.data.calendar import is_trading_day, latest_trading_day  # noqa: PLC0415
    from ashare_lab.data.fetcher import (  # noqa: PLC0415
        fetch_today_data,
        fetch_cross_validation_sample,
        refresh_stock_names_cache,
        CSI1000_SAMPLE_SYMBOLS,
    )
    from ashare_lab.data.validator import validate_daily_data  # noqa: PLC0415

    today = dt.date.today()
    if not is_trading_day(today):
        print("not a trading day")
        return 0

    trade_date = str(latest_trading_day(today))

    try:
        tushare_data = fetch_today_data(trade_date)
    except RuntimeError as exc:
        log.error("fetch_today_data failed: %s", exc)
        return 1

    cross_val_data: dict = {}
    try:
        cross_val_data = fetch_cross_validation_sample(
            trade_date, CSI1000_SAMPLE_SYMBOLS
        )
    except Exception as exc:
        log.warning("cross-validation fetch failed (non-fatal): %s", exc)

    names_cache = PROJECT_ROOT / "data" / "stock_names_cache.csv"
    stock_names: dict[str, str] | None = None
    try:
        stock_names = refresh_stock_names_cache(names_cache)
    except Exception as exc:
        log.warning("stock name cache refresh failed (non-fatal): %s", exc)

    result = validate_daily_data(
        tushare_data, trade_date,
        cross_val_data=cross_val_data or None,
        stock_names=stock_names,
    )
    if not result.passed:
        for w in result.warnings:
            log.warning("validation: %s", w)

    try:
        with tempfile.TemporaryDirectory() as tmp_csv_dir:
            _write_csvs(tushare_data, Path(tmp_csv_dir))
            _dump_bin_update(Path(tmp_csv_dir), DEFAULT_PROVIDER_URI)
    except Exception as exc:
        log.error("write/dump failed: %s", exc)
        return 1

    log.info("fetch-today complete for %s (%d symbols)", trade_date, len(tushare_data))
    return 0


def cmd_chenditc_snapshot(args: argparse.Namespace) -> int:
    import datetime as dt  # noqa: PLC0415
    import json as _json  # noqa: PLC0415
    from ashare_lab.data.calendar import is_trading_day, latest_trading_day  # noqa: PLC0415
    from ashare_lab.data.fetcher import CSI1000_SAMPLE_SYMBOLS  # noqa: PLC0415

    today = dt.date.today()
    if not is_trading_day(today):
        print("not a trading day")
        return 0

    trade_date = str(latest_trading_day(today))

    try:
        import qlib  # noqa: PLC0415
        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
        from qlib.data import D  # noqa: PLC0415
        df = D.features(
            instruments=CSI1000_SAMPLE_SYMBOLS,
            fields=["$close"],
            start_time=trade_date,
            end_time=trade_date,
        )
        snapshot = {}
        if df is not None and not df.empty:
            for instrument, row in df.iterrows():
                sym = instrument[0] if isinstance(instrument, tuple) else instrument
                snapshot[str(sym)] = float(row["$close"])
        else:
            log.warning("No 17:00 incremental for %s; diff will be skipped", trade_date)
    except Exception as exc:
        log.warning("qlib query failed (non-fatal): %s", exc)
        snapshot = {}

    out = Path(f"/tmp/incremental_snapshot_{trade_date}.json")
    out.write_text(_json.dumps(snapshot))
    log.info("snapshot saved: %s (%d symbols)", out, len(snapshot))
    return 0


def cmd_chenditc_diff(args: argparse.Namespace) -> int:
    import datetime as dt  # noqa: PLC0415
    import json as _json  # noqa: PLC0415
    from ashare_lab.data.calendar import latest_trading_day  # noqa: PLC0415

    trade_date = str(latest_trading_day(dt.date.today()))
    snap_path = Path(f"/tmp/incremental_snapshot_{trade_date}.json")

    if not snap_path.exists():
        log.info("No snapshot; diff skipped")
        return 0

    snapshot = _json.loads(snap_path.read_text())
    if not snapshot:
        log.info("No snapshot; diff skipped")
        return 0

    try:
        import qlib  # noqa: PLC0415
        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
        from qlib.data import D  # noqa: PLC0415
        symbols = list(snapshot.keys())
        df = D.features(
            instruments=symbols,
            fields=["$close"],
            start_time=trade_date,
            end_time=trade_date,
        )
    except Exception as exc:
        log.error("qlib query failed in diff: %s", exc)
        return 0

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    diff_path = log_dir / f"chenditc_diff_{trade_date}.jsonl"

    discrepancies = 0
    with open(diff_path, "w") as f:
        if df is not None and not df.empty:
            for instrument, row in df.iterrows():
                sym = str(instrument[0] if isinstance(instrument, tuple) else instrument)
                chenditc_close = float(row["$close"])
                incr_close = snapshot.get(sym)
                if incr_close is not None and abs(chenditc_close - incr_close) > 0.001:
                    entry = {
                        "symbol": sym,
                        "incremental_close": incr_close,
                        "chenditc_close": chenditc_close,
                        "diff": round(chenditc_close - incr_close, 4),
                    }
                    f.write(_json.dumps(entry) + "\n")
                    log.warning("diff: %s incr=%.4f chenditc=%.4f", sym, incr_close, chenditc_close)
                    discrepancies += 1

    log.info("chenditc diff complete: %d discrepancies logged to %s", discrepancies, diff_path)
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    try:
        rc = gap_fill(args.from_date, args.to_date)
        if rc == 1:
            log.info("backfill: no gap detected")
        elif rc == 2:
            log.warning("backfill: partial fill -- some dates still missing")
        return rc
    except Exception as exc:
        log.error("backfill failed: %s", exc)
        return 2


def cmd_validate(args: argparse.Namespace) -> int:
    raw = spot_check_raw()
    ret = check_return_consistency()
    inst = validate_instruments()
    print(json.dumps({"raw_check": raw, "return_check": ret, "instruments": inst}, indent=2))
    failed = any(v.get("status") == "FAIL" for v in raw.values())
    if raw and all(v.get("status") == "SKIP" for v in raw.values()):
        log.warning("all symbols skipped in raw check; data sources unreachable")
        return 1
    if inst.get("status") == "FAIL":
        failed = True
    return 2 if failed else 0


def cmd_paper_settle(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415
        return run_daily(
            trade_date=getattr(args, "date", None),
            force=getattr(args, "force", False),
            steps={"settle"},
        )
    except Exception as exc:
        log.error("paper settle failed: %s", exc)
        return 2


def cmd_paper_signal(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415
        return run_daily(
            trade_date=getattr(args, "date", None),
            force=False,
            steps={"signal"},
        )
    except Exception as exc:
        log.error("paper signal failed: %s", exc)
        return 2


def cmd_paper_run_all(args: argparse.Namespace) -> int:
    try:
        from pathlib import Path  # noqa: PLC0415
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415
        pred = getattr(args, "pred_path", None)
        return run_daily(
            trade_date=getattr(args, "date", None),
            force=getattr(args, "force", False),
            pred_path=Path(pred) if pred else None,
        )
    except Exception as exc:
        log.error("paper run-all failed: %s", exc)
        return 2


def cmd_paper_backfill(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.pipeline import run_backfill  # noqa: PLC0415
        return run_backfill(
            args.from_date,
            args.to_date,
            force=getattr(args, "force", False),
        )
    except Exception as exc:
        log.error("paper backfill failed: %s", exc)
        return 2


def cmd_paper_status(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.pipeline import get_status  # noqa: PLC0415
        status = get_status()
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        log.error("paper status failed: %s", exc)
        return 2


def cmd_paper_report(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.report import generate_and_send_report  # noqa: PLC0415
        from ashare_lab.paper.ledger import get_connection
        from ashare_lab.config import load_config, PROJECT_ROOT
        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]
        conn = get_connection(db_path)
        
        trade_date = getattr(args, "date", None)
        if trade_date is None:
            latest_settled = conn.execute(
                "SELECT MAX(trade_date) FROM runs WHERE status='settled'"
            ).fetchone()[0]
            if not latest_settled:
                print("No settled runs found")
                return 1
            trade_date = latest_settled

        rc = generate_and_send_report(
            trade_date, conn, config,
            dry_run=getattr(args, "dry_run", False),
        )
        conn.commit()
        return rc
    except Exception as exc:
        log.error("paper report failed: %s", exc)
        return 2


def cmd_paper_sentiment(args: argparse.Namespace) -> int:
    try:
        from ashare_lab.paper.sentiment import run_sentiment_veto  # noqa: PLC0415
        from ashare_lab.paper.ledger import get_connection, init_schema
        from ashare_lab.paper.pipeline import _load_industry_cache
        from ashare_lab.config import load_config, PROJECT_ROOT

        config = load_config()
        db_path = PROJECT_ROOT / config["paper"]["db_path"]
        conn = get_connection(db_path)
        init_schema(conn)

        trade_date = getattr(args, "date", None)
        if trade_date is None:
            latest_settled = conn.execute(
                "SELECT MAX(trade_date) FROM runs WHERE status='settled'"
            ).fetchone()[0]
            if not latest_settled:
                print("No settled runs found")
                return 1
            trade_date = latest_settled

        symbols = getattr(args, "symbol", None)
        if symbols:
            buy_syms = list(symbols)
        else:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM orders "
                "WHERE side='buy' AND status IN ('pending','carry') "
                "AND created_run_date=?",
                (trade_date,),
            ).fetchall()
            buy_syms = [r[0] for r in rows]

        all_syms = set(buy_syms)
        industry_map = _load_industry_cache(trade_date, all_syms) or {}

        result = run_sentiment_veto(
            buy_syms, trade_date, industry_map, conn, config,
        )

        # dry-run: rollback cache writes and event logs;
        # normal: commit so writes persist
        if getattr(args, "dry_run", False):
            conn.rollback()
        else:
            conn.commit()

        print(f"trade_date: {trade_date}")
        print(f"buy_syms checked: {len(buy_syms)}")
        print(f"vetoed stocks: {result.vetoed_stocks}")
        print(f"vetoed industries: {result.vetoed_industries}")
        print(f"global halted: {result.global_halted}")
        if result.global_score is not None:
            print(f"global score: {result.global_score}")
        return 0
    except Exception as exc:
        log.error("paper sentiment failed: %s", exc)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="ashare-lab", description="A-share data pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bootstrap", help="first-time data download")
    sub.add_parser("update", help="daily data refresh")
    sub.add_parser("fetch-today", help="tushare same-day fetch + qlib dump_update")
    sub.add_parser("validate", help="cross-source spot-check")
    sub.add_parser("chenditc-snapshot", help="save qlib close prices before chenditc refresh")
    sub.add_parser("chenditc-diff", help="compare incremental vs chenditc close prices")

    p_backfill = sub.add_parser("backfill", help="gap-fill missing dates from baostock")
    p_backfill.add_argument("from_date", help="start date YYYY-MM-DD (inclusive)")
    p_backfill.add_argument("to_date", help="end date YYYY-MM-DD (inclusive)")

    # Paper engine subcommands
    p_paper = sub.add_parser("paper", help="paper trading engine")
    paper_sub = p_paper.add_subparsers(dest="paper_command")

    p_settle = paper_sub.add_parser("settle", help="run settle phase only")
    p_settle.add_argument("--date", help="trade date YYYY-MM-DD")
    p_settle.add_argument("--force", action="store_true", help="force re-run")

    p_signal = paper_sub.add_parser("signal", help="run signal generation only")
    p_signal.add_argument("--date", help="trade date YYYY-MM-DD")

    p_run_all = paper_sub.add_parser("run-all", help="full daily pipeline")
    p_run_all.add_argument("--date", help="trade date YYYY-MM-DD")
    p_run_all.add_argument("--force", action="store_true", help="force re-run")
    p_run_all.add_argument("--pred-path", type=str, help="path to prediction parquet (stale fallback)")

    p_pbackfill = paper_sub.add_parser("backfill", help="replay missed days")
    p_pbackfill.add_argument("from_date", help="start date YYYY-MM-DD")
    p_pbackfill.add_argument("to_date", help="end date YYYY-MM-DD")
    p_pbackfill.add_argument("--force", action="store_true", help="force teardown and replay")

    paper_sub.add_parser("status", help="show current ledger state")

    p_report = paper_sub.add_parser("report", help="generate and send daily report")
    p_report.add_argument("--date", help="trade date YYYY-MM-DD (default: latest settled)")
    p_report.add_argument("--dry-run", action="store_true", help="generate only, skip delivery")

    p_sentiment = paper_sub.add_parser("sentiment", help="run sentiment veto check")
    p_sentiment.add_argument("--date", help="trade date YYYY-MM-DD (default: latest settled)")
    p_sentiment.add_argument("--symbol", action="append", help="stock code to check (repeatable)")
    p_sentiment.add_argument("--dry-run", action="store_true", help="score only, rollback DB writes")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "paper":
        pc = getattr(args, "paper_command", None)
        if not pc:
            p_paper.print_help()
            return 1
        paper_commands = {
            "settle": cmd_paper_settle,
            "signal": cmd_paper_signal,
            "run-all": cmd_paper_run_all,
            "backfill": cmd_paper_backfill,
            "status": cmd_paper_status,
            "report": cmd_paper_report,
            "sentiment": cmd_paper_sentiment,
        }
        return paper_commands[pc](args)

    commands = {
        "bootstrap": cmd_bootstrap,
        "update": cmd_update,
        "fetch-today": cmd_fetch_today,
        "validate": cmd_validate,
        "backfill": cmd_backfill,
        "chenditc-snapshot": cmd_chenditc_snapshot,
        "chenditc-diff": cmd_chenditc_diff,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
