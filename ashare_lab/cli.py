"""CLI entry point for ashare-lab data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ashare_lab.data.fallback import gap_fill
from ashare_lab.data.update import bootstrap, daily_refresh
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
        from ashare_lab.paper.pipeline import run_daily  # noqa: PLC0415
        return run_daily(
            trade_date=getattr(args, "date", None),
            force=getattr(args, "force", False),
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
            force_detailed=getattr(args, "detailed", False)
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
    sub.add_parser("validate", help="cross-source spot-check")

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

    p_pbackfill = paper_sub.add_parser("backfill", help="replay missed days")
    p_pbackfill.add_argument("from_date", help="start date YYYY-MM-DD")
    p_pbackfill.add_argument("to_date", help="end date YYYY-MM-DD")
    p_pbackfill.add_argument("--force", action="store_true", help="force teardown and replay")

    paper_sub.add_parser("status", help="show current ledger state")

    p_report = paper_sub.add_parser("report", help="generate and send daily report")
    p_report.add_argument("--date", help="trade date YYYY-MM-DD (default: latest settled)")
    p_report.add_argument("--detailed", action="store_true", help="force detailed mode")
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
        "validate": cmd_validate,
        "backfill": cmd_backfill,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
