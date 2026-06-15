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


def main() -> int:
    parser = argparse.ArgumentParser(prog="ashare-lab", description="A-share data pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bootstrap", help="first-time data download")
    sub.add_parser("update", help="daily data refresh")
    sub.add_parser("validate", help="cross-source spot-check")

    p_backfill = sub.add_parser("backfill", help="gap-fill missing dates from baostock")
    p_backfill.add_argument("from_date", help="start date YYYY-MM-DD (inclusive)")
    p_backfill.add_argument("to_date", help="end date YYYY-MM-DD (inclusive)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "bootstrap": cmd_bootstrap,
        "update": cmd_update,
        "validate": cmd_validate,
        "backfill": cmd_backfill,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
