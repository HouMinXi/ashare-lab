"""Nightly ledger snapshot, independent of the daily pipeline.

Why this exists separately from the pipeline's own backup: the pipeline
snapshots in its final step, so a run that fails earlier leaves no backup
at all. On 2026-10-21 the pipeline left no settled marker and no snapshot,
and recovering the book later depended on one existing.

Why it is not a `cp`: the ledger runs in WAL mode, so a byte copy of the
main file can miss everything sitting in the write-ahead log. The library's
hot_backup_with_integrity runs an integrity check, uses SQLite's online
backup API, and verifies the result by digest.

Why it writes to its own directory: the previous script wrote
backups/paper_DATE.db, the same name the pipeline uses. The pipeline ran at
06:00 and this at 11:00, so on every ordinary trading day the later write
replaced a real snapshot with whatever it had copied. Measured on
2026-10-27: backups/paper_2026-10-24.db and paper_2026-10-25.db were both
143360-byte copies of a file last modified in July, and the only real
snapshots on disk belonged to days whose pipeline happened to run after
11:00.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from ashare_lab.config import PROJECT_ROOT, load_config
from ashare_lab.paper.ledger import hot_backup_with_integrity

logger = logging.getLogger(__name__)

from ashare_lab.data.calendar import market_today

SUBDIR = "nightly"
def exchange_today(now: dt.datetime | None = None) -> dt.date:
    """The trading-day label for a snapshot.

    The calendar owns the calculation; this name keeps the backup job's callers
    reading naturally, and ``now`` passes through for the boundary tests.
    """
    return market_today(now)


def configured_ledger() -> Path:
    """The ledger this project is actually configured to use.

    A missing setting is an operator-facing error naming the setting,
    rather than a KeyError from inside a nightly service.
    """
    db_path = load_config().get("paper", {}).get("db_path")
    if not db_path:
        raise SystemExit("paper.db_path is not set in configs/baseline.yaml")
    return PROJECT_ROOT / db_path


def prune(directory: Path, keep_days: int, today: dt.date) -> list[Path]:
    """Delete snapshots older than keep_days, by the date in the name.

    By name rather than by mtime: a snapshot restored or copied around gets
    a fresh mtime and would otherwise outlive its retention silently. Files
    whose name does not parse are left alone -- this prunes what it wrote,
    not whatever else is in the directory.
    """
    removed = []
    cutoff = today - dt.timedelta(days=keep_days)
    for path in sorted(directory.glob("paper_*.db")):
        try:
            stamp = dt.date.fromisoformat(path.stem[len("paper_"):])
        except ValueError:
            continue
        if stamp < cutoff:
            try:
                path.unlink()
            except FileNotFoundError:
                # lost the race with another cleaner; nothing left to do
                continue
            except OSError as exc:
                # the snapshot is already written; a prune failure must not
                # make the whole job look like a failed backup
                logger.warning("could not prune %s: %s", path, exc)
                continue
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=None,
        help="snapshot label, defaults to today in the local timezone",
    )
    parser.add_argument("--keep-days", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    ledger = configured_ledger()
    if not ledger.exists():
        logger.error("no ledger at %s; nothing to snapshot", ledger)
        return 1

    try:
        today = (
            dt.date.fromisoformat(args.date)
            if args.date
            else exchange_today()
        )
    except ValueError:
        # a CLI mistake should print usage, not a traceback
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    keep_days = (
        args.keep_days
        if args.keep_days is not None
        else load_config().get("paper", {}).get("backup_retention_days", 7)
    )

    # beside the ledger, in our own subdirectory: the pipeline owns
    # backups/paper_DATE.db and we must not be the last writer of its name
    target_dir = ledger.parent / "backups" / SUBDIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # a read-only or unwritable destination is an operator problem to
        # report, not a traceback from inside a nightly service
        logger.error("cannot create snapshot directory %s: %s", target_dir, exc)
        return 1
    target = target_dir / f"paper_{today.isoformat()}.db"

    result = hot_backup_with_integrity(ledger, target)
    if not result.success:
        logger.error(
            "snapshot of %s failed its integrity check; no file written",
            ledger,
        )
        return 1

    logger.info(
        "snapshot of %s written to %s (%d bytes)",
        ledger,
        target,
        target.stat().st_size,
    )

    for gone in prune(target_dir, keep_days, today):
        logger.info("pruned %s", gone.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
