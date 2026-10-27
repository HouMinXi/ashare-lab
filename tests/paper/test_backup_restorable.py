"""A backup has to be restorable without surgery, and belong to its ledger.

Two defects in _step12_backup_and_finalize, both found while recovering the
main book on 2026-10-26.

The backup is taken at :1926 and the day is recorded settled at :1940, so
every backup lacks its own last day's settled marker. get_latest_positions
anchors carry-forward on the newest settled run, so restoring a backup and
replaying anchors a day early and resurrects a position that day had sold.
In a rehearsal that inflated day one by 8,310 yuan; the real recovery
needed the marker inserted by hand before the replay reproduced the
recorded book.

The backup directory was anchored at PROJECT_ROOT rather than at the
ledger, so a run pointed at another ledger still wrote into -- and pruned,
via cleanup_old_backups -- the real one's backups.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper.ledger import get_connection, init_schema, record_run
from ashare_lab.paper.pipeline import (
    DailyRunContext,
    _step12_backup_and_finalize,
)

TRADE_DATE = "2026-10-22"


def _ctx(tmp_path: Path) -> DailyRunContext:
    """Minimal context for the backup step, in the style of the book-B tests."""
    db_path = tmp_path / "paper.db"
    conn = get_connection(db_path)
    init_schema(conn)
    # a settled yesterday, so the step has a predecessor like a real run
    record_run(conn, "2026-10-21", "settled")
    conn.commit()

    ctx = DailyRunContext.__new__(DailyRunContext)
    ctx.conn = conn
    ctx.db_path = db_path
    ctx.trade_date = TRADE_DATE
    ctx.paper_cfg = {"backup_retention_days": 7}
    return ctx


@pytest.fixture
def ctx(tmp_path: Path):
    context = _ctx(tmp_path)
    yield context
    context.conn.close()


def _runs_in(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {
            row[0]: row[1]
            for row in conn.execute("select trade_date, status from runs")
        }
    finally:
        conn.close()


def test_the_backup_lands_beside_its_own_ledger(
    ctx: DailyRunContext, tmp_path: Path
) -> None:
    _step12_backup_and_finalize(ctx)
    expected = tmp_path / "backups" / f"paper_{TRADE_DATE}.db"
    assert expected.exists(), (
        f"no backup at {expected}; files under tmp: "
        f"{sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))}"
    )


def test_the_backup_records_its_own_day_as_settled(
    ctx: DailyRunContext, tmp_path: Path
) -> None:
    """Restoring this backup must not anchor carry-forward a day early.

    get_latest_positions reads MAX(trade_date) FROM runs WHERE
    status='settled'. A backup missing its own day sends that anchor back
    to the previous day, whose positions still include whatever this day
    sold.
    """
    _step12_backup_and_finalize(ctx)
    backup = tmp_path / "backups" / f"paper_{TRADE_DATE}.db"
    assert backup.exists(), "the backup was not created"

    runs = _runs_in(backup)
    assert runs.get(TRADE_DATE) == "settled", (
        f"the backup does not know {TRADE_DATE} settled; it holds {runs}. "
        "Restoring it would anchor carry-forward on the previous day."
    )


def test_the_live_ledger_still_records_the_day(
    ctx: DailyRunContext, tmp_path: Path
) -> None:
    """Whatever the ordering, the ledger itself must end up settled."""
    _step12_backup_and_finalize(ctx)
    runs = _runs_in(ctx.db_path)
    assert runs.get(TRADE_DATE) == "settled", f"ledger holds {runs}"


def test_nothing_is_written_outside_the_ledger_directory(
    ctx: DailyRunContext, tmp_path: Path, monkeypatch
) -> None:
    """The step must not reach the project directory for its backups.

    Anchoring on PROJECT_ROOT meant a run on any ledger wrote into, and
    pruned, the real backups.
    """
    from ashare_lab.paper import pipeline

    sentinel = tmp_path / "not-the-ledger-dir"
    sentinel.mkdir()
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", sentinel)

    _step12_backup_and_finalize(ctx)

    strays = sorted(p.name for p in sentinel.rglob("*"))
    assert not strays, f"the step wrote outside the ledger directory: {strays}"
    assert (tmp_path / "backups" / f"paper_{TRADE_DATE}.db").exists()


def test_a_failed_backup_still_records_the_day_settled(
    ctx: DailyRunContext, tmp_path: Path, monkeypatch
) -> None:
    """Pin the contract that was already there, not a new one.

    main's failure log says "Pipeline result still recorded (pipeline did
    execute)" and the old order recorded the run after a failed backup too,
    so the ordering fix did not introduce this. It is written down here
    because a future "fix" that withholds the settled marker until a backup
    succeeds would re-create the torn-backup problem this file exists to
    prevent, one failure mode later.
    """
    from ashare_lab.paper import pipeline

    class _Failed:
        success = False

    monkeypatch.setattr(
        pipeline, "hot_backup_with_integrity", lambda *a, **k: _Failed()
    )
    monkeypatch.setattr(pipeline, "create_golden_backup", lambda *a, **k: None)

    _step12_backup_and_finalize(ctx)

    runs = _runs_in(ctx.db_path)
    assert runs.get(TRADE_DATE) == "settled", (
        f"a failed backup changed the settled contract: {runs}"
    )
    assert not (tmp_path / "backups" / f"paper_{TRADE_DATE}.db").exists(), (
        "a backup file exists despite the failed integrity check"
    )
