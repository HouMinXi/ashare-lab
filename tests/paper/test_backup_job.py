"""The nightly snapshot must be restorable and must not clobber the pipeline's.

Three defects in the script this replaces, all measured on 2026-10-27:

It copied a hardcoded data/paper.db, a 143360-byte file last modified in
July, while the real ledger sat elsewhere. Every nightly log line said
"Backed up ... (140K)" and exited 0.

It used `cp`, which cannot capture a WAL-mode ledger: anything in the
write-ahead log is simply absent from the copy.

It wrote backups/paper_DATE.db, the same name the pipeline's own final step
writes. Running at 11:00 against the pipeline's 06:00 meant the later write
won, so on an ordinary trading day the real snapshot was replaced by the
stale copy. backups/paper_2026-10-24.db and _25 were both 143360 bytes;
the only real snapshots on disk belonged to days whose pipeline ran after
11:00.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper import backup_job

TODAY = dt.date(2026, 10, 27)


def _wal_ledger(path: Path, rows: list[str]) -> sqlite3.Connection:
    """A ledger in WAL mode with rows not yet checkpointed.

    The connection stays open: closing it checkpoints the WAL into the main
    file, which is exactly the state that would hide the defect.
    """
    conn = sqlite3.connect(path)
    conn.execute("pragma journal_mode=WAL")
    conn.execute("create table nav (trade_date text, total_nav real)")
    conn.commit()
    for d in rows:
        conn.execute("insert into nav values (?, ?)", (d, 100000.0))
    conn.commit()
    return conn


def _nav_dates(db: Path) -> set[str] | None:
    """Dates in nav, or None when the table is not even there.

    A byte copy of a WAL-mode ledger can come out with no schema at all:
    in WAL mode the create-table commit lands in the log, not in the main
    file, so copying the main file alone yields an empty database. None is
    the strongest form of the defect, not an error in the test.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("select trade_date from nav")}
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def test_the_job_captures_the_wal_not_just_the_main_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Through main(), because main() is where the mechanism is chosen.

    An earlier version of this test called hot_backup_with_integrity
    directly. Replacing the call inside main() with shutil.copy2 -- the
    defect as shipped -- left it green, because it never exercised the call
    site being fixed.
    """
    ledger = tmp_path / "paper.db"
    rows = ["2026-10-20", "2026-10-21", "2026-10-26"]
    conn = _wal_ledger(ledger, rows)
    try:
        assert (tmp_path / "paper.db-wal").exists(), "no WAL to test against"
        monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
        monkeypatch.setattr(
            backup_job,
            "load_config",
            lambda: {"paper": {"backup_retention_days": 7}},
        )
        assert backup_job.main(["--date", TODAY.isoformat()]) == 0
    finally:
        conn.close()

    snapshot = (
        tmp_path / "backups" / "nightly" / f"paper_{TODAY.isoformat()}.db"
    )
    assert _nav_dates(snapshot) == set(rows), (
        f"the job's snapshot is missing WAL content: {_nav_dates(snapshot)}"
    )


def test_a_byte_copy_would_lose_the_wal(tmp_path: Path) -> None:
    """The defect on its own, so the test above cannot pass by luck.

    If a byte copy ever did capture the WAL, this fails and says so rather
    than leaving the test above looking like it proved something.
    """
    ledger = tmp_path / "paper.db"
    rows = ["2026-10-20", "2026-10-21", "2026-10-26"]
    conn = _wal_ledger(ledger, rows)
    try:
        naive = tmp_path / "naive.db"
        shutil.copy2(ledger, naive)
    finally:
        conn.close()

    naive_dates = _nav_dates(naive)
    assert naive_dates != set(rows), (
        "a byte copy captured the WAL, so the test above no longer "
        "demonstrates the defect it was written for"
    )
    # measured: the copy comes out with no schema at all, because the
    # create-table commit was in the WAL too
    assert naive_dates is None or naive_dates < set(rows), (
        f"unexpected byte-copy state: {naive_dates}"
    )


def test_a_failed_integrity_check_leaves_nothing_behind(
    tmp_path: Path, monkeypatch
) -> None:
    """A snapshot that failed its check must not sit there looking valid."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )

    class _Failed:
        success = False

    monkeypatch.setattr(
        backup_job, "hot_backup_with_integrity", lambda *a, **k: _Failed()
    )

    assert backup_job.main(["--date", TODAY.isoformat()]) == 1, (
        "a failed integrity check was reported as success"
    )
    snapshot = (
        tmp_path / "backups" / "nightly" / f"paper_{TODAY.isoformat()}.db"
    )
    assert not snapshot.exists(), (
        "a failed integrity check still left a snapshot file behind"
    )


def test_the_snapshot_does_not_use_the_pipelines_filename(
    tmp_path: Path, monkeypatch
) -> None:
    """The collision that cost the real snapshots."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)

    pipeline_backup = tmp_path / "backups" / f"paper_{TODAY.isoformat()}.db"
    pipeline_backup.parent.mkdir(parents=True)
    pipeline_backup.write_bytes(b"the pipeline's own snapshot")

    assert backup_job.main(["--date", TODAY.isoformat()]) == 0

    assert pipeline_backup.read_bytes() == b"the pipeline's own snapshot", (
        "the nightly snapshot overwrote the pipeline's"
    )
    assert (
        tmp_path / "backups" / "nightly" / f"paper_{TODAY.isoformat()}.db"
    ).exists()


def test_it_snapshots_the_configured_ledger(tmp_path: Path, monkeypatch) -> None:
    """Not a hardcoded path that can go stale without anyone noticing."""
    ledger = tmp_path / "elsewhere" / "paper.db"
    ledger.parent.mkdir()
    _wal_ledger(ledger, ["2026-10-26"]).close()

    cfg = {"paper": {"db_path": str(ledger), "backup_retention_days": 7}}
    monkeypatch.setattr(backup_job, "load_config", lambda: cfg)
    monkeypatch.setattr(backup_job, "PROJECT_ROOT", tmp_path / "project")

    assert backup_job.main(["--date", TODAY.isoformat()]) == 0
    snapshot = (
        ledger.parent / "backups" / "nightly" / f"paper_{TODAY.isoformat()}.db"
    )
    assert snapshot.exists(), f"nothing at {snapshot}"
    assert _nav_dates(snapshot) == {"2026-10-26"}


def test_a_missing_ledger_is_an_error_not_a_quiet_success(
    tmp_path: Path, monkeypatch
) -> None:
    """The old script reported success for three months against a dead file."""
    monkeypatch.setattr(
        backup_job, "configured_ledger", lambda: tmp_path / "gone.db"
    )
    assert backup_job.main(["--date", TODAY.isoformat()]) == 1


def test_pruning_goes_by_the_date_in_the_name(tmp_path: Path) -> None:
    """A restored snapshot gets a fresh mtime and would outlive retention."""
    d = tmp_path / "nightly"
    d.mkdir()
    old = d / "paper_2026-10-01.db"
    recent = d / "paper_2026-10-25.db"
    for f in (old, recent):
        f.write_bytes(b"x")

    removed = backup_job.prune(d, keep_days=7, today=TODAY)

    assert old in removed and not old.exists()
    assert recent.exists() and recent not in removed


def test_pruning_leaves_names_it_does_not_recognise(tmp_path: Path) -> None:
    """It prunes what it wrote, not whatever else is in the directory."""
    d = tmp_path / "nightly"
    d.mkdir()
    stranger = d / "paper_golden_2026.db"
    empty_stem = d / "paper_.db"
    for f in (stranger, empty_stem):
        f.write_bytes(b"x")

    backup_job.prune(d, keep_days=7, today=TODAY)

    for f in (stranger, empty_stem):
        assert f.exists(), f"pruning deleted {f.name}, which it did not name"


def test_an_invalid_date_is_a_clean_cli_error(
    tmp_path: Path, monkeypatch
) -> None:
    """--date banana exits 2 with usage, not a ValueError traceback."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)

    with pytest.raises(SystemExit) as excinfo:
        backup_job.main(["--date", "banana"])
    assert excinfo.value.code == 2, (
        f"expected argparse's exit 2, got {excinfo.value.code}"
    )


def test_keep_days_overrides_the_configured_retention(
    tmp_path: Path, monkeypatch
) -> None:
    """The CLI flag, not the config value, decides what gets pruned."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )

    nightly = tmp_path / "backups" / "nightly"
    nightly.mkdir(parents=True)
    old = nightly / "paper_2026-10-01.db"
    old.write_bytes(b"older than the configured week")

    assert backup_job.main(
        ["--date", TODAY.isoformat(), "--keep-days", "60"]
    ) == 0
    assert old.exists(), "--keep-days 60 pruned a file inside its window"

    assert backup_job.main(
        ["--date", TODAY.isoformat(), "--keep-days", "7"]
    ) == 0
    assert not old.exists(), "--keep-days 7 kept a file outside its window"


def test_a_config_without_db_path_fails_with_the_setting_name(
    tmp_path: Path, monkeypatch
) -> None:
    """Not a bare KeyError from the nightly service."""
    monkeypatch.setattr(backup_job, "load_config", lambda: {})
    with pytest.raises(SystemExit) as excinfo:
        backup_job.main(["--date", TODAY.isoformat()])
    assert "paper.db_path" in str(excinfo.value), (
        f"the error does not name the missing setting: {excinfo.value}"
    )


def test_snapshot_date_follows_the_exchange_timezone() -> None:
    """Same UTC instant, different host timezone, same intended label."""
    before_close = dt.datetime(2026, 10, 27, 15, 30, tzinfo=dt.timezone.utc)
    after_midnight = dt.datetime(2026, 10, 27, 16, 30, tzinfo=dt.timezone.utc)
    # 15:30 UTC is 23:30 in +08:00, still the same trading day
    assert backup_job.exchange_today(before_close) == TODAY
    # 16:30 UTC is 00:30 the next day in +08:00
    assert backup_job.exchange_today(after_midnight) == TODAY + dt.timedelta(days=1)


def test_the_default_label_comes_from_exchange_today(
    tmp_path: Path, monkeypatch
) -> None:
    """main() must use the helper, not re-derive the date itself."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )
    # deliberately not today: a main() that computed its own current date
    # would pass this test if the sentinel were simply today's date
    sentinel = dt.date(2001, 2, 3)
    monkeypatch.setattr(backup_job, "exchange_today", lambda: sentinel)

    assert backup_job.main([]) == 0
    expected = tmp_path / "backups" / "nightly" / f"paper_{sentinel}.db"
    assert expected.exists(), (
        f"no snapshot at {expected}; main() did not use exchange_today"
    )


def test_an_unwritable_backup_directory_is_a_clean_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A PermissionError from mkdir must not escape as a traceback."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )

    def _read_only(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", _read_only)

    assert backup_job.main(["--date", TODAY.isoformat()]) == 1, (
        "a failed mkdir was not reported as an error exit"
    )


def test_a_prune_permission_failure_does_not_fail_the_job(
    tmp_path: Path, monkeypatch
) -> None:
    """The new snapshot still counts as written when an old file will not go."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )

    nightly = tmp_path / "backups" / "nightly"
    nightly.mkdir(parents=True)
    stale = nightly / "paper_2026-10-01.db"
    stale.write_bytes(b"old")

    def _deny(self, *args, **kwargs):
        raise PermissionError("cannot remove")

    monkeypatch.setattr(Path, "unlink", _deny)

    assert backup_job.main(["--date", TODAY.isoformat()]) == 0, (
        "a prune permission error failed the job after the snapshot was written"
    )
    assert (
        nightly / f"paper_{TODAY.isoformat()}.db"
    ).exists(), "the new snapshot is missing"
    assert stale.exists(), "the file that could not be removed is gone"


def test_a_file_already_gone_during_pruning_is_not_an_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Losing the race with another cleaner is a no-op, not a failure."""
    ledger = tmp_path / "paper.db"
    _wal_ledger(ledger, ["2026-10-26"]).close()
    monkeypatch.setattr(backup_job, "configured_ledger", lambda: ledger)
    monkeypatch.setattr(
        backup_job,
        "load_config",
        lambda: {"paper": {"backup_retention_days": 7}},
    )

    nightly = tmp_path / "backups" / "nightly"
    nightly.mkdir(parents=True)
    (nightly / "paper_2026-10-01.db").write_bytes(b"old")

    def _already_gone(self, *args, **kwargs):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(Path, "unlink", _already_gone)

    assert backup_job.main(["--date", TODAY.isoformat()]) == 0
    assert (nightly / f"paper_{TODAY.isoformat()}.db").exists()
