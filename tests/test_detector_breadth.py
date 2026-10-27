"""The live-ledger detector must not be satisfied by one table.

An adversarial pass ran 22 ways of destroying a real ledger past the
guard; 21 succeeded, every one of them because the detector asked `nav`
and stopped. These reproduce the three that were measured directly.

A real ledger carries twelve dated tables and no views, so each of these
shapes is a ledger that is unmistakably live while nav alone says
otherwise.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.conftest import FIXTURE_LEDGER_DATES, ledger_holds_live_history

REAL_DATES = ["2026-08-21", "2026-09-15", "2026-10-26"]
FIXTURE_DATES = sorted(FIXTURE_LEDGER_DATES)


def _build(db: Path, *, nav: list[str], others: list[str]) -> None:
    """A ledger shaped like the real one: nav plus the other dated tables."""
    conn = sqlite3.connect(db)
    conn.execute("create table nav (trade_date text, total_nav real)")
    for table in ("positions", "trades", "orders", "signals", "runs"):
        conn.execute(f"create table {table} (trade_date text, symbol text)")
    for d in nav:
        conn.execute("insert into nav values (?, ?)", (d, 100000.0))
    for table in ("positions", "trades", "orders", "signals", "runs"):
        for d in others:
            conn.execute(
                f"insert into {table} values (?, ?)", (d, "SZ000001")
            )
    conn.commit()
    conn.close()


def test_an_emptied_nav_does_not_hide_the_rest(tmp_path: Path) -> None:
    """Measured bypass: nav truncated, every other table still real."""
    db = tmp_path / "paper.db"
    _build(db, nav=[], others=REAL_DATES)
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "nav was emptied and the guard stood down while five tables still "
        "held real history"
    )


def test_a_fixture_shaped_nav_does_not_hide_the_rest(tmp_path: Path) -> None:
    """Measured bypass: nav rewritten to fixture dates, rest untouched."""
    db = tmp_path / "paper.db"
    _build(db, nav=FIXTURE_DATES, others=REAL_DATES)
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "nav was made to look like a fixture leftover and the guard believed "
        "it"
    )


def test_nav_as_a_view_is_still_read(tmp_path: Path) -> None:
    """Measured bypass: nav replaced by a view, skipped by type='table'."""
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.execute("create table nav_real (trade_date text, total_nav real)")
    for d in REAL_DATES:
        conn.execute("insert into nav_real values (?, ?)", (d, 100000.0))
    conn.execute("create view nav as select * from nav_real")
    conn.commit()
    conn.close()
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "nav was a view over real rows and the guard skipped it"
    )


def test_a_fixture_only_ledger_is_still_not_live(tmp_path: Path) -> None:
    """The guard must stay out of the way of a genuine fixture leftover."""
    db = tmp_path / "paper.db"
    _build(db, nav=FIXTURE_DATES, others=FIXTURE_DATES)
    assert not ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "a ledger holding only the fixture's own dates was called live, "
        "which would skip the replay tests everywhere"
    )


def test_an_absent_ledger_is_not_live(tmp_path: Path) -> None:
    assert not ledger_holds_live_history(
        tmp_path / "nope.db", FIXTURE_LEDGER_DATES
    ), "a ledger that does not exist was called live"


def test_an_empty_ledger_is_not_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    sqlite3.connect(db).close()
    assert not ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "a ledger with no tables was called live"
    )


def test_a_file_that_will_not_open_counts_as_live(tmp_path: Path) -> None:
    """Unreadable at the ledger path is more likely a book than a leftover.

    A skipped test says so and is recoverable; the other way round cost the
    ledger once.
    """
    db = tmp_path / "paper.db"
    db.write_bytes(b"this is not a database" * 64)
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "a file that will not open was treated as nothing to protect"
    )


@pytest.mark.parametrize("column", ["date", "as_of_date"])
def test_other_date_column_names_are_read(tmp_path: Path, column: str) -> None:
    """Not every dated table in the ledger names its column trade_date."""
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.execute("create table nav (trade_date text, total_nav real)")
    conn.execute(f"create table extra ({column} text, note text)")
    conn.execute("insert into extra values (?, ?)", (REAL_DATES[0], "x"))
    conn.commit()
    conn.close()
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        f"a table dated by {column} was not consulted"
    )

def _quoted_name_ledger(db: Path, dates: list[str]) -> None:
    """A ledger whose only dated table has a double quote in its name."""
    conn = sqlite3.connect(db)
    conn.execute('create table "odd""name" (trade_date text)')
    for d in dates:
        conn.execute('insert into "odd""name" values (?)', (d,))
    conn.commit()
    conn.close()


def test_a_quoted_table_name_with_only_fixture_dates_is_not_live(
    tmp_path: Path,
) -> None:
    """A syntax error here used to be read as 'live' and skipped tests.

    pragma table_info("odd"name") is not valid SQL, and the detector's
    except branch answered True for anything it could not read. That is the
    safe answer for a real ledger and the wrong one for a fixture leftover.
    """
    db = tmp_path / "paper.db"
    _quoted_name_ledger(db, FIXTURE_DATES)
    assert not ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "a table name with a double quote made a fixture-only ledger look "
        "live, which would skip the replay tests for no reason"
    )


def test_a_quoted_table_name_with_real_history_is_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    _quoted_name_ledger(db, REAL_DATES)
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "real history behind an oddly named table was not detected"
    )


def test_a_question_mark_in_the_filename_does_not_force_live(
    tmp_path: Path,
) -> None:
    """A fixture-only ledger whose name contains '?' is not live.

    Interpolating the path into a URI lets '?' be read as a query string.
    The measured failure mode is that the detector reads the wrong or an
    empty target and answers not-live; this case pins the other side, so a
    correct URI does not simply turn every odd filename into "live".
    """
    db = tmp_path / "odd?name.db"
    _build(db, nav=FIXTURE_DATES, others=FIXTURE_DATES)
    assert not ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "a '?' in the filename made a fixture-only ledger look live"
    )


def test_a_question_mark_in_the_filename_with_real_history_is_live(
    tmp_path: Path,
) -> None:
    """The dangerous direction: real history must not be disarmed.

    Before as_uri(), this returned False on a real ledger whose filename
    contained '?', which would have switched the guard off entirely.
    """
    db = tmp_path / "odd?name.db"
    _build(db, nav=FIXTURE_DATES, others=REAL_DATES)
    assert ledger_holds_live_history(db, FIXTURE_LEDGER_DATES), (
        "real history behind a '?' filename was not detected"
    )
