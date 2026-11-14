"""Guard: the suite must never delete a ledger holding real history.

TestRealPredictionReplay calls run_daily(), which resolves its own database
from configs/baseline.yaml relative to PROJECT_ROOT and takes no path
argument, so the test deletes that file to start clean. In a checkout that
carries models/w1.pt and real predictions -- a production checkout -- that
file is the live paper ledger, and running the suite destroyed months of
history once.

The guard refuses the delete and skips instead. It is autouse, so a test
written later that never heard of it is covered too.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import (
    FIXTURE_LEDGER_DATES,
    configured_ledger_path,
    ledger_holds_live_history,
)


def _make_ledger(path: Path, dates: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("create table nav (trade_date text primary key, total_nav real)")
    for d in dates:
        conn.execute("insert into nav values (?, ?)", (d, 100000.0))
    conn.commit()
    conn.close()


def test_the_guard_is_autouse() -> None:
    """Read the fixture marker, not the source text it happens to sit in."""
    from _pytest.fixtures import getfixturemarker
    from tests import conftest

    for name in ("_protect_live_ledger", "_redirect_configured_ledger"):
        marker = getfixturemarker(getattr(conftest, name))
        assert marker is not None, f"{name} is not a fixture"
        assert marker.autouse, f"{name} is not autouse"


def test_a_ledger_with_recent_history_is_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    _make_ledger(db, ["2026-10-20", "2026-10-21"])
    assert ledger_holds_live_history(db, {"2021-07-01", "2021-07-02"})


def test_a_ledger_of_only_fixture_dates_is_not_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    _make_ledger(db, ["2021-07-01", "2021-07-02"])
    assert not ledger_holds_live_history(db, {"2021-07-01", "2021-07-02"})


def test_a_missing_ledger_is_not_live(tmp_path: Path) -> None:
    assert not ledger_holds_live_history(tmp_path / "absent.db", {"2021-07-01"})


def test_an_empty_ledger_is_not_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    _make_ledger(db, [])
    assert not ledger_holds_live_history(db, {"2021-07-01"})


def test_some_other_sqlite_file_is_not_live(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.execute("create table other (x int)")
    conn.commit()
    conn.close()
    assert not ledger_holds_live_history(db, {"2021-07-01"})


def test_the_fixture_dates_cover_the_destructive_class() -> None:
    """Every date the ledger-deleting class replays must be listed.

    A date missing from the set makes the guard read that fixture's own
    leftovers as live history and skip the test from then on. Only the class
    that deletes the configured ledger matters here; tests on tmp_path or an
    in-memory connection never reach it.
    """
    import re

    src = Path("tests/paper/test_acceptance.py").read_text()
    start = src.index("class TestRealPredictionReplay")
    rest = src[start:]
    end = rest.find("\nclass ")
    cls = rest[:end] if end > 0 else rest
    assert "db_path.unlink()" in cls, (
        "TestRealPredictionReplay no longer deletes the ledger; "
        "re-point this test at whatever does"
    )
    used = set(re.findall(r'"(20[12]\d-\d\d-\d\d)"', cls))
    assert used, "no dates found in that class; update this test"
    missing = used - set(FIXTURE_LEDGER_DATES)
    assert not missing, f"dates the guard does not know about: {sorted(missing)}"


def test_deleting_a_live_ledger_skips_instead(tmp_path: Path) -> None:
    """End to end: the delete is refused and the rows survive."""
    ledger = tmp_path / "paper.db"
    _make_ledger(ledger, ["2026-10-20", "2026-10-21"])

    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(Path.cwd())!r})
            import tests.conftest as real

            def _configured():
                return Path({str(ledger)!r})

            real.configured_ledger_path = _configured
            _protect_live_ledger = real._protect_live_ledger
            """
        )
    )
    (tmp_path / "test_wipe.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_would_wipe_the_ledger():
                Path({str(ledger)!r}).unlink()
            """
        )
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "test_wipe.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert ledger.exists(), f"the ledger was deleted:\n{result.stdout}"
    conn = sqlite3.connect(ledger)
    rows = conn.execute("select count(*) from nav").fetchone()[0]
    conn.close()
    assert rows == 2, f"ledger rows changed: {rows}"
    # the bare word would also show up in a collection error, so pin the run
    # down: it has to have succeeded, and our test has to be the one skipped
    assert result.returncode == 0, (
        f"pytest exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert "1 skipped" in result.stdout, f"not skipped:\n{result.stdout}"


def test_a_path_that_cannot_be_resolved_is_refused(tmp_path: Path) -> None:
    """When the guard cannot tell what it is looking at, it must refuse.

    resolve() can raise OSError, and reading that as "not the ledger" opens
    the delete on exactly the paths the guard knows least about. Same shape
    as the test above, with resolve() broken inside the child.
    """
    ledger = tmp_path / "paper.db"
    _make_ledger(ledger, ["2026-10-20", "2026-10-21"])

    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(Path.cwd())!r})
            import tests.conftest as real

            real.configured_ledger_path = lambda: Path({str(ledger)!r})
            _protect_live_ledger = real._protect_live_ledger
            """
        )
    )
    (tmp_path / "test_wipe.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            LEDGER = Path({str(ledger)!r})

            def test_delete_with_resolve_broken(monkeypatch):
                real_resolve = Path.resolve

                def exploding(self, *args, **kwargs):
                    if self.name == "paper.db":
                        raise OSError("cannot resolve")
                    return real_resolve(self, *args, **kwargs)

                monkeypatch.setattr(Path, "resolve", exploding)
                LEDGER.unlink()
            """
        )
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "test_wipe.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert ledger.exists(), (
        "the ledger was deleted when resolve() failed; the guard must refuse "
        f"rather than allow:\n{result.stdout}"
    )
    assert "1 skipped" in result.stdout, (
        f"expected a skip, got:\n{result.stdout}"
    )


def test_an_unrelated_delete_still_works(tmp_path: Path) -> None:
    """The guard must not break every other test's cleanup."""
    scratch = tmp_path / "scratch.txt"
    scratch.write_text("x")
    scratch.unlink()
    assert not scratch.exists()


def test_the_real_checkout_is_reported_honestly() -> None:
    """Whatever this checkout is, say which branch the guard takes here."""
    ledger = configured_ledger_path()
    assert ledger is not None, "could not resolve the configured ledger"
    live = ledger_holds_live_history(ledger, FIXTURE_LEDGER_DATES)
    print(f"configured ledger {ledger} live={live}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
