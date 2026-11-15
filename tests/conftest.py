"""Root-level pytest fixtures shared across all test subdirectories."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _ashare_dry_run(monkeypatch):
    """Prevent real HTTP sends to push services during tests.

    Sets ASHARE_DRY_RUN=1 so send_serverchan/send_pushplus skip actual
    HTTP calls.  Tests that need to verify the real HTTP path must
    explicitly ``monkeypatch.delenv("ASHARE_DRY_RUN")``.

    Note: iLink and alert_bridge paths are always mocked in tests via
    @patch, so they don't need this guard.
    """
    monkeypatch.setenv("ASHARE_DRY_RUN", "1")


@pytest.fixture(autouse=True)
def _no_mock_artifacts(request):
    """Fail if a test creates MagicMock-named filesystem artifacts.

    After each test, scans the current working directory for any directory
    whose name contains 'MagicMock'.  If found, removes it and fails the
    test with an actionable message.

    This catches the class of bug where a MagicMock object is passed to
    Path() / os.mkdir() / pathlib operations, producing literal
    directories like:
        <MagicMock name='...' id='...'>/
    """
    yield
    cwd = Path.cwd()
    for entry in cwd.iterdir():
        if entry.is_dir() and "MagicMock" in entry.name:
            # Remove the artifact so it doesn't pollute later runs.
            try:
                entry.rmdir()
            except OSError:
                pass  # non-empty; leave it, but still fail
            pytest.fail(
                f"Test created MagicMock filesystem artifact: {entry.name}\n"
                f"Fix the test to avoid passing mock objects as paths "
                f"(Path(), os.mkdir(), pd.DataFrame.to_csv(), etc.)."
            )


# The repository's own experiment record, resolved from this file rather than
# from the cwd, so a test that moved the cwd is still measured against the
# real directory.
REPO_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "control_books"


def _artifact_names() -> frozenset[str]:
    try:
        return frozenset(p.name for p in REPO_ARTIFACT_DIR.iterdir())
    except OSError:
        return frozenset()


@pytest.fixture(scope="session")
def _artifacts_at_session_start() -> frozenset[str]:
    return _artifact_names()


@pytest.fixture(autouse=True)
def _no_artifacts_in_the_repo(_artifacts_at_session_start, request):
    """Fail a test that writes into the real experiment record.

    _write_abc_artifact builds a relative path, so a test that reaches it
    without moving the cwd drops fixture values next to genuine daily
    records. Six files did, and the fixture artifacts they left in the
    production directory are indistinguishable from real ones at a glance:
    same naming, plausible shape, nav figures that are simply wrong.

    Adding a chdir fixture per file fixes the files that exist today. This
    fails the next one on its first run, and says which test did it.
    """
    yield
    added = _artifact_names() - _artifacts_at_session_start
    if added:
        for name in added:
            try:
                (REPO_ARTIFACT_DIR / name).unlink()
            except OSError:
                pass
        pytest.fail(
            f"{request.node.nodeid} wrote {sorted(added)} into "
            f"{REPO_ARTIFACT_DIR}, the real experiment record.\n"
            "The artifact writer uses a relative path: add "
            "monkeypatch.chdir(tmp_path) to this file, as the other "
            "artifact tests do."
        )


# Column names that carry a trading date somewhere in the ledger.
LEDGER_DATE_COLUMNS = ("trade_date", "date", "as_of_date")


def ledger_holds_live_history(db_path, fixture_dates):
    """Is this ledger carrying history a fixture run cannot account for?

    False only for a ledger that is absent, or whose every dated table holds
    nothing but the dates a fixture itself writes. Anything else is a real
    book.

    Every dated table gets a say, not just nav. Asking nav alone let 21 of
    22 destruction vectors through: emptying one table, or shadowing it with
    a view, disarmed the guard while months of history sat in the rest.
    """
    import sqlite3

    db_path = Path(db_path)
    if not db_path.exists():
        return False
    try:
        # as_uri() quotes the path ('?' and '#' are not safe to interpolate)
        # and produces the absolute file:/// form on every platform
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    except (sqlite3.Error, OSError, ValueError):
        # Something is at the ledger path that will not open. Treat it as a
        # real book: a skipped test is recoverable, a deleted ledger is not.
        return True
    try:
        # views count, so replacing a table with one cannot hide its rows
        relations = [
            row[0]
            for row in conn.execute(
                "select name from sqlite_master "
                "where type in ('table','view') and name not like 'sqlite_%'"
            )
        ]
        dates = set()
        for name in relations:
            # identifiers escape by doubling, not by backslash
            quoted = name.replace('"', '""')
            cols = {
                row[1]
                for row in conn.execute(f'pragma table_info("{quoted}")')
            }
            for col in LEDGER_DATE_COLUMNS:
                if col in cols:
                    dates.update(
                        row[0]
                        for row in conn.execute(
                            f'select distinct "{col}" from "{quoted}"'
                        )
                        if row[0]
                    )
                    break
    except sqlite3.Error:
        # Readable enough to open, not enough to clear. Same reasoning.
        return True
    finally:
        conn.close()
    return bool(dates - set(fixture_dates))


def configured_ledger_path():
    """The ledger the pipeline will use, or None if it cannot be resolved."""
    try:
        from ashare_lab.config import PROJECT_ROOT, load_config

        return PROJECT_ROOT / load_config()["paper"]["db_path"]
    except Exception:
        return None


# The window TestRealPredictionReplay replays into the configured ledger.
# A ledger holding only these dates is that fixture's own leftovers.
FIXTURE_LEDGER_DATES = frozenset(
    {
        "2021-07-01",
        "2021-07-02",
        "2021-07-05",
        "2021-07-06",
        "2021-07-07",
        "2021-07-08",
    }
)


def _clear_config_cache(cfg) -> None:
    """Clear load_config's cache when it has one.

    Tests are allowed to replace load_config with a plain function; there
    is no cache to clear then, and asking for one fails the teardown.
    """
    clear = getattr(cfg.load_config, "cache_clear", None)
    if clear is not None:
        clear()


@pytest.fixture(scope="session")
def _redirected_config_file(tmp_path_factory):
    """A copy of baseline.yaml whose ledger points at scratch space.

    Built once. Applied per test by the fixture below, because nothing
    applied once stays applied: load_config.cache_clear() is called from
    eight places and matrix_runner reassigns CONFIG_PATH without restoring
    it.
    """
    import yaml

    from ashare_lab import config as cfg

    if not ledger_holds_live_history(
        configured_ledger_path(), FIXTURE_LEDGER_DATES
    ):
        # Nothing live in this checkout, so there is nothing to redirect and
        # a fixture-only ledger keeps behaving exactly as it did.
        return None

    scratch_dir = tmp_path_factory.mktemp("configured-ledger")
    data = yaml.safe_load(cfg.CONFIG_PATH.read_text())
    # absolute, so PROJECT_ROOT / value returns the value unchanged and the
    # three places the acceptance test computes the path itself follow too
    data["paper"]["db_path"] = str(scratch_dir / "paper.db")

    path = scratch_dir / "baseline.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture(autouse=True)
def _redirect_configured_ledger(_redirected_config_file, monkeypatch):
    """Keep the live ledger out of the path production code resolves.

    An adversarial pass found 21 of 22 ways to destroy a live ledger past an
    unlink interceptor -- os.remove, open(path,'w'), shutil.move, SQL DROP, a
    subprocess, a reference to the original unlink saved before the patch, a
    Path subclass, a fixture scoped above the guard. The worst case deleted
    nothing at all: a test calling run_daily() for fixture dates wrote 2021
    rows over 2026 history and reported 1 passed.

    Interception has to enumerate the ways in. Redirecting the config file
    removes the destination instead, so those vectors land on scratch.
    """
    if _redirected_config_file is None:
        yield
        return

    from ashare_lab import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", _redirected_config_file)
    _clear_config_cache(cfg)
    try:
        yield
    finally:
        # hand the next test a clean read of whatever CONFIG_PATH it gets
        _clear_config_cache(cfg)


@pytest.fixture(autouse=True)
def _protect_live_ledger(monkeypatch):
    """Refuse to delete a ledger that holds real history.

    The replay tests drive run_daily(), which resolves its database from the
    config and takes no path argument, so they delete that file to start
    clean. In a checkout that also carries the production ledger, that is
    months of paper-trading history, and the suite destroyed it once.

    Deleting is the step that cannot be undone, so that is where this stops:
    any test, including one written later that never heard of this guard,
    gets skipped instead.
    """
    ledger = configured_ledger_path()
    if ledger is None or not ledger_holds_live_history(
        ledger, FIXTURE_LEDGER_DATES
    ):
        return

    resolved = ledger.resolve()
    real_unlink = Path.unlink

    def guarded_unlink(self, *args, **kwargs):
        try:
            same = self.resolve() == resolved
        except OSError:
            # cannot tell what this path is. Refusing costs a skipped test;
            # allowing cost the ledger once already, so refuse.
            same = True
        if same:
            pytest.skip(
                f"{self} holds paper-trading history outside the fixture "
                "dates, so this checkout owns a live ledger. Run this test in "
                "a worktree or a copy, where the pipeline has nothing to lose."
            )
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)


_STACK_SKIPS: list[str] = []


def _note_stack_skip(report) -> None:
    longrepr = str(getattr(report, "longrepr", "") or "")
    if "where the training stack is installed" in longrepr:
        _STACK_SKIPS.append(str(getattr(report, "nodeid", "")) or longrepr[:200])


def pytest_runtest_logreport(report):
    if report.skipped:
        _note_stack_skip(report)


def pytest_collectreport(report):
    if report.skipped:
        _note_stack_skip(report)


def pytest_sessionfinish(session, exitstatus):
    """A skipped training check is invisible unless someone asks for it.

    With ASHARE_REQUIRE_STACK=1 the run is claiming to have exercised the
    training path, so any such skip turns the session red.
    """
    import os

    if os.environ.get("ASHARE_REQUIRE_STACK", "").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return
    if not _STACK_SKIPS:
        return
    print(
        "\nASHARE_REQUIRE_STACK is set but these checks were skipped:",
        *sorted(set(_STACK_SKIPS)),
        sep="\n  ",
    )
    session.exitstatus = 1
