"""Redirect the ledger instead of intercepting the delete.

An adversarial pass ran 22 ways of destroying a live ledger with the
interception guard armed, and 21 worked: os.remove, os.unlink, os.truncate,
open(path,'w'), Path.write_bytes, shutil.copy2/move, Path.rename/replace,
shutil.rmtree of the parent, SQL DROP and DELETE, a subprocess, a
module-level saved reference to the original unlink, functools.partial
bound at import, a test re-patching unlink back, a Path subclass with its
own unlink, and higher-scoped fixtures that run before the guard is
installed or after monkeypatch is undone. Patching one attribute catches
one attribute.

The worst case did not delete anything: test_real_prediction_idempotent
calls run_daily() for five fixture dates with no unlink at all, writing
2021 rows over 2026 history. The adversarial run lost 56 real days and
gained 6 fixture days, reporting 1 passed. No delete guard can see that.

load_config is lru_cached and hands every caller the same dict, so moving
paper.db_path in that dict moves it for the whole process -- including
run_daily's own internal resolution and the three places test_acceptance
computes the path itself. An absolute replacement survives being joined to
PROJECT_ROOT, because joining an absolute path returns it unchanged.

With no live path reachable, all 22 vectors operate on a scratch copy.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

from ashare_lab.config import PROJECT_ROOT, load_config

from tests.conftest import (
    FIXTURE_LEDGER_DATES,
    configured_ledger_path,
    ledger_holds_live_history,
)


def test_the_configured_ledger_is_not_the_real_one_during_tests() -> None:
    """The path production code resolves must not be the live book."""
    resolved = PROJECT_ROOT / load_config()["paper"]["db_path"]
    real = PROJECT_ROOT / "paper.db"
    if not ledger_holds_live_history(real, FIXTURE_LEDGER_DATES):
        # nothing live in this checkout, so there is nothing to redirect
        return
    assert resolved.resolve() != real.resolve(), (
        f"tests resolve the ledger to {resolved}, which is the live book"
    )


def test_the_redirect_survives_the_project_root_join() -> None:
    """Every call site joins PROJECT_ROOT, so the value has to be absolute."""
    configured = load_config()["paper"]["db_path"]
    real = PROJECT_ROOT / "paper.db"
    if not ledger_holds_live_history(real, FIXTURE_LEDGER_DATES):
        return
    assert Path(configured).is_absolute(), (
        f"db_path is {configured!r}; a relative value would land back under "
        "the project root"
    )
    assert (PROJECT_ROOT / configured) == Path(configured)


def test_writes_through_the_configured_path_miss_the_real_ledger(
    tmp_path: Path,
) -> None:
    """The vector that needed no delete: write fixture rows and check."""
    real = PROJECT_ROOT / "paper.db"
    if not ledger_holds_live_history(real, FIXTURE_LEDGER_DATES):
        return
    before = real.read_bytes()

    # its own table name, so the assertion is about where the write landed
    # and not about whatever schema happens to be at the other end
    target = configured_ledger_path()
    conn = sqlite3.connect(target)
    conn.execute("create table if not exists redirect_probe (note text)")
    conn.execute("insert into redirect_probe values ('written by a test')")
    conn.commit()
    conn.close()

    assert real.read_bytes() == before, (
        "a write through the configured path reached the live ledger"
    )


def test_a_child_process_still_sees_the_real_ledger(tmp_path: Path) -> None:
    """A recorded gap, not a closed one.

    The redirect lives in this process's lru_cached config dict. A child
    re-reads baseline.yaml from disk and so resolves the real path. No test
    in the suite currently hands a child a command that writes to the
    ledger, which is the only reason this is survivable.

    Pinned deliberately: if the redirect is ever extended to children, this
    fails and gets updated, rather than the gap quietly persisting as an
    assumption nobody rechecks.
    """
    real = PROJECT_ROOT / "paper.db"
    if not ledger_holds_live_history(real, FIXTURE_LEDGER_DATES):
        return

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from ashare_lab.config import PROJECT_ROOT, load_config
        print(PROJECT_ROOT / load_config()["paper"]["db_path"])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-400:]
    child_path = Path(result.stdout.strip().splitlines()[-1])
    assert child_path == real, (
        f"a child resolved {child_path}, not the real ledger at {real}. "
        "If the redirect now reaches children, update this test to assert "
        "that instead."
    )
