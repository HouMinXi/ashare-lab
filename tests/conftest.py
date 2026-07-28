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
