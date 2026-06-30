"""Tests for chenditc CLI subcommands (snapshot + diff).

These test the early-exit and error-handling paths that do not require
a live qlib installation.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# cmd_chenditc_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_non_trading_day():
    """Non-trading day prints message and returns 0 without touching qlib."""
    from ashare_lab.cli import cmd_chenditc_snapshot

    with patch("ashare_lab.data.calendar.is_trading_day", return_value=False):
        rc = cmd_chenditc_snapshot(Namespace())
    assert rc == 0


def test_snapshot_qlib_import_error(tmp_path):
    """When qlib is not installed, snapshot writes empty JSON and returns 0."""
    from ashare_lab.cli import cmd_chenditc_snapshot

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _raise_on_qlib(name, *args, **kwargs):
        if name == "qlib":
            raise ImportError("No module named 'qlib'")
        return real_import(name, *args, **kwargs)

    snap_file = tmp_path / "snap.json"

    with (
        patch("ashare_lab.data.calendar.is_trading_day", return_value=True),
        patch("ashare_lab.data.calendar.latest_trading_day", return_value="2026-06-26"),
        patch("builtins.__import__", side_effect=_raise_on_qlib),
        patch("ashare_lab.cli.Path", side_effect=lambda p: snap_file if "snapshot" in str(p) else Path(p)),
    ):
        rc = cmd_chenditc_snapshot(Namespace())

    assert rc == 0
    assert json.loads(snap_file.read_text()) == {}


# ---------------------------------------------------------------------------
# cmd_chenditc_diff
# ---------------------------------------------------------------------------


def test_diff_no_snapshot_file():
    """Missing snapshot file: returns 0 without querying qlib."""
    from ashare_lab.cli import cmd_chenditc_diff

    with (
        patch("ashare_lab.data.calendar.latest_trading_day", return_value="2026-06-26"),
        patch.object(Path, "exists", return_value=False),
    ):
        rc = cmd_chenditc_diff(Namespace())
    assert rc == 0


def test_diff_empty_snapshot():
    """Empty snapshot dict: returns 0 without querying qlib."""
    from ashare_lab.cli import cmd_chenditc_diff

    with (
        patch("ashare_lab.data.calendar.latest_trading_day", return_value="2026-06-26"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "read_text", return_value="{}"),
    ):
        rc = cmd_chenditc_diff(Namespace())
    assert rc == 0


def test_diff_qlib_failure_returns_zero():
    """Qlib query failure in diff is non-fatal: returns 0."""
    from ashare_lab.cli import cmd_chenditc_diff

    snapshot = {"SZ000001": 10.5}

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _raise_on_qlib(name, *args, **kwargs):
        if name == "qlib":
            raise ImportError("No module named 'qlib'")
        return real_import(name, *args, **kwargs)

    with (
        patch("ashare_lab.data.calendar.latest_trading_day", return_value="2026-06-26"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "read_text", return_value=json.dumps(snapshot)),
        patch("builtins.__import__", side_effect=_raise_on_qlib),
    ):
        rc = cmd_chenditc_diff(Namespace())
    assert rc == 0
