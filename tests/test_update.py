"""Tests for ashare_lab.data.update (non-network functions)."""
import tempfile
import datetime as dt
from pathlib import Path
import pytest
from ashare_lab.data.update import (
    _download_url,
    _read_calendar_last_date,
    CHENDITC_REPO,
    ASSET_NAME,
)


def test_download_url_format():
    url = _download_url(CHENDITC_REPO, "2026-06-12")
    assert "chenditc/investment_data" in url
    assert "2026-06-12" in url
    assert ASSET_NAME in url
    assert url.startswith("https://github.com/")


def test_read_calendar_last_date_missing_dir():
    with tempfile.TemporaryDirectory() as tmp:
        result = _read_calendar_last_date(Path(tmp))
    assert result is None


def test_read_calendar_last_date_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(tmp) / "calendars"
        cal_dir.mkdir()
        result = _read_calendar_last_date(Path(tmp))
    assert result is None


def test_read_calendar_last_date_valid():
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(tmp) / "calendars"
        cal_dir.mkdir()
        (cal_dir / "day.txt").write_text("2026-06-10\n2026-06-11\n2026-06-12\n")
        result = _read_calendar_last_date(Path(tmp))
    assert result == "2026-06-12"


def test_read_calendar_last_date_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(tmp) / "calendars"
        cal_dir.mkdir()
        (cal_dir / "day.txt").write_text("")
        result = _read_calendar_last_date(Path(tmp))
    assert result is None
