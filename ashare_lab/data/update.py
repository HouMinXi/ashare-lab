"""Bootstrap and daily refresh of qlib data from chenditc/investment_data."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests

from ashare_lab.data.calendar import is_trading_day, latest_trading_day

log = logging.getLogger(__name__)

CHENDITC_REPO = "chenditc/investment_data"
ASSET_NAME = "qlib_bin.tar.gz"
DEFAULT_PROVIDER_URI = Path.home() / ".qlib" / "qlib_data" / "cn_data"
GITHUB_API = "https://api.github.com"
DOWNLOAD_TIMEOUT = 1800
CHUNK_SIZE = 8 * 1024 * 1024


def _latest_release_tag(repo: str = CHENDITC_REPO) -> str | None:
    """Get the latest release tag from GitHub (date string like '2026-06-12').

    Tries ``gh`` CLI first (authenticated), falls back to unauthenticated
    GitHub API.  Returns None on any transient failure so callers can
    decide whether to retry or skip.
    """
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/latest", "--jq", ".tag_name"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{repo}/releases/latest",
            timeout=30,
        )
        if resp.status_code == 403:
            log.warning("GitHub API rate-limited (HTTP 403); try `gh auth login`")
            return None
        resp.raise_for_status()
        return resp.json().get("tag_name")
    except requests.RequestException as exc:
        log.warning("GitHub API request failed: %s", exc)
        return None


def _download_url(repo: str, tag: str) -> str:
    return (
        f"https://github.com/{repo}/releases/download/{tag}/{ASSET_NAME}"
    )


def _download_tarball(url: str, dest: Path) -> Path:
    """Download qlib_bin.tar.gz with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = dest / ASSET_NAME
    headers = {}
    mode = "wb"
    existing = 0
    if out.exists():
        existing = out.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    log.info("downloading %s -> %s (resume from %d)", url, out, existing)
    resp = requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)

    if resp.status_code == 416:
        log.info("file already complete")
        return out
    resp.raise_for_status()

    with open(out, mode) as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)

    actual_size = out.stat().st_size
    expected_size = resp.headers.get("Content-Length")
    if expected_size and actual_size < int(expected_size):
        out.unlink()
        raise RuntimeError(
            f"incomplete download: got {actual_size}, expected {expected_size}"
        )
    log.info("download complete: %s (%d bytes)", out, actual_size)
    return out


def _extract_and_swap(tarball: Path, provider_uri: Path) -> None:
    """Extract tarball to a staging dir, then atomically swap into place."""
    parent = provider_uri.parent
    parent.mkdir(parents=True, exist_ok=True)

    staging = parent / f"cn_data_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    log.info("extracting %s -> %s", tarball, staging)
    with tarfile.open(tarball) as tf:
        tf.extractall(staging, filter="data")

    extracted_dirs = list(staging.iterdir())
    if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
        inner = extracted_dirs[0]
        actual = parent / f"cn_data_inner_{os.getpid()}"
        inner.rename(actual)
        shutil.rmtree(staging)
        staging = actual

    prev = parent / "cn_data_prev"
    if prev.exists():
        shutil.rmtree(prev)

    if provider_uri.exists():
        provider_uri.rename(prev)
        log.info("old data moved to %s", prev)

    try:
        staging.rename(provider_uri)
    except OSError:
        if prev.exists():
            prev.rename(provider_uri)
            log.error("swap failed, rolled back to previous data")
        raise
    log.info("new data live at %s", provider_uri)


def _read_calendar_last_date(provider_uri: Path) -> str | None:
    """Read the last line of calendars/day.txt to get the latest date in data."""
    cal_file = provider_uri / "calendars" / "day.txt"
    if not cal_file.exists():
        return None
    lines = cal_file.read_text().strip().splitlines()
    return lines[-1].strip() if lines else None


def bootstrap(
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    repo: str = CHENDITC_REPO,
) -> Path:
    """First-time download: get latest release and extract to provider_uri."""
    tag = _latest_release_tag(repo)
    if not tag:
        raise RuntimeError("cannot determine latest release tag")
    log.info("bootstrap: latest release tag = %s", tag)

    url = _download_url(repo, tag)
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = _download_tarball(url, Path(tmpdir))
        _extract_and_swap(tarball, provider_uri)

    last_date = _read_calendar_last_date(provider_uri)
    log.info("bootstrap complete. calendar last date: %s", last_date)
    return provider_uri


def daily_refresh(
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    repo: str = CHENDITC_REPO,
) -> int:
    """Refresh qlib data if a new trading day's data is expected.

    Returns:
        0 = success (data updated or no update needed)
        1 = stale (chenditc release not yet available, day skipped)

    Raises on fatal errors (network, extraction, filesystem).
    """
    expected = latest_trading_day()
    if not is_trading_day(expected):
        log.info("not a trading day (%s), skipping", expected)
        return 0

    current_last = _read_calendar_last_date(provider_uri)
    if current_last and current_last >= expected.isoformat():
        log.info("data already current (last=%s, expected=%s)", current_last, expected)
        return 0

    tag = _latest_release_tag(repo)
    if not tag:
        log.warning("cannot fetch release tag")
        return 1

    if tag < expected.isoformat():
        log.warning(
            "chenditc release %s is behind expected %s -- data stale",
            tag,
            expected,
        )
        return 1

    log.info("refreshing: release %s covers expected %s", tag, expected)
    url = _download_url(repo, tag)
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = _download_tarball(url, Path(tmpdir))
        _extract_and_swap(tarball, provider_uri)

    new_last = _read_calendar_last_date(provider_uri)
    log.info("refresh complete. calendar last date: %s", new_last)
    return 0
