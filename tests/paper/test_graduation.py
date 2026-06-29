"""Tests for the graduation gate logic.

Covers: empty table, all-success, below-threshold, exact-threshold,
stale-exclusion, stale-exceeds-max, boundary (30+3), window-excludes-
old-stale, custom thresholds, keyword args, and one-shot notification
guard (first pass, after notification, regression clearing, delivery
success, delivery failure).
"""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from ashare_lab.paper.graduation import check_graduation, notify_graduation
from ashare_lab.paper.ledger import init_schema, insert_pipeline_run


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory SQLite with full ledger schema."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    yield c
    c.close()


def _fill(conn: sqlite3.Connection, n: int, status: str = "success",
          date_prefix: str = "2026-01") -> None:
    """Insert *n* pipeline_run rows with sequential dates."""
    for i in range(1, n + 1):
        insert_pipeline_run(
            conn, f"{date_prefix}-{i:02d}", status, 10.0, None,
            f"{date_prefix}-{i:02d}" if status == "success" else None,
        )
    conn.commit()


# -- Gate logic tests --------------------------------------------------------

def test_empty_table(conn: sqlite3.Connection) -> None:
    passed, s = check_graduation(conn)
    assert not passed
    assert s["denominator"] == 0
    assert s["days_remaining"] == 30
    assert s["should_notify"] is False


def test_all_success_30(conn: sqlite3.Connection) -> None:
    _fill(conn, 30)
    passed, s = check_graduation(conn)
    assert passed
    assert s["rate"] == 1.0
    assert s["days_remaining"] == 0
    assert s["should_notify"] is True


def test_below_threshold(conn: sqlite3.Connection) -> None:
    """28 success + 2 error = 93.3% < 95% -> fail."""
    _fill(conn, 28)
    for i in range(29, 31):
        insert_pipeline_run(conn, f"2026-01-{i:02d}", "error", 5.0,
                            "boom", None)
    conn.commit()
    passed, s = check_graduation(conn)
    assert not passed
    assert s["denominator"] == 30
    assert abs(s["rate"] - 28 / 30) < 1e-4


def test_exact_at_threshold(conn: sqlite3.Connection) -> None:
    """29 success + 1 error = 96.7% >= 95% -> pass."""
    _fill(conn, 29)
    insert_pipeline_run(conn, "2026-01-30", "error", 5.0, "oops", None)
    conn.commit()
    passed, s = check_graduation(conn)
    assert passed
    assert s["denominator"] == 30
    assert abs(s["rate"] - 29 / 30) < 1e-4


def test_stale_excluded_from_denominator(conn: sqlite3.Connection) -> None:
    """30 success + 2 stale -> denominator=30, rate=1.0, stale=2 -> pass."""
    _fill(conn, 30)
    for i in range(1, 3):
        insert_pipeline_run(conn, f"2026-02-{i:02d}", "stale", 10.0,
                            None, "2026-01-25")
    conn.commit()
    passed, s = check_graduation(conn)
    assert passed
    assert s["denominator"] == 30
    assert s["rate"] == 1.0
    assert s["stale"] == 2


def test_stale_exceeds_max(conn: sqlite3.Connection) -> None:
    """30 success + 4 stale -> fail (stale > max_stale=3)."""
    _fill(conn, 30)
    for i in range(1, 5):
        insert_pipeline_run(conn, f"2026-02-{i:02d}", "stale", 10.0,
                            None, "2026-01-25")
    conn.commit()
    passed, s = check_graduation(conn)
    assert not passed
    assert s["stale"] == 4


def test_stale_pauses_not_fails(conn: sqlite3.Connection) -> None:
    """25 success + 3 stale -> denominator=25 < 30 -> not enough days."""
    _fill(conn, 25)
    for i in range(26, 29):
        insert_pipeline_run(conn, f"2026-01-{i:02d}", "stale", 10.0,
                            None, "2026-01-20")
    conn.commit()
    passed, s = check_graduation(conn)
    assert not passed
    assert s["denominator"] == 25
    assert s["days_remaining"] == 5


def test_30_success_3_stale_boundary(conn: sqlite3.Connection) -> None:
    """30 success + 3 stale -> pass (stale exactly at max_stale)."""
    _fill(conn, 30)
    for i in range(1, 4):
        insert_pipeline_run(conn, f"2026-02-{i:02d}", "stale", 10.0,
                            None, "2026-01-25")
    conn.commit()
    passed, s = check_graduation(conn)
    assert passed
    assert s["stale"] == 3


def test_custom_thresholds(conn: sqlite3.Connection) -> None:
    """Custom min_days=10, min_rate=0.9, max_stale=5."""
    _fill(conn, 9)
    insert_pipeline_run(conn, "2026-01-10", "error", 5.0, "err", None)
    conn.commit()
    # 9/10 = 0.9 -> pass at min_rate=0.9
    passed, s = check_graduation(conn, min_days=10, min_rate=0.9,
                                  max_stale=5)
    assert passed
    assert s["denominator"] == 10


def test_keyword_args(conn: sqlite3.Connection) -> None:
    """Verify insert_pipeline_run accepts keyword arguments."""
    insert_pipeline_run(
        conn,
        trade_date="2026-03-01",
        status="success",
        duration_s=7.5,
        error_msg=None,
        predictions_date="2026-03-01",
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE trade_date='2026-03-01'"
    ).fetchone()
    assert row["status"] == "success"
    assert row["duration_s"] == 7.5


def test_window_excludes_old_stale(conn: sqlite3.Connection) -> None:
    """55 rows total (50 success + 5 old stale). LIMIT=43 excludes old stale.

    The 5 stale rows have the oldest dates (2025-11-01..05), so the
    LIMIT 43 window only fetches the 43 most recent success rows.
    The gate sees 30 successes (stops at denominator=30), 0 stale -> pass.
    """
    # 5 old stale rows
    for i in range(1, 6):
        insert_pipeline_run(conn, f"2025-11-{i:02d}", "stale", 10.0,
                            None, "2025-10-30")
    # 50 success rows (dates after the stale ones)
    for i in range(1, 51):
        d = 1 + i  # 2026-01-02 through 2026-02-20
        month = (d - 1) // 28 + 1
        day = (d - 1) % 28 + 1
        insert_pipeline_run(conn, f"2026-{month:02d}-{day:02d}",
                            "success", 10.0, None,
                            f"2026-{month:02d}-{day:02d}")
    conn.commit()

    # Verify total rows = 55
    total = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    assert total == 55

    passed, s = check_graduation(conn)
    assert passed
    assert s["stale"] == 0  # old stale excluded by window
    assert s["denominator"] == 30
    assert s["rate"] == 1.0


# -- One-shot notification guard tests --------------------------------------

def test_should_notify_true_on_first_graduation(
    conn: sqlite3.Connection,
) -> None:
    _fill(conn, 30)
    passed, s = check_graduation(conn)
    assert passed
    assert s["should_notify"] is True


def test_should_notify_false_after_notification(
    conn: sqlite3.Connection,
) -> None:
    _fill(conn, 30)
    conn.execute(
        "INSERT OR REPLACE INTO graduation_status "
        "(id, graduated_at, notified_at) VALUES (1, '2026-01-30', '2026-01-30')"
    )
    conn.commit()
    passed, s = check_graduation(conn)
    assert passed
    assert s["should_notify"] is False


def test_should_notify_cleared_on_regression(
    conn: sqlite3.Connection,
) -> None:
    """After regression, notified_at is cleared; on recovery, should_notify=True."""
    _fill(conn, 30)
    conn.execute(
        "INSERT OR REPLACE INTO graduation_status "
        "(id, graduated_at, notified_at) VALUES (1, '2026-01-30', '2026-01-30')"
    )
    conn.commit()

    # Add 4 stale -> gate fails -> clears notified_at
    for i in range(1, 5):
        insert_pipeline_run(conn, f"2026-02-{i:02d}", "stale", 10.0,
                            None, "2026-01-25")
    conn.commit()
    passed, s = check_graduation(conn)
    assert not passed

    row = conn.execute(
        "SELECT notified_at FROM graduation_status WHERE id=1"
    ).fetchone()
    assert row is None or row["notified_at"] is None

    # Add enough successes to fill the denominator window before hitting
    # any stale rows (need 30 success rows more recent than the stale rows)
    for i in range(5, 35):
        month = 2 + (i - 1) // 28
        day = (i - 1) % 28 + 1
        insert_pipeline_run(conn, f"2026-{month:02d}-{day:02d}",
                            "success", 10.0, None,
                            f"2026-{month:02d}-{day:02d}")
    conn.commit()
    passed, s = check_graduation(conn)
    assert passed
    assert s["should_notify"] is True


def test_notify_graduation_writes_timestamp(
    conn: sqlite3.Connection,
) -> None:
    """Mock iLink send -> verify graduation_status.notified_at is set."""
    _fill(conn, 30)
    _, stats = check_graduation(conn)

    mock_send = AsyncMock(return_value={"errcode": 0})
    with (
        patch("ashare_lab.paper.report._get_secret",
              return_value="fake"),
        patch("ashare_lab.paper.report.send_text_ilink", mock_send),
    ):
        result = notify_graduation(conn, stats)

    assert result is True
    row = conn.execute(
        "SELECT notified_at FROM graduation_status WHERE id=1"
    ).fetchone()
    assert row is not None and row["notified_at"] is not None


def test_notify_graduation_no_write_on_failure(
    conn: sqlite3.Connection,
) -> None:
    """Mock iLink send to raise -> notified_at stays NULL."""
    _fill(conn, 30)
    _, stats = check_graduation(conn)

    mock_send = AsyncMock(side_effect=RuntimeError("network error"))
    with (
        patch("ashare_lab.paper.report._get_secret",
              return_value="fake"),
        patch("ashare_lab.paper.report.send_text_ilink", mock_send),
    ):
        result = notify_graduation(conn, stats)

    assert result is False
    row = conn.execute(
        "SELECT notified_at FROM graduation_status WHERE id=1"
    ).fetchone()
    assert row is None or row["notified_at"] is None
