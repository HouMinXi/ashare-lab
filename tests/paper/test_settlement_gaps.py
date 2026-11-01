"""A skipped day must appear in the status output, not vanish."""

from __future__ import annotations

import sqlite3

import pytest

from ashare_lab.paper.ledger import init_schema, record_run, settlement_gaps


@pytest.fixture()
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    init_schema(connection)
    yield connection
    connection.close()


# 2026-09-25 is the Mid-Autumn holiday, so it is not a session and must not
# appear here: the function reports gaps among the days it is given.
CALENDAR = [
    "2026-09-23",
    "2026-09-24",
    "2026-09-28",
    "2026-09-29",
]


def test_a_stale_day_between_two_settled_days_is_reported(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-23", "settled")
    record_run(conn, "2026-09-24", "skipped_stale")
    record_run(conn, "2026-09-29", "settled")

    gaps = settlement_gaps(conn, CALENDAR)

    assert [gap["date"] for gap in gaps] == ["2026-09-24", "2026-09-28"]
    assert gaps[0]["status"] == "skipped_stale"
    assert gaps[1]["status"] == "no_run_record"


def test_no_gaps_when_every_calendar_day_is_settled(conn: sqlite3.Connection) -> None:
    for day in CALENDAR:
        record_run(conn, day, "settled")

    assert settlement_gaps(conn, CALENDAR) == []


def test_nothing_settled_means_nothing_to_report(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-24", "skipped_stale")

    assert settlement_gaps(conn, CALENDAR) == []


def test_days_outside_the_settled_range_are_not_gaps(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-24", "settled")
    record_run(conn, "2026-09-28", "settled")

    assert settlement_gaps(conn, CALENDAR) == []


def test_a_day_after_the_anchor_is_not_a_gap(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-23", "settled")
    record_run(conn, "2026-09-24", "settled")
    record_run(conn, "2026-09-28", "skipped_stale")

    assert [gap["date"] for gap in settlement_gaps(conn, CALENDAR)] == []


def test_the_gap_scan_degrades_to_empty_when_the_calendar_fails(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ashare_lab.paper import pipeline

    record_run(conn, "2026-09-23", "settled")
    record_run(conn, "2026-09-24", "skipped_stale")
    record_run(conn, "2026-09-29", "settled")

    def explode(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr("ashare_lab.data.calendar.trading_days_between", explode)

    with caplog.at_level("WARNING"):
        assert pipeline._status_gaps(conn) == []

    # The empty list alone would also come from a silent import failure; the
    # warning proves the guarded path is the one that ran.
    assert any("settlement gap scan skipped" in record.message for record in caplog.records)


def test_the_gap_scan_degrades_to_empty_when_a_date_is_unreadable(
    conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    from ashare_lab.paper import pipeline

    record_run(conn, "2026-09-23", "settled")
    conn.execute("UPDATE runs SET trade_date = 'not-a-date' WHERE trade_date = '2026-09-23'")
    conn.commit()

    with caplog.at_level("WARNING"):
        assert pipeline._status_gaps(conn) == []

    assert any("settlement gap scan skipped" in record.message for record in caplog.records)


def test_a_single_settled_day_has_no_gaps(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-24", "settled")

    assert settlement_gaps(conn, CALENDAR) == []


def test_an_empty_calendar_reports_no_gaps(conn: sqlite3.Connection) -> None:
    record_run(conn, "2026-09-23", "settled")
    record_run(conn, "2026-09-29", "settled")

    assert settlement_gaps(conn, []) == []


def test_get_status_exposes_the_gaps_it_computed(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from ashare_lab.paper import pipeline

    record_run(conn, "2026-09-23", "settled")
    record_run(conn, "2026-09-24", "skipped_stale")
    record_run(conn, "2026-09-29", "settled")

    monkeypatch.setattr(pipeline, "load_config", lambda: {"paper": {"db_path": "paper.db"}})
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", Path("/nonexistent"))
    monkeypatch.setattr(pipeline, "get_connection", lambda _path: conn)
    monkeypatch.setattr(pipeline, "init_schema", lambda _conn: None)
    monkeypatch.setattr(
        "ashare_lab.data.calendar.trading_days_between",
        lambda start, end: [__import__("datetime").date(2026, 9, 24)],
    )

    status = pipeline.get_status()

    assert status["gap_count"] == len(status["gaps"]) == 1
    assert status["gaps"][0]["date"] == "2026-09-24"
    assert status["gaps"][0]["status"] == "skipped_stale"


def test_the_gap_scan_degrades_when_the_runs_table_cannot_be_read(
    conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    from ashare_lab.paper import pipeline

    conn.execute("DROP TABLE runs")
    conn.commit()

    with caplog.at_level("WARNING"):
        assert pipeline._status_gaps(conn) == []

    assert any("settlement gap scan skipped" in record.message for record in caplog.records)
