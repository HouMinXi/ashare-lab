"""Attribution arithmetic, on hand-built NAV rows.

The numbers matter more than the printing: the passive column exists to separate
"the picks were wrong" from "the book was mostly in cash", and a sign error
there inverts the conclusion.
"""

from __future__ import annotations

import pytest

from ashare_lab.research.attribution import compute_attribution, format_table, load_window


def _row(date, nav, cash, bench):
    return {"trade_date": date, "total_nav": nav, "cash": cash, "benchmark_csi1000": bench}


def test_a_fully_invested_lookalike_explains_the_gap():
    """The book earns half the index because it holds half in cash.

    Cash is half of NAV on every day, so the average exposure is exactly 0.50
    and the passive comparison is the index return halved. The pick skill shows
    up as zero excess against that column, not against the raw index.
    """
    rows = [
        _row("2026-08-03", 100_000.0, 50_000.0, 1000.0),
        _row("2026-08-04", 105_000.0, 52_500.0, 1100.0),
        _row("2026-08-05", 110_000.0, 55_000.0, 1200.0),
    ]

    stats = compute_attribution(rows)

    assert stats["book_return"] == pytest.approx(0.10)
    assert stats["benchmark_return"] == pytest.approx(0.20)
    assert stats["avg_exposure"] == pytest.approx(0.50)
    assert stats["passive_return"] == pytest.approx(0.10)
    assert stats["excess_vs_passive_pp"] == pytest.approx(0.0, abs=1e-9)
    assert stats["excess_vs_benchmark_pp"] == pytest.approx(-10.0, abs=1e-9)


def test_a_book_that_beats_its_exposure_shows_positive_alpha():
    """Half invested, up 20% while the index gains 10%: the picks did the work."""
    rows = [
        _row("2026-08-03", 100_000.0, 50_000.0, 1000.0),
        _row("2026-08-04", 120_000.0, 60_000.0, 1100.0),
    ]

    stats = compute_attribution(rows)

    assert stats["book_return"] == pytest.approx(0.20)
    assert stats["benchmark_return"] == pytest.approx(0.10)
    assert stats["avg_exposure"] == pytest.approx(0.50)
    assert stats["passive_return"] == pytest.approx(0.05)
    assert stats["excess_vs_passive_pp"] == pytest.approx(15.0)


def test_a_window_without_a_benchmark_reports_the_book_but_not_the_comparison():
    rows = [
        _row("2026-08-03", 100_000.0, 0.0, None),
        _row("2026-08-04", 101_000.0, 0.0, None),
    ]

    stats = compute_attribution(rows)

    assert stats["book_return"] == pytest.approx(0.01)
    assert stats["benchmark_return"] is None
    assert stats["passive_return"] is None
    assert stats["excess_vs_passive_pp"] is None


def test_one_row_is_not_a_return():
    stats = compute_attribution([_row("2026-08-03", 100_000.0, 0.0, 1000.0)])

    assert stats["days"] == 1
    assert stats["book_return"] is None
    assert "nothing to compare" in format_table(stats)


def test_an_empty_window_does_not_crash_the_formatter():
    stats = compute_attribution([])

    assert stats["days"] == 0
    assert "nothing to compare" in format_table(stats)


def test_load_window_reads_the_ledger_in_date_order(tmp_path):
    import sqlite3

    from ashare_lab.paper.ledger import init_schema

    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    for date, nav in (("2026-08-05", 3.0), ("2026-08-03", 1.0), ("2026-08-04", 2.0)):
        conn.execute(
            "INSERT INTO nav (trade_date, cash, market_value, total_nav, benchmark_csi1000) "
            "VALUES (?, 0, 0, ?, 1000)",
            (date, nav),
        )
    conn.commit()

    rows = load_window(conn, "2026-08-04", "2026-08-05")
    conn.close()

    assert [r["trade_date"] for r in rows] == ["2026-08-04", "2026-08-05"]
