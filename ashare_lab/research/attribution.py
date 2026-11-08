"""Book versus benchmark versus the same exposure held passively.

The daily report shows the book's return next to CSI1000, which leaves the
obvious question open: when the book underperforms, is that the picks or the
low exposure? A cash-heavy book almost always loses to a fully invested index in
a rising market, and that is a sizing decision, not a stock-picking one.

The passive column re-scales the benchmark by the book's average equity
exposure, so the comparison is like for like. The excess that survives that
adjustment is what the picks contributed.

Run: ashare-attribution --since 2026-08-11 [--db paper.db] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def load_window(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    """NAV rows in [start, end], oldest first."""
    cursor = conn.execute(
        "SELECT trade_date, cash, total_nav, benchmark_csi1000 FROM nav "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start, end),
    )
    return [dict(row) for row in cursor.fetchall()]


def compute_attribution(rows: list[dict]) -> dict:
    """Return the four numbers the comparison needs.

    Empty or single-row windows have no return to speak of, so the caller gets
    None rather than a zero that reads like a flat month.
    """
    stats: dict = {
        "days": len(rows),
        "start": rows[0]["trade_date"] if rows else None,
        "end": rows[-1]["trade_date"] if rows else None,
        "book_return": None,
        "benchmark_return": None,
        "avg_exposure": None,
        "passive_return": None,
        "excess_vs_passive_pp": None,
        "excess_vs_benchmark_pp": None,
    }
    if len(rows) < 2:
        return stats

    first, last = rows[0], rows[-1]
    book_return = last["total_nav"] / first["total_nav"] - 1.0
    stats["book_return"] = book_return

    exposures = [
        (row["total_nav"] - row["cash"]) / row["total_nav"]
        for row in rows
        if row["total_nav"]
    ]
    avg_exposure = sum(exposures) / len(exposures) if exposures else None
    stats["avg_exposure"] = avg_exposure

    bench_first = first.get("benchmark_csi1000")
    bench_last = last.get("benchmark_csi1000")
    if bench_first and bench_last:
        bench_return = bench_last / bench_first - 1.0
        stats["benchmark_return"] = bench_return
        stats["excess_vs_benchmark_pp"] = (book_return - bench_return) * 100.0
        if avg_exposure is not None:
            passive = avg_exposure * bench_return
            stats["passive_return"] = passive
            stats["excess_vs_passive_pp"] = (book_return - passive) * 100.0
    return stats


def format_table(stats: dict) -> str:
    if stats["book_return"] is None:
        return f"{stats['days']} sessions in window: nothing to compare yet"

    def pct(value):
        return "n/a" if value is None else f"{value * 100:+.2f}%"

    def pp(value):
        return "n/a" if value is None else f"{value:+.2f} pp"

    lines = [
        f"{stats['start']} .. {stats['end']}  ({stats['days']} sessions)",
        f"  book             {pct(stats['book_return'])}",
        f"  CSI1000          {pct(stats['benchmark_return'])}",
        f"  avg exposure     {pct(stats['avg_exposure'])}",
        f"  passive at that  {pct(stats['passive_return'])}",
        f"  vs benchmark     {pp(stats['excess_vs_benchmark_pp'])}",
        f"  vs passive       {pp(stats['excess_vs_passive_pp'])}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="paper.db")
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", default="9999-12-31")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.db)
    if not path.exists():
        parser.error(f"no ledger at {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = load_window(conn, args.since, args.until)
    finally:
        conn.close()

    stats = compute_attribution(rows)
    print(json.dumps(stats, indent=2) if args.json else format_table(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
