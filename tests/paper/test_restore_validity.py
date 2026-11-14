"""A restore candidate is judged by the history it holds, not by its name.

find_best_backup validated candidates with PRAGMA integrity_check, which
answers "is this a well-formed SQLite file" and nothing about whether the file
holds a ledger, then ordered them by filename.  A snapshot of the wrong file
lands under a perfectly good name, so on the live host the chosen backup was a
143KB file with zero nav rows sitting above a 516KB file with 75 -- and
restore would have reported success after emptying the ledger.
"""
import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper.ledger import (
    _backup_data_extent,
    _rank_restore_candidates,
    check_db_integrity,
    find_best_backup,
    init_schema,
    record_nav,
    record_run,
)


def _make_backup(path: Path, nav_dates=(), settled=None):
    """Create a structurally valid backup holding the given nav history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    init_schema(conn)
    for i, d in enumerate(nav_dates):
        record_nav(
            conn,
            trade_date=d,
            cash=1000.0,
            market_value=0.0,
            total_nav=1000.0 + i,
            pre_trade_nav=None,
            post_trade_nav=None,
            benchmark_csi300=None,
            benchmark_csi1000=None,
        )
    if settled:
        record_run(conn, settled, "settled")
    conn.commit()
    conn.close()
    return path


class TestDataExtent:
    def test_reports_the_latest_nav_date(self, tmp_path):
        p = _make_backup(tmp_path / "a.db", ["2026-10-20", "2026-10-26"])
        assert _backup_data_extent(p) == "2026-10-26"

    def test_an_empty_ledger_has_no_extent(self, tmp_path):
        p = _make_backup(tmp_path / "b.db")
        assert _backup_data_extent(p) == ""

    def test_a_file_without_the_table_has_no_extent(self, tmp_path):
        p = tmp_path / "c.db"
        conn = sqlite3.connect(p)
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
        conn.close()
        assert _backup_data_extent(p) == ""

    def test_a_non_database_has_no_extent(self, tmp_path):
        p = tmp_path / "d.db"
        p.write_bytes(b"this is not a database")
        assert _backup_data_extent(p) == ""

    def test_a_missing_file_has_no_extent(self, tmp_path):
        assert _backup_data_extent(tmp_path / "absent.db") == ""

    @pytest.mark.parametrize("oddity", ["odd#name.db", "odd%name.db",
                                        "odd&name.db", "odd+name.db"])
    def test_an_awkward_filename_still_reads(self, tmp_path, oddity):
        # Characters that mean something in a URI but are legal in a filename.
        # '#' truncates a hand-built file: URI at the fragment separator the
        # same way '?' does at the query separator, making a populated backup
        # look empty -- the direction that disarms the ranking. '%' is the
        # escape character itself, so a literal one has to survive encoding.
        # '?' is deliberately absent: Windows rejects it outright, and a test
        # that can only run on one platform covers nothing on the other.
        p = _make_backup(tmp_path / oddity, ["2026-10-26"])
        assert _backup_data_extent(p) == "2026-10-26"


class TestRanking:
    def test_more_history_outranks_a_higher_sorting_name(self, tmp_path):
        # The live hazard, reproduced: the empty file sorts first by name.
        empty = _make_backup(tmp_path / "paper_2026-10-28.db")
        full = _make_backup(
            tmp_path / "paper_2026-10-26.db", ["2026-10-22", "2026-10-26"]
        )
        # the complete order, not just the winner: a regression that puts the
        # right file first but scrambles the rest is still a regression
        assert _rank_restore_candidates([empty, full]) == [
            (full, "2026-10-26"),
            (empty, ""),
        ]

    def test_newer_history_wins_between_two_populated_backups(self, tmp_path):
        older = _make_backup(tmp_path / "paper_2026-10-20.db", ["2026-10-20"])
        newer = _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        assert _rank_restore_candidates([older, newer]) == [
            (newer, "2026-10-26"),
            (older, "2026-10-20"),
        ]

    def test_equal_history_falls_back_to_filename(self, tmp_path):
        a = _make_backup(tmp_path / "paper_2026-10-25.db", ["2026-10-26"])
        b = _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        assert _rank_restore_candidates([a, b]) == [
            (b, "2026-10-26"),
            (a, "2026-10-26"),
        ]

    def test_empty_backups_keep_filename_order_among_themselves(self, tmp_path):
        a = _make_backup(tmp_path / "paper_2026-10-25.db")
        b = _make_backup(tmp_path / "paper_2026-10-26.db")
        assert _rank_restore_candidates([a, b]) == [(b, ""), (a, "")]


class TestDegradedRankingIsAnnounced:
    """Falling back to filename order is the original defect; say so.

    Every lookup returning nothing means the ordering carries no information,
    and the one realistic cause -- a renamed ledger table -- produces no other
    symptom. The behaviour is unchanged, only now it is visible.
    """

    def test_warns_when_no_candidate_reports_history(self, tmp_path, caplog):
        a = _make_backup(tmp_path / "paper_2026-10-25.db")
        b = _make_backup(tmp_path / "paper_2026-10-26.db")
        with caplog.at_level("WARNING"):
            _rank_restore_candidates([a, b])
        assert "ranking 2 candidates by filename only" in caplog.text

    def test_stays_quiet_when_a_candidate_has_history(self, tmp_path, caplog):
        a = _make_backup(tmp_path / "paper_2026-10-25.db")
        b = _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        with caplog.at_level("WARNING"):
            _rank_restore_candidates([a, b])
        assert "filename only" not in caplog.text

    def test_stays_quiet_for_a_single_candidate(self, tmp_path, caplog):
        # One empty backup is not a degraded ordering, it is the only choice.
        only = _make_backup(tmp_path / "paper_2026-10-26.db")
        with caplog.at_level("WARNING"):
            _rank_restore_candidates([only])
        assert "filename only" not in caplog.text

    def test_warns_when_the_ledger_table_is_renamed(self, tmp_path, caplog):
        # The realistic trigger, driven for real rather than described: a
        # schema rename makes every lookup fail and the ranking lose its basis.
        import sqlite3

        for name in ("paper_2026-10-25.db", "paper_2026-10-26.db"):
            path = tmp_path / name
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE nav_renamed (trade_date TEXT)")
            conn.execute("INSERT INTO nav_renamed VALUES ('2026-10-26')")
            conn.commit()
            conn.close()
        with caplog.at_level("WARNING"):
            _rank_restore_candidates(list(tmp_path.glob("paper_*.db")))
        assert "ranking 2 candidates by filename only" in caplog.text


class TestFindBestBackup:
    def test_picks_the_populated_backup_over_the_empty_newer_one(self, tmp_path):
        _make_backup(tmp_path / "paper_2026-10-27.db")
        _make_backup(tmp_path / "paper_2026-10-28.db")
        full = _make_backup(
            tmp_path / "paper_2026-10-26.db", ["2026-10-22", "2026-10-26"]
        )
        assert find_best_backup(tmp_path) == full

    def test_a_single_empty_backup_is_still_returned(self, tmp_path):
        # A genuinely fresh ledger has no rows yet; narrowing what counts as
        # restorable would break recovery on a new deployment.
        only = _make_backup(tmp_path / "paper_2026-10-28.db")
        assert find_best_backup(tmp_path) == only

    def test_an_explicit_target_date_still_wins(self, tmp_path):
        _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        asked = _make_backup(tmp_path / "paper_2026-10-22.db", ["2026-10-22"])
        assert find_best_backup(tmp_path, target_date="2026-10-22") == asked

    def test_no_backups_at_all_returns_none(self, tmp_path):
        assert find_best_backup(tmp_path) is None

    def test_a_corrupt_candidate_does_not_block_a_good_one(self, tmp_path):
        bad = tmp_path / "paper_2026-10-28.db"
        bad.write_bytes(b"not a database at all")
        good = _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        # state the assumption rather than relying on it: the corrupt file has
        # to be reported unhealthy, and the check has to return rather than
        # raise, or this test would pass for an unrelated reason
        assert check_db_integrity(bad).is_healthy is False
        assert find_best_backup(tmp_path) == good

    def test_golden_is_ranked_by_history_too(self, tmp_path):
        golden = tmp_path / "golden"
        _make_backup(golden / "2026-10.db")
        full = _make_backup(golden / "2026-09.db", ["2026-09-30"])
        assert find_best_backup(tmp_path) == full

    def test_an_existing_empty_target_is_still_honoured(self, tmp_path):
        # An explicit target is a request, not a guess: if the operator names a
        # date and that file is readable, they get it even though the daily
        # scan would have preferred the populated one.
        asked = _make_backup(tmp_path / "paper_2026-10-28.db")
        _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        assert find_best_backup(tmp_path, target_date="2026-10-28") == asked

    def test_an_absent_target_falls_back_to_ranking(self, tmp_path):
        _make_backup(tmp_path / "paper_2026-10-28.db")
        full = _make_backup(tmp_path / "paper_2026-10-26.db", ["2026-10-26"])
        assert find_best_backup(tmp_path, target_date="2026-12-31") == full
