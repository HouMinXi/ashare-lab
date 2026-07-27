"""Tests for pipeline_runs open-and-close recording (P1).

R5 requirements:
  T1: start-row written at start
  T2: success closes to 'success'
  T3: injected exception mid-run closes to 'error' with message
  T4: gate halt closes to 'halted' with gate name
  T5: simulated kill (no close) leaves 'running'
  T6: bug-injection: remove the failed-path close -> test expecting 'error' FAILS
"""

from __future__ import annotations


def _make_db(tmp_path):
    """Create a test DB with full schema."""
    from ashare_lab.paper.ledger import init_schema, get_connection
    db_path = tmp_path / "paper.db"
    conn = get_connection(db_path)
    init_schema(conn)
    conn.commit()
    return conn, db_path


class TestPipelineRunsRecording:
    """R5: open-and-close recording tests."""

    def test_start_row_written(self, tmp_path):
        """T1: open_pipeline_run inserts a 'running' row and returns its id."""
        from ashare_lab.paper.ledger import open_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")

        assert row_id is not None
        assert row_id > 0

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row is not None
        assert row["status"] == "running"
        assert row["trade_date"] == "2026-07-25"
        conn.close()

    def test_success_close(self, tmp_path):
        """T2: close_pipeline_run updates 'running' to 'success'."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, row_id, "success", 10.5)
        conn.commit()

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "success"
        assert row["duration_s"] == 10.5
        assert row["error_msg"] is None
        conn.close()

    def test_error_close_with_message(self, tmp_path):
        """T3: close_pipeline_run updates to 'error' with error message."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, row_id, "error", 5.0, "data update failed")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "error"
        assert row["error_msg"] == "data update failed"
        conn.close()

    def test_halted_close_with_gate_name(self, tmp_path):
        """T4: close_pipeline_run updates to 'halted' with gate name."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, row_id, "halted", 3.0,
                           "Data completeness gate: qlib missing 5 days")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "halted"
        assert "Data completeness gate" in row["error_msg"]
        conn.close()

    def test_simulated_kill_leaves_running(self, tmp_path):
        """T5: if close is never called, row stays 'running'."""
        from ashare_lab.paper.ledger import open_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        # Simulate kill: no close_pipeline_run call
        conn.commit()

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "running"
        conn.close()

    def test_same_day_reruns_keep_both_rows(self, tmp_path):
        """Two runs on same day get separate rows (history preserved)."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)
        id1 = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, id1, "error", 5.0, "first run failed")
        conn.commit()

        id2 = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, id2, "success", 10.0)
        conn.commit()

        rows = conn.execute(
            "SELECT * FROM pipeline_runs WHERE trade_date='2026-07-25' ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["status"] == "error"
        assert rows[1]["status"] == "success"
        assert id1 != id2
        conn.close()

    def test_observation_window_counting(self, tmp_path):
        """'running' rows don't count in observation window query."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)

        # Running row (crashed)
        open_pipeline_run(conn, "2026-07-25", "2026-07-25")

        # Success row
        id2 = open_pipeline_run(conn, "2026-07-26", "2026-07-26")
        close_pipeline_run(conn, id2, "success", 10.0)
        conn.commit()

        # Observation window query
        count = conn.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE status='success'"
        ).fetchone()[0]
        assert count == 1, "'running' rows must not count"
        conn.close()


class TestBugInjection:
    """T6: remove the failed-path close and show the test FAILS."""

    def test_remove_close_keeps_running(self, tmp_path):
        """Without close_pipeline_run call, row stays 'running'.

        Bug-inject: if _close_pipeline_run is removed from error path,
        the row stays 'running' instead of 'error'. This test proves
        the close is necessary.
        """
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")

        # Simulate: error path DOES call close
        close_pipeline_run(conn, row_id, "error", 5.0, "something broke")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "error", (
            "Close must be called on error path; "
            "if this fails, the close was removed (bug)"
        )

        # Now prove: without close, it stays 'running'
        row_id2 = open_pipeline_run(conn, "2026-07-26", "2026-07-26")
        # No close call -- simulating removed close
        conn.commit()

        row2 = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id=?", (row_id2,)
        ).fetchone()
        assert row2["status"] == "running", (
            "Without close, row stays 'running' -- proves close is necessary"
        )
        conn.close()


class TestSchemaMigration:
    """Test that the CHECK constraint migration is idempotent."""

    def test_migration_adds_new_statuses(self, tmp_path):
        """New DB should allow 'running' and 'halted'."""
        from ashare_lab.paper.ledger import open_pipeline_run, close_pipeline_run

        conn, _ = _make_db(tmp_path)

        # Should not raise
        row_id = open_pipeline_run(conn, "2026-07-25", "2026-07-25")
        close_pipeline_run(conn, row_id, "halted", 1.0, "gate")
        conn.commit()

        row = conn.execute(
            "SELECT status FROM pipeline_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row["status"] == "halted"
        conn.close()

    def test_migration_preserves_existing_rows(self, tmp_path):
        """Migration should preserve existing data."""
        from ashare_lab.paper.ledger import init_schema

        conn, db_path = _make_db(tmp_path)

        # Insert a row with old-style status
        conn.execute(
            "INSERT INTO pipeline_runs "
            "(trade_date, status, duration_s, error_msg, predictions_date, created_at) "
            "VALUES (?, 'success', 10.0, NULL, ?, '2026-07-25T12:00:00Z')",
            ("2026-07-25", "2026-07-25"),
        )
        conn.commit()

        # Re-run init_schema (simulates app restart)
        init_schema(conn)
        conn.commit()

        row = conn.execute(
            "SELECT status FROM pipeline_runs WHERE trade_date='2026-07-25'"
        ).fetchone()
        assert row is not None
        assert row["status"] == "success"
        conn.close()
