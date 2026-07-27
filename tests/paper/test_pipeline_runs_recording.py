"""Tests for pipeline_runs open-and-close recording (P1).

Unit tests (ledger helpers):
  T1-T7: open/close semantics, same-day reruns, observation window
  T8-T10: schema migration

Integration tests (through run_daily):
  T11: data-completeness gate -> 'halted'
  T12: price-sanity gate -> 'halted'
  T13: settle error -> 'error'
  T14: uncaught exception -> 'error' with real message
  T15: close-once: gate halt not overwritten by outer handler

Bug-injection: for each instrumented path, delete the close call,
show the test RED, restore, show GREEN.
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


class TestLedgerCloseSemantics:
    """T8: ledger-level close behavior (not a bug-injection proof).

    Integration tests in TestIntegration* prove the run_daily wiring.
    """

    def test_close_changes_status(self, tmp_path):
        """Close changes 'running' to 'error'; no-close leaves 'running'."""
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


# ---------------------------------------------------------------------------
# Integration tests: through run_daily (adopted from PM's test_pm_verify_p1.py)
# ---------------------------------------------------------------------------

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import _pipeline_patches  # noqa: E402

_MOD = "ashare_lab.paper.pipeline"


@pytest.fixture()
def int_db(tmp_path):
    """Create a test DB for integration tests."""
    from ashare_lab.paper.ledger import get_connection, init_schema
    p = tmp_path / "paper.db"
    conn = get_connection(p)
    init_schema(conn)
    conn.close()
    return p


@pytest.fixture()
def int_config(int_db):
    return {
        "paper": {
            "db_path": str(int_db),
            "carry_days": 3,
            "volume_participation_pct": 0.05,
            "topk": 15,
            "n_drop": 1,
            "initial_cash": 300_000,
            "backup_retention_days": 7,
            "slippage": 0.001,
            "listing_min_days": 60,
            "liquidity_min_turnover": 50_000_000,
            "predictions_dir": "predictions",
            "min_signal_coverage": 100,
            "risk": {
                "drawdown_hard": 0.15,
                "daily_loss": 0.03,
                "concentration": 0.15,
                "market_regime_decline": 0.08,
                "market_regime_days": 10,
            },
        },
    }


def _final_rows(db_path, trade_date="2025-06-20"):
    """Get pipeline_runs rows for a trade date."""
    from ashare_lab.paper.ledger import get_connection
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT status, error_msg FROM pipeline_runs WHERE trade_date=? ORDER BY id",
        (trade_date,),
    ).fetchall()
    conn.close()
    return rows


class TestIntegrationGateHalt:
    """T11/T12: gate halt through run_daily -> final row 'halted'."""

    def test_data_gate_halt_records_halted(self, int_db, int_config):
        """T11: data-completeness gate -> 'halted' with gate name."""
        extra = {
            "gate_data": patch(
                f"{_MOD}._gate_data_completeness",
                side_effect=RuntimeError("Data completeness gate: qlib missing 5 days"),
            ),
        }
        with _pipeline_patches(int_db, int_config, extra_patches=extra):
            from ashare_lab.paper.pipeline import run_daily
            try:
                run_daily("2025-06-20")
            except RuntimeError:
                pass

        rows = _final_rows(int_db)
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
        assert rows[0]["status"] == "halted", (
            f"gate halt must leave 'halted', got '{rows[0]['status']}' "
            f"(error_msg={rows[0]['error_msg']!r})"
        )
        assert "Data completeness gate" in (rows[0]["error_msg"] or "")

    def test_price_gate_halt_records_halted(self, int_db, int_config):
        """T12: price-sanity gate -> 'halted' with gate name."""
        extra = {
            "gate_price": patch(
                f"{_MOD}._gate_price_sanity",
                side_effect=RuntimeError("Price sanity gate: 3 anomalies"),
            ),
        }
        with _pipeline_patches(int_db, int_config, extra_patches=extra):
            from ashare_lab.paper.pipeline import run_daily
            try:
                run_daily("2025-06-20")
            except RuntimeError:
                pass

        rows = _final_rows(int_db)
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
        assert rows[0]["status"] == "halted", (
            f"gate halt must leave 'halted', got '{rows[0]['status']}' "
            f"(error_msg={rows[0]['error_msg']!r})"
        )


class TestIntegrationSettleError:
    """T13: settle error (rc=2) through run_daily -> 'error'."""

    def test_settle_error_records_error(self, int_db, int_config):
        """T13: settle returning 2 -> 'error' with message."""
        from ashare_lab.paper.engine import SettleResult
        settle_result = SettleResult()
        settle_result.pre_trade_nav = 300_000
        settle_result.post_trade_nav = 300_000
        settle_result.cash = 300_000

        def settle_side_effect(*args, **kwargs):
            conn = args[0]
            conn.execute(
                "INSERT INTO pipeline_runs (trade_date, status, duration_s, error_msg, "
                "predictions_date, created_at) VALUES (?, 'error', 0, 'test', NULL, '')",
                ("2025-06-20",),
            )
            return settle_result

        extra = {
            "settle_day": patch(
                f"{_MOD}.settle_day",
                side_effect=settle_side_effect,
            ),
        }
        with _pipeline_patches(int_db, int_config, extra_patches=extra):
            from ashare_lab.paper.pipeline import run_daily
            # settle_day mock doesn't return rc=2 through the normal path
            # Instead, patch _step8_settle to return 2
            with patch(f"{_MOD}._step8_settle", return_value=2):
                result = run_daily("2025-06-20")

        assert result == 2
        rows = _final_rows(int_db)
        assert any(r["status"] == "error" for r in rows), (
            f"settle error must leave 'error', got {rows}"
        )


class TestIntegrationUncaughtException:
    """T14: uncaught exception through run_daily -> 'error' with real message."""

    def test_uncaught_exception_records_error(self, int_db, int_config):
        """T14: exception in _step9 -> 'error' with real exception message."""
        extra = {
            "risk": patch(
                f"{_MOD}._step9_risk_checks",
                side_effect=ValueError("boom: risk engine crashed"),
            ),
        }
        with _pipeline_patches(int_db, int_config, extra_patches=extra):
            from ashare_lab.paper.pipeline import run_daily
            try:
                run_daily("2025-06-20")
            except ValueError:
                pass

        rows = _final_rows(int_db)
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
        assert rows[0]["status"] == "error", (
            f"uncaught exception must leave 'error', got '{rows[0]['status']}'"
        )
        assert "boom" in (rows[0]["error_msg"] or ""), (
            f"error_msg must contain real exception, got: {rows[0]['error_msg']!r}"
        )


class TestCloseOnceSemantics:
    """T15: close-once -- gate halt not overwritten by outer handler."""

    def test_gate_halt_not_overwritten(self, int_db, int_config):
        """T15: inner 'halted' close survives outer 'error' handler.

        This is the F1 fix verification: _close_pipeline_run sets
        pipeline_run_id=None after close, so the outer except block
        is a no-op.
        """
        extra = {
            "gate_data": patch(
                f"{_MOD}._gate_data_completeness",
                side_effect=RuntimeError("Data completeness gate: qlib missing 5 days"),
            ),
        }
        with _pipeline_patches(int_db, int_config, extra_patches=extra):
            from ashare_lab.paper.pipeline import run_daily
            try:
                run_daily("2025-06-20")
            except RuntimeError:
                pass

        rows = _final_rows(int_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "halted", (
            f"close-once: gate halt must survive outer handler, "
            f"got '{rows[0]['status']}'"
        )
