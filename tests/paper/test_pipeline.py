"""Tests for the daily pipeline orchestrator."""

from __future__ import annotations

import contextlib
import datetime as dt
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.paper.ledger import (
    get_connection,
    init_schema,
    record_nav,
    record_run,
)
from ashare_lab.paper.risk import RiskCheckResult

_MOD = "ashare_lab.paper.pipeline"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "paper.db"
    conn = get_connection(p)
    init_schema(conn)
    conn.close()
    return p


@pytest.fixture()
def base_config(db_path: Path) -> dict:
    return {
        "paper": {
            "db_path": str(db_path),
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
                "trailing_stop": 0.20,
                "trailing_cooldown_days": 10,
                "industry_cap": 0.30,
                "soft_drawdown": 0.10,
                "soft_drawdown_recovery": 0.95,
                "default_topk": 15,
                "reduced_topk": 7,
            },
        },
        "cost_model": {"risk_degree": 0.95},
        "universe": {"exclude_close_above_cny": 300},
    }


def _risk_result(**overrides) -> RiskCheckResult:
    defaults = dict(
        buying_halted=False,
        forced_sells={},
        blocked_industries=set(),
        blocked_rebuys=set(),
        topk_override=None,
        cooldown_entries={},
        suspension_risk={},
    )
    defaults.update(overrides)
    return RiskCheckResult(**defaults)


def _settle_mock(cash=300_000):
    return MagicMock(
        carries_to_bump=[], carries_suspended=[],
        cash=cash, pre_trade_nav=300_000, post_trade_nav=cash,
    )


def _qlib_mock():
    """Return a mock that behaves like qlib.data.D."""
    m = MagicMock()
    m.list_instruments.return_value = []
    empty_df = MagicMock(empty=True, __bool__=lambda s: False)
    m.features.return_value = empty_df
    return m


@contextlib.contextmanager
def _pipeline_patches(
    db_path, config, *,
    risk_result=None,
    settle_side_effect=None,
    signals_side_effect=None,
    signals_return=None,
    positions=None,
    cooldowns=None,
    cooldown_managed=None,
    extra_patches=None,
):
    """Apply all common pipeline patches using ExitStack.

    Patches deferred imports (qlib, daily_refresh, baostock) via
    sys.modules injection so they resolve correctly inside function
    bodies.
    """
    if risk_result is None:
        risk_result = _risk_result()
    settle_ret = _settle_mock()
    if signals_return is None:
        signals_return = {}
    if positions is None:
        positions = {}
    if cooldowns is None:
        cooldowns = {}
    if cooldown_managed is None:
        cooldown_managed = cooldowns
    if extra_patches is None:
        extra_patches = {}

    pred_dir = db_path.parent / "predictions"
    pred_dir.mkdir(exist_ok=True)
    (pred_dir / "2025-06-20.parquet").touch()

    # Prepare mock for qlib deferred import
    qlib_d_mock = _qlib_mock()
    qlib_data_mod = MagicMock()
    qlib_data_mod.D = qlib_d_mock

    # Prepare mock for daily_refresh deferred import
    update_mod = MagicMock()
    update_mod.daily_refresh = MagicMock(return_value=0)

    # Build settle and signal mocks
    if settle_side_effect is not None:
        settle_patch_args = {"side_effect": settle_side_effect}
    else:
        settle_patch_args = {"return_value": settle_ret}

    if signals_side_effect is not None:
        signal_patch_args = {"side_effect": signals_side_effect}
    else:
        signal_patch_args = {"return_value": signals_return}

    patches = {
        "load_config": patch(f"{_MOD}.load_config", return_value=config),
        "PROJECT_ROOT": patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
        "is_day_settled": patch(
            f"{_MOD}.is_day_settled", return_value=False
        ),
        "get_latest_positions": patch(
            f"{_MOD}.get_latest_positions", return_value=positions
        ),
        "get_latest_cash": patch(
            f"{_MOD}.get_latest_cash", return_value=300_000
        ),
        "get_cooldowns": patch(
            f"{_MOD}.get_cooldowns", return_value=cooldowns
        ),
        "manage_trailing_cooldown": patch(
            f"{_MOD}.manage_trailing_cooldown",
            return_value=cooldown_managed,
        ),
        "delete_expired_cooldowns": patch(
            f"{_MOD}.delete_expired_cooldowns"
        ),
        "ipo_calendar": patch(
            f"{_MOD}._fetch_ipo_calendar", return_value=[]
        ),
        "benchmarks": patch(
            f"{_MOD}._fetch_benchmark_closes",
            return_value={"csi300": 4000.0, "csi1000": 6000.0},
        ),
        "settle_day": patch(f"{_MOD}.settle_day", **settle_patch_args),
        "generate_signals": patch(
            f"{_MOD}.generate_signals", **signal_patch_args
        ),
        "run_all_risk_checks": patch(
            f"{_MOD}.run_all_risk_checks", return_value=risk_result
        ),
        "compute_nav": patch(f"{_MOD}.compute_nav", return_value=300_000),
        "hot_backup_with_integrity": patch(f"{_MOD}.hot_backup_with_integrity"),
        "create_golden_backup": patch(f"{_MOD}.create_golden_backup"),
        "cleanup_old_backups": patch(f"{_MOD}.cleanup_old_backups"),
        "PREDICTIONS_DIR": patch(f"{_MOD}.PREDICTIONS_DIR", pred_dir),
        "latest_trading_day": patch(
            f"{_MOD}.latest_trading_day",
            return_value=dt.date(2025, 6, 20),
        ),
        "next_trading_day": patch(
            f"{_MOD}.next_trading_day",
            return_value=dt.date(2025, 6, 23),
        ),
        "previous_trading_day": patch(
            f"{_MOD}.previous_trading_day",
            return_value=dt.date(2025, 6, 19),
        ),
        "bump_carry_days": patch(f"{_MOD}.bump_carry_days"),
        "bump_suspension_carry_days": patch(f"{_MOD}.bump_suspension_carry_days"),
        # Patch deferred imports via sys.modules
        "qlib_data": patch.dict(
            sys.modules,
            {"qlib": MagicMock(), "qlib.data": qlib_data_mod},
        ),
        "baostock": patch.dict(
            sys.modules,
            {"baostock": MagicMock()},
        ),
        "update_mod": patch.dict(
            sys.modules,
            {"ashare_lab.data.update": update_mod},
        ),
    }
    patches.update(extra_patches)

    mocks = {}
    stack = contextlib.ExitStack()
    try:
        for name, p in patches.items():
            mocks[name] = stack.enter_context(p)
        # Expose useful inner mocks
        mocks["qlib_D"] = qlib_d_mock
        mocks["daily_refresh_fn"] = update_mod.daily_refresh
        yield mocks
    finally:
        stack.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestIdempotentSkip:
    """D-02: run_daily returns 0 when the day is already settled."""

    def test_already_settled_returns_0(
        self, db_path: Path, base_config: dict
    ) -> None:
        conn = get_connection(db_path)
        record_run(conn, "2025-06-20", "settled")
        conn.commit()
        conn.close()

        with (
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
        ):
            from ashare_lab.paper.pipeline import run_daily
            rc = run_daily("2025-06-20", force=False)
        assert rc == 0


class TestForceReset:
    """run_daily with force=True calls force_reset_day."""

    def test_force_calls_reset(
        self, db_path: Path, base_config: dict
    ) -> None:
        conn = get_connection(db_path)
        record_run(conn, "2025-06-20", "settled")
        conn.commit()
        conn.close()

        extra = {
            "force_reset_day": patch(f"{_MOD}.force_reset_day"),
        }
        with _pipeline_patches(
            db_path, base_config, extra_patches=extra
        ) as mocks:
            from ashare_lab.paper.pipeline import run_daily
            run_daily("2025-06-20", force=True)
        mocks["force_reset_day"].assert_called_once()


class TestLedgerErrorRejection:
    """A day marked ledger_error refuses automatic retry; --force is
    required to proceed past it."""

    def test_ledger_error_day_blocks_auto_retry(
        self, db_path: Path, base_config: dict
    ) -> None:
        conn = get_connection(db_path)
        record_run(conn, "2025-06-20", "ledger_error")
        conn.commit()
        conn.close()

        with (
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
        ):
            from ashare_lab.paper.pipeline import run_daily
            rc = run_daily("2025-06-20", force=False)
        assert rc == 2

        conn2 = get_connection(db_path)
        row = conn2.execute(
            "SELECT status FROM runs WHERE trade_date='2025-06-20'"
        ).fetchone()
        assert row["status"] == "ledger_error"  # not overwritten by the rejection
        pr_row = conn2.execute(
            "SELECT status FROM pipeline_runs WHERE trade_date='2025-06-20' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert pr_row["status"] == "error"
        conn2.close()

    def test_force_bypasses_ledger_error_block(
        self, db_path: Path, base_config: dict
    ) -> None:
        """--force still proceeds past a ledger_error day (the existing
        force path resets the day before continuing)."""
        conn = get_connection(db_path)
        record_run(conn, "2025-06-20", "ledger_error")
        conn.commit()
        conn.close()

        extra = {
            "force_reset_day": patch(f"{_MOD}.force_reset_day"),
        }
        with _pipeline_patches(
            db_path, base_config, extra_patches=extra
        ) as mocks:
            from ashare_lab.paper.pipeline import run_daily
            run_daily("2025-06-20", force=True)
        mocks["force_reset_day"].assert_called_once()


class TestDataStaleSkip:
    """D-27: stale data returns 1 and records skipped_stale."""

    def test_stale_returns_1(
        self, db_path: Path, base_config: dict
    ) -> None:
        # Inject update module mock that returns stale=1
        update_mod = MagicMock()
        update_mod.daily_refresh = MagicMock(return_value=1)
        update_mod._read_calendar_last_date = MagicMock(return_value=None)

        with (
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
            patch(f"{_MOD}.is_day_settled", return_value=False),
            patch.dict(
                sys.modules,
                {"ashare_lab.data.update": update_mod},
            ),
        ):
            from ashare_lab.paper.pipeline import run_daily
            # Use today so the date falls within the 10-day staleness
            # window and daily_refresh is actually invoked.
            today = dt.date.today().isoformat()
            rc = run_daily(today)
        assert rc == 1

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT status FROM runs WHERE trade_date=?", (today,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["status"] == "skipped_stale"


class TestPipelineOrder:
    """D-11-ext: settle is called BEFORE signal generation."""

    def test_settle_before_signal(
        self, db_path: Path, base_config: dict
    ) -> None:
        call_order: list[str] = []

        def track_settle(*a, **kw):
            call_order.append("settle")
            return _settle_mock()

        def track_signals(*a, **kw):
            call_order.append("signal")
            return {}

        with _pipeline_patches(
            db_path, base_config,
            settle_side_effect=track_settle,
            signals_side_effect=track_signals,
        ):
            from ashare_lab.paper.pipeline import run_daily
            rc = run_daily("2025-06-20")
        assert rc == 0
        assert call_order.index("settle") < call_order.index("signal")


class TestRiskAfterSettle:
    """Risk checks run after settle (use post-settle NAV)."""

    def test_risk_after_settle(
        self, db_path: Path, base_config: dict
    ) -> None:
        call_order: list[str] = []

        def track_settle(*a, **kw):
            call_order.append("settle")
            return _settle_mock(cash=290_000)

        def track_risk(*a, **kw):
            call_order.append("risk")
            return _risk_result()

        extra = {
            "run_all_risk_checks": patch(
                f"{_MOD}.run_all_risk_checks", side_effect=track_risk
            ),
        }
        with _pipeline_patches(
            db_path, base_config,
            settle_side_effect=track_settle,
            extra_patches=extra,
        ):
            from ashare_lab.paper.pipeline import run_daily
            run_daily("2025-06-20")
        assert call_order == ["settle", "risk"]


class TestBuyingHalted:
    """Circuit breaker blocks all buys."""

    def test_halted_no_buy_orders(
        self, db_path: Path, base_config: dict
    ) -> None:
        extra = {
            "insert_order": patch(f"{_MOD}.insert_order", return_value=1),
        }
        with _pipeline_patches(
            db_path, base_config,
            risk_result=_risk_result(buying_halted=True),
            signals_return={"SH600001": 0.9, "SH600002": 0.8},
            extra_patches=extra,
        ) as mocks:
            from ashare_lab.paper.pipeline import run_daily
            run_daily("2025-06-20")

        buy_calls = [
            c for c in mocks["insert_order"].call_args_list
            if len(c.args) >= 4 and c.args[3] == "buy"
        ]
        assert len(buy_calls) == 0


class TestBackfill:
    """D-15: run_backfill replays days sequentially."""

    def test_backfill_sequential(self, tmp_path: Path) -> None:
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        for d in ["2025-06-18", "2025-06-19", "2025-06-20"]:
            (pred_dir / f"{d}.parquet").touch()

        called_dates: list[str] = []

        def mock_run_daily(date, force=False, pred_path=None, **kw):
            called_dates.append(date)
            return 0

        with (
            patch(f"{_MOD}.run_daily", side_effect=mock_run_daily),
            patch(f"{_MOD}.PREDICTIONS_DIR", pred_dir),
            patch(
                f"{_MOD}.trading_days_between",
                return_value=[
                    dt.date(2025, 6, 18),
                    dt.date(2025, 6, 19),
                    dt.date(2025, 6, 20),
                ],
            ),
        ):
            from ashare_lab.paper.pipeline import run_backfill
            rc = run_backfill("2025-06-18", "2025-06-20")
        assert rc == 0
        assert called_dates == ["2025-06-18", "2025-06-19", "2025-06-20"]


class TestBackfillForceTeardown:
    """R-14: force tears down ALL days LIFO, then replays forward."""

    def test_force_tears_down_reversed(
        self, db_path: Path, base_config: dict, tmp_path: Path
    ) -> None:
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        for d in ["2025-06-18", "2025-06-19", "2025-06-20"]:
            (pred_dir / f"{d}.parquet").touch()

        reset_calls: list[str] = []

        def mock_reset(conn, d):
            reset_calls.append(d)

        with (
            patch(f"{_MOD}.run_daily", return_value=0),
            patch(f"{_MOD}.PREDICTIONS_DIR", pred_dir),
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
            patch(f"{_MOD}.force_reset_day", side_effect=mock_reset),
            patch(
                f"{_MOD}.trading_days_between",
                return_value=[
                    dt.date(2025, 6, 18),
                    dt.date(2025, 6, 19),
                    dt.date(2025, 6, 20),
                ],
            ),
        ):
            from ashare_lab.paper.pipeline import run_backfill
            rc = run_backfill("2025-06-18", "2025-06-20", force=True)
        assert rc == 0
        assert reset_calls == ["2025-06-20", "2025-06-19", "2025-06-18"]


class TestGetStatus:
    """get_status returns correct structure."""

    def test_empty_db(self, db_path: Path, base_config: dict) -> None:
        with (
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
        ):
            from ashare_lab.paper.pipeline import get_status
            status = get_status()
        assert status["last_trade_date"] is None
        assert status["total_nav"] is None
        assert status["position_count"] == 0

    def test_with_data(self, db_path: Path, base_config: dict) -> None:
        conn = get_connection(db_path)
        record_run(conn, "2025-06-20", "settled")
        record_nav(
            conn, "2025-06-20", 280_000, 20_000, 300_000,
            None, None, 4000.0, 6000.0,
        )
        conn.commit()
        conn.close()

        with (
            patch(f"{_MOD}.load_config", return_value=base_config),
            patch(f"{_MOD}.PROJECT_ROOT", db_path.parent),
        ):
            from ashare_lab.paper.pipeline import get_status
            status = get_status()
        assert status["last_trade_date"] == "2025-06-20"
        assert status["total_nav"] == 300_000
        assert status["run_status"] == "settled"


class TestDateValidation:
    """Invalid date format returns error code 2."""

    def test_bad_date(self) -> None:
        from ashare_lab.paper.pipeline import run_daily
        assert run_daily("not-a-date") == 2

    def test_partial_date(self) -> None:
        from ashare_lab.paper.pipeline import run_daily
        assert run_daily("2025-6-1") == 2


class TestCooldownGuard:
    """R-21: cooldown_until == trade_date is still blocked."""

    def test_equal_date_still_active(
        self, db_path: Path, base_config: dict
    ) -> None:
        cd = {
            "SH600001": {
                "cooldown_until": "2025-06-20",
                "holding_high": 10.0,
            },
        }
        extra = {
            "insert_order": patch(f"{_MOD}.insert_order", return_value=1),
        }
        with _pipeline_patches(
            db_path, base_config,
            cooldowns=cd,
            cooldown_managed=cd,
            risk_result=_risk_result(blocked_rebuys={"SH600001"}),
            signals_return={"SH600001": 0.9},
            extra_patches=extra,
        ) as mocks:
            from ashare_lab.paper.pipeline import run_daily
            run_daily("2025-06-20")

        buy_calls = [
            c for c in mocks["insert_order"].call_args_list
            if len(c.args) >= 4 and c.args[3] == "buy"
            and c.args[2] == "SH600001"
        ]
        assert len(buy_calls) == 0


class TestPipelineRunRecorded:
    """Step 14: run_daily records a row in pipeline_runs."""

    def test_success_run_recorded(
        self, db_path: Path, base_config: dict
    ) -> None:
        with _pipeline_patches(db_path, base_config):
            from ashare_lab.paper.pipeline import run_daily
            rc = run_daily("2025-06-20")
        assert rc == 0

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT status, duration_s, predictions_date "
            "FROM pipeline_runs WHERE trade_date = ?",
            ("2025-06-20",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["status"] == "success"
        assert row["duration_s"] > 0
        assert row["predictions_date"] is not None

    def test_stale_env_records_stale(
        self, db_path: Path, base_config: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASHARE_USE_STALE", "1")
        with _pipeline_patches(db_path, base_config):
            from ashare_lab.paper.pipeline import run_daily
            rc = run_daily("2025-06-20")
        assert rc == 0

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT status FROM pipeline_runs WHERE trade_date = ?",
            ("2025-06-20",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["status"] == "stale"


class TestResolvePredictionFile:
    """_resolve_prediction_file returns Path or None."""

    def test_exists(self, tmp_path: Path) -> None:
        from ashare_lab.paper.pipeline import _resolve_prediction_file
        with patch(f"{_MOD}.PREDICTIONS_DIR", tmp_path):
            (tmp_path / "2025-06-20.parquet").touch()
            result = _resolve_prediction_file("2025-06-20")
        assert result is not None
        assert result.name == "2025-06-20.parquet"

    def test_missing(self, tmp_path: Path) -> None:
        from ashare_lab.paper.pipeline import _resolve_prediction_file
        with patch(f"{_MOD}.PREDICTIONS_DIR", tmp_path):
            result = _resolve_prediction_file("2025-06-20")
        assert result is None


class TestPreCommitNavGate:
    """Pre-commit NAV gate: delete anomalous NAV row before commit."""

    def test_anomalous_nav_deleted_before_commit(self, tmp_path):
        """NAV change >20% -> row deleted, preventing corrupted persistence."""
        from ashare_lab.paper.ledger import get_connection, init_schema, record_nav
        from ashare_lab.paper.pipeline import DailyRunContext, _step12_backup_and_finalize
        from dataclasses import dataclass, field
        from pathlib import Path
        from unittest.mock import patch

        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        init_schema(conn)

        # Insert 2 NAV rows: yesterday normal, today anomalous (+50%)
        record_nav(conn, "2024-01-01", 100000, 200000, 300000, None, None, None, None)
        record_nav(conn, "2024-01-02", 100000, 350000, 450000, None, None, None, None)
        conn.commit()

        # Verify both rows exist
        count = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert count == 2

        ctx = DailyRunContext(
            trade_date="2024-01-02", force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str="",
        )
        ctx.conn = conn
        ctx.db_path = db_path
        ctx.config = {"paper": {"backup_retention_days": 7}}
        ctx.paper_cfg = {"backup_retention_days": 7}

        # Patch backup functions to avoid filesystem operations
        with patch("ashare_lab.paper.pipeline.hot_backup_with_integrity") as mock_backup, \
             patch("ashare_lab.paper.pipeline.create_golden_backup"), \
             patch("ashare_lab.paper.pipeline.cleanup_old_backups"), \
             patch("ashare_lab.paper.pipeline.record_run"):
            mock_backup.return_value = type("R", (), {"success": True})()
            _step12_backup_and_finalize(ctx)

        # Today's anomalous NAV row should be deleted
        rows = conn.execute(
            "SELECT trade_date, total_nav FROM nav ORDER BY trade_date"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["trade_date"] == "2024-01-01"
        assert rows[0]["total_nav"] == 300000.0
        conn.close()

    def test_normal_nav_preserved(self, tmp_path):
        """NAV change <20% -> row preserved."""
        from ashare_lab.paper.ledger import get_connection, init_schema, record_nav
        from ashare_lab.paper.pipeline import DailyRunContext, _step12_backup_and_finalize
        from unittest.mock import patch

        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        init_schema(conn)

        # Insert 2 NAV rows: both normal (+5%)
        record_nav(conn, "2024-01-01", 100000, 200000, 300000, None, None, None, None)
        record_nav(conn, "2024-01-02", 100000, 205000, 305000, None, None, None, None)
        conn.commit()

        ctx = DailyRunContext(
            trade_date="2024-01-02", force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str="",
        )
        ctx.conn = conn
        ctx.db_path = db_path
        ctx.config = {"paper": {"backup_retention_days": 7}}
        ctx.paper_cfg = {"backup_retention_days": 7}

        with patch("ashare_lab.paper.pipeline.hot_backup_with_integrity") as mock_backup, \
             patch("ashare_lab.paper.pipeline.create_golden_backup"), \
             patch("ashare_lab.paper.pipeline.cleanup_old_backups"), \
             patch("ashare_lab.paper.pipeline.record_run"):
            mock_backup.return_value = type("R", (), {"success": True})()
            _step12_backup_and_finalize(ctx)

        # Both rows preserved
        count = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert count == 2
        conn.close()


class TestHedgePrefetch:
    """Layer 3: _prefetch_hedge_prices_for_settle fetches ETF prices
    before settle so hedge orders can fill."""

    def test_prefetch_populates_hedge_prices(self, tmp_path):
        """With hedge enabled, prefetch adds ETF prices to ctx.prices."""
        from ashare_lab.paper.pipeline import (
            DailyRunContext, _prefetch_hedge_prices_for_settle,
        )
        from ashare_lab.paper.hedge import HedgeConfig, HedgeLeg
        from unittest.mock import patch

        ctx = DailyRunContext(
            trade_date="2024-01-02", force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str="",
        )
        ctx.config = {"paper": {"hedge": {"enabled": True}}}
        ctx.prices = {}

        mock_hedge_cfg = HedgeConfig(
            equity_ramp=[(0.0, 0.8), (0.10, 0.2)],
            legs=[HedgeLeg(symbol="511260", weight=1.0, leg_type="treasury_etf")],
            activate_dd=0.10,
        )
        mock_prices = {"511260": {"close": 10.5, "volume": 1e12, "factor": 1.0, "change": 0.0, "threshold": 0.10}}
        with patch("ashare_lab.paper.hedge._load_hedge_config", return_value=mock_hedge_cfg),              patch("ashare_lab.paper.pipeline._fetch_hedge_prices") as mock_fetch:
            def side_effect(ctx, syms):
                ctx.prices.update(mock_prices)
            mock_fetch.side_effect = side_effect
            _prefetch_hedge_prices_for_settle(ctx)

        assert "511260" in ctx.prices
        assert ctx.prices["511260"]["volume"] > 0
        assert ctx.prices["511260"]["volume"] < float("inf")

    def test_prefetch_skips_when_disabled(self):
        """With hedge disabled, prefetch is a no-op."""
        from ashare_lab.paper.pipeline import (
            DailyRunContext, _prefetch_hedge_prices_for_settle,
        )

        ctx = DailyRunContext(
            trade_date="2024-01-02", force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str="",
        )
        ctx.config = {"paper": {"hedge": {"enabled": False}}}
        ctx.prices = {}

        _prefetch_hedge_prices_for_settle(ctx)
        assert ctx.prices == {}  # nothing added


class TestPrefetchCallOrder:
    """Layer 3 wiring: _prefetch_hedge_prices_for_settle must be called
    before _step8_settle in the pipeline execution sequence."""

    def test_prefetch_called_before_settle(self):
        """Deleting the prefetch call at pipeline.py:1796 must fail this test."""
        import ast
        import inspect
        source = inspect.getsource(
            __import__('ashare_lab.paper.pipeline', fromlist=['run_daily'])
        )
        tree = ast.parse(source)
        # Find run_daily function body
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.FunctionDef):
                if node.name == 'run_daily':
                    body = ast.dump(node)
                    # prefetch must appear before settle in the AST
                    prefetch_pos = body.find('_prefetch_hedge_prices_for_settle')
                    settle_pos = body.find('_step8_settle')
                    assert prefetch_pos > 0, "_prefetch_hedge_prices_for_settle not found in run_daily"
                    assert settle_pos > 0, "_step8_settle not found in run_daily"
                    assert prefetch_pos < settle_pos, (
                        f"_prefetch_hedge_prices_for_settle (pos {prefetch_pos}) must come "
                        f"before _step8_settle (pos {settle_pos}) in run_daily"
                    )
                    return
        pytest.fail("run_daily function not found in pipeline.py")


class TestSettleIdentityGate:
    """_step8_settle halts when the DB positions snapshot plus
    settled cash don't reconstruct the post-trade NAV settle_day
    itself computed."""

    def test_identity_violation_halts_with_error(self, db_path: Path) -> None:
        from ashare_lab.paper.pipeline import DailyRunContext, _step8_settle
        from ashare_lab.paper.ledger import (
            get_connection, insert_order, snapshot_positions,
        )

        trade_date = "2025-06-20"
        conn = get_connection(db_path)

        # Seed a positions snapshot that will NOT reconcile with the
        # (mocked) settle result below -- stands in for a ledger
        # where persisted cash/positions have drifted from true NAV.
        snapshot_positions(conn, trade_date, {
            "SZ000001": {
                "qty": 100, "avg_cost": 10.0, "market_value": 1000.0,
                "buy_date": trade_date, "holding_high": 10.0, "factor": 1.0,
            },
        })

        # Seed a real carry order so the gate-before-bump ordering can be
        # checked directly: if bump_carry_days ran before the identity
        # gate, this order's carry_day would already be incremented by
        # the time the gate trips and we inspect it below.
        carry_order_id = insert_order(
            conn, trade_date, "SZ000002", "buy", 100, None,
            "carry", 1, trade_date,
        )
        conn.commit()

        ctx = DailyRunContext(
            trade_date=trade_date, force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str=trade_date,
        )
        ctx.conn = conn
        ctx.cooldown_state = {}
        ctx.current_positions = {}
        ctx.cash = 50_000.0
        ctx.prices = {}
        ctx.benchmarks = {}
        ctx.paper_cfg = {}

        fake_result = _settle_mock(cash=50_000)
        fake_result.post_trade_nav = 999_999.0  # deliberately unreconcilable
        # Explicit, not just _settle_mock's default: pin exactly which
        # order a violation must NOT touch, so the assertion below is
        # unambiguous about what "not bumped" means.
        fake_result.carries_to_bump = [
            {"order_id": carry_order_id, "symbol": "SZ000002", "side": "buy"},
        ]
        fake_result.carries_suspended = []

        with patch(f"{_MOD}.settle_day", return_value=fake_result):
            rc = _step8_settle(ctx)

        assert rc == 2
        row = conn.execute(
            "SELECT status FROM runs WHERE trade_date=?", (trade_date,)
        ).fetchone()
        assert row["status"] == "ledger_error"

        # The gate runs before bump_carry_days, so a violation must not
        # leave this order's carry_day incremented.
        order_row = conn.execute(
            "SELECT carry_day FROM orders WHERE id=?", (carry_order_id,)
        ).fetchone()
        assert order_row["carry_day"] == 1, (
            "carry_day must stay at its seeded value -- bump_carry_days "
            "must not run when the identity gate trips"
        )

        conn.close()

    def test_reconciled_settle_does_not_halt(self, db_path: Path) -> None:
        """Control case: matching numbers must NOT trip the gate."""
        from ashare_lab.paper.pipeline import DailyRunContext, _step8_settle
        from ashare_lab.paper.ledger import get_connection, snapshot_positions

        trade_date = "2025-06-20"
        conn = get_connection(db_path)

        snapshot_positions(conn, trade_date, {
            "SZ000001": {
                "qty": 100, "avg_cost": 10.0, "market_value": 1000.0,
                "buy_date": trade_date, "holding_high": 10.0, "factor": 1.0,
            },
        })
        conn.commit()

        ctx = DailyRunContext(
            trade_date=trade_date, force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str=trade_date,
        )
        ctx.conn = conn
        ctx.cooldown_state = {}
        ctx.current_positions = {}
        ctx.cash = 50_000.0
        ctx.prices = {}
        ctx.benchmarks = {}
        ctx.paper_cfg = {}

        # market_value(1000) + cash(50_000) == post_trade_nav(51_000)
        fake_result = _settle_mock(cash=50_000)
        fake_result.post_trade_nav = 51_000.0

        with patch(f"{_MOD}.settle_day", return_value=fake_result):
            rc = _step8_settle(ctx)

        assert rc is None
        row = conn.execute(
            "SELECT status FROM runs WHERE trade_date=?", (trade_date,)
        ).fetchone()
        assert row is None  # gate did not touch runs
        conn.close()

    def _run_with_drift(self, db_path: Path, drift: float) -> tuple[int | None, sqlite3.Row | None]:
        """Seed market_value=1000, cash=50_000, and pick post_trade_nav so
        that mv_db + cash - post_trade_nav equals exactly *drift*."""
        from ashare_lab.paper.pipeline import DailyRunContext, _step8_settle
        from ashare_lab.paper.ledger import get_connection, snapshot_positions

        trade_date = "2025-06-20"
        conn = get_connection(db_path)

        snapshot_positions(conn, trade_date, {
            "SZ000001": {
                "qty": 100, "avg_cost": 10.0, "market_value": 1000.0,
                "buy_date": trade_date, "holding_high": 10.0, "factor": 1.0,
            },
        })
        conn.commit()

        ctx = DailyRunContext(
            trade_date=trade_date, force=False, steps=None,
            pred_path=None, start_time=0.0, predictions_date_str=trade_date,
        )
        ctx.conn = conn
        ctx.cooldown_state = {}
        ctx.current_positions = {}
        ctx.cash = 50_000.0
        ctx.prices = {}
        ctx.benchmarks = {}
        ctx.paper_cfg = {}

        fake_result = _settle_mock(cash=50_000)
        fake_result.post_trade_nav = 1000.0 + 50_000.0 - drift

        with patch(f"{_MOD}.settle_day", return_value=fake_result):
            rc = _step8_settle(ctx)

        row = conn.execute(
            "SELECT status FROM runs WHERE trade_date=?", (trade_date,)
        ).fetchone()
        conn.close()
        return rc, row

    def test_drift_just_under_threshold_does_not_halt(self, db_path: Path) -> None:
        """drift=0.99 is inside the 1.0 CNY tolerance -- must not trip."""
        rc, row = self._run_with_drift(db_path, 0.99)
        assert rc is None
        assert row is None

    def test_drift_at_threshold_halts(self, db_path: Path) -> None:
        """drift=1.00 meets the >= 1.0 boundary -- must trip."""
        rc, row = self._run_with_drift(db_path, 1.00)
        assert rc == 2
        assert row["status"] == "ledger_error"

    def test_negative_drift_just_under_threshold_does_not_halt(
        self, db_path: Path
    ) -> None:
        """drift=-0.99 is inside the tolerance on the other side of zero --
        abs(drift) must be checked, not drift alone."""
        rc, row = self._run_with_drift(db_path, -0.99)
        assert rc is None
        assert row is None

    def test_negative_drift_at_threshold_halts(self, db_path: Path) -> None:
        """drift=-1.00 meets the |drift| >= 1.0 boundary -- must trip."""
        rc, row = self._run_with_drift(db_path, -1.00)
        assert rc == 2
        assert row["status"] == "ledger_error"
