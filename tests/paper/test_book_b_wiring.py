"""L2 timing layer wiring tests for Book B shadow ledgers.

Covers wiring the three pre-registered timing nulls (n1/n2/n3) plus the
existing "none" canary into the daily pipeline as shadow Book B ledgers.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ashare_lab.paper.ledger import get_connection, init_schema
from ashare_lab.paper.pipeline import _run_book_b, _step12_book_b


@pytest.fixture(autouse=True)
def _cd_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own tmp_path so artifact I/O is isolated."""
    monkeypatch.chdir(tmp_path)


def _make_ctx(tmp_path, trade_date, pred_path, book_a_nav=1_000_000.0):
    """Build a minimal DailyRunContext for _run_book_b wiring tests."""
    from ashare_lab.paper.pipeline import DailyRunContext

    ctx = DailyRunContext.__new__(DailyRunContext)
    ctx.trade_date = trade_date
    ctx.db_path = tmp_path / "paper.db"
    ctx.paper_cfg = {
        "initial_cash": book_a_nav,
        "topk": 15,
        "n_drop": 1,
        "listing_min_days": 60,
        "turnover_cap": None,
    }
    ctx.config = {
        "cost_model": {"risk_degree": 0.95},
        "universe": {
            "listing_min_days": 60,
            "min_avg_turnover_20d": 0.0,
            "exclude_close_above_cny": 300.0,
        },
        "carry_days": 3,
        "slippage": 0.001,
        "volume_participation_pct": 0.05,
    }
    ctx.risk_result = MagicMock(
        buying_halted=False,
        forced_sells={},
        blocked_industries=set(),
        blocked_rebuys=set(),
        topk_override=None,
        cooldown_entries={},
    )
    ctx.prices = {"SH600519": {"close": 180.0}}
    ctx.pred_path = pred_path
    ctx.universe_symbols = ["SH600519"]
    ctx.ipo_listing_syms = set()
    ctx.st_names = set()
    ctx.market_data = {
        "SH600519": {
            "close": 180.0,
            "listing_days": 100,
            "avg_turnover_20d": 100_000_000.0,
        },
    }
    ctx.industry_map = {}
    ctx.benchmarks = {"csi1000": 1.0}
    ctx.hedge_state = None
    ctx.hedge_symbols = set()
    ctx.total_nav = book_a_nav
    ctx.current_positions = {}
    ctx.cash = book_a_nav
    return ctx


def _init_prod_and_book_b_db(tmp_path, book_id="none", cash=300000.0):
    """Create production DB and bootstrapped Book B DB with cash state."""
    prod_db = tmp_path / "paper.db"
    conn = sqlite3.connect(str(prod_db))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.execute(
        "INSERT OR IGNORE INTO paper_state (key, value) VALUES (?, ?)",
        ("cash", str(cash)),
    )
    conn.commit()
    conn.close()

    book_b_db = tmp_path / f"paper_b_{book_id}.db"
    conn_b = get_connection(book_b_db)
    init_schema(conn_b)
    conn_b.execute(
        "INSERT OR IGNORE INTO paper_state (key, value) VALUES (?, ?)",
        ("cash", str(cash)),
    )
    conn_b.commit()
    conn_b.close()
    return prod_db, book_b_db


def _make_pred_path(tmp_path, trade_date):
    """Create a placeholder prediction parquet (path only used by ctx)."""
    pred_path = tmp_path / "predictions" / f"{trade_date}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    # Real signal generation is mocked in wiring tests; the parquet just
    # needs to exist so ctx.pred_path resolves.
    pd.DataFrame({"instrument": ["SH600519"], "score": [0.8]}).to_parquet(pred_path)
    return pred_path


# -- Contract 1: book_id parameterization -------------------------------------

class TestBookIdParameterization:
    """_run_book_b(ctx, book_id) opens the right DB and asks timing_multiplier
    for the right model."""

    def test_run_book_b_n1_opens_paper_b_n1_db(self, tmp_path):
        trade_date = "2026-07-25"
        _init_prod_and_book_b_db(tmp_path, "n1")
        pred_path = _make_pred_path(tmp_path, trade_date)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        with patch("ashare_lab.paper.pipeline.timing_multiplier") as mock_m, \
             patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            mock_m.return_value = 0.5
            _run_book_b(ctx, "n1")

        assert (tmp_path / "paper_b_n1.db").exists()
        mock_m.assert_called_once_with(trade_date, model="n1")

    def test_run_book_b_each_model_opens_respective_db(self, tmp_path):
        trade_date = "2026-07-25"
        pred_path = _make_pred_path(tmp_path, trade_date)
        for book_id in ("n1", "n2", "n3"):
            _init_prod_and_book_b_db(tmp_path, book_id)
            ctx = _make_ctx(tmp_path, trade_date, pred_path)
            with patch("ashare_lab.paper.pipeline.timing_multiplier") as mock_m, \
                 patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
                mock_m.return_value = 0.5
                _run_book_b(ctx, book_id)
            mock_m.assert_called_once_with(trade_date, model=book_id)
            assert (tmp_path / f"paper_b_{book_id}.db").exists()


# -- Contract 2: loop isolation -----------------------------------------------

class TestLoopIsolation:
    """One Book B ledger raising must not stop the other three."""

    def test_one_book_failure_others_still_run(self, tmp_path, caplog):
        trade_date = "2026-07-25"
        ctx = MagicMock()
        ctx.trade_date = trade_date

        call_order = []

        def fake_run_book_b(ctx, book_id):
            call_order.append(book_id)
            if book_id == "n2":
                raise RuntimeError("simulated n2 failure")

        with patch("ashare_lab.paper.pipeline._run_book_b", side_effect=fake_run_book_b):
            _step12_book_b(ctx)

        assert call_order == ["none", "n1", "n2", "n3"]
        assert "simulated n2 failure" in caplog.text


# -- Contract 3: R3 canary ----------------------------------------------------

class TestR3Canary:
    """book_id='none' with m=1.0 reproduces Book A NAV to the cent."""

    def test_none_book_b_nav_matches_book_a_with_m_one(self, tmp_path):
        trade_date = "2026-07-22"
        next_td = dt.date.fromisoformat(trade_date) + dt.timedelta(days=1)
        _init_prod_and_book_b_db(tmp_path, "none")
        pred_path = _make_pred_path(tmp_path, trade_date)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        with patch("ashare_lab.paper.pipeline.next_trading_day", return_value=next_td), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            _run_book_b(ctx, "none")

        artifact_path = Path("experiments/control_books") / f"{trade_date}_none.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["book_id"] == "none"
        assert artifact["m"] == 1.0
        assert artifact["nav_a"] == artifact["nav_b"]

    def test_m_scaling_changes_target_value(self, tmp_path):
        """m=0.5 produces roughly half the buy qty of m=1.0.

        This is the detection surface for PM bug-injection (a): if the
        `* m` in target_value is deleted, Book B with m=0.5 receives the
        same target_value as m=1.0, so the buy-qty ratio collapses to 1.0
        and this test fails.
        """
        trade_date = "2026-07-22"
        next_td = dt.date.fromisoformat(trade_date) + dt.timedelta(days=1)

        def run_with_m(m):
            _init_prod_and_book_b_db(tmp_path, f"n1_m{int(m * 10)}")
            pred_path = _make_pred_path(tmp_path, trade_date)
            ctx = _make_ctx(tmp_path, trade_date, pred_path)
            captured = []

            def capture_insert_order(conn, trade_date, symbol, side, target_qty, *args, **kwargs):
                if side == "buy":
                    captured.append(target_qty)

            with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=m), \
                 patch("ashare_lab.paper.pipeline.next_trading_day", return_value=next_td), \
                 patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}), \
                 patch("ashare_lab.paper.pipeline.insert_order", side_effect=capture_insert_order):
                _run_book_b(ctx, "n1")
            return captured

        full = run_with_m(1.0)
        half = run_with_m(0.5)
        assert full and half
        # Round lots (100-share) make the ratio approximate; the important
        # property is that m=0.5 yields strictly less buy exposure than m=1.0.
        assert 0.25 < max(half) / max(full) < 0.45


# -- Contract 4: memoization --------------------------------------------------

class TestMemoization:
    """The CSI1000 fetch is shared across all books for a single trade_date."""

    def test_three_books_one_fetch(self, tmp_path):
        trade_date = "2026-07-25"
        pred_path = _make_pred_path(tmp_path, trade_date)
        for book_id in ("none", "n1", "n2", "n3"):
            _init_prod_and_book_b_db(tmp_path, book_id)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        fake_closes = pd.Series(
            list(range(300, 600)),
            index=pd.date_range(end=trade_date, periods=300, freq="B"),
        )

        with patch("ashare_lab.paper.pipeline.subprocess.run") as mock_run, \
             patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = fake_closes.to_json(date_format="iso")
            _step12_book_b(ctx)

        assert mock_run.call_count == 1


# -- Contract 5: artifact shape -----------------------------------------------

class TestArtifactShape:
    """Each book writes its own artifact with the right m and schema."""

    def test_all_four_artifacts_written(self, tmp_path):
        trade_date = "2026-07-25"
        pred_path = _make_pred_path(tmp_path, trade_date)
        for book_id in ("none", "n1", "n2", "n3"):
            _init_prod_and_book_b_db(tmp_path, book_id)
        ctx = _make_ctx(tmp_path, trade_date, pred_path)

        model_m = {"none": 1.0, "n1": 0.5, "n2": 0.0, "n3": 1.0}

        def fake_timing_multiplier(trade_date, model):
            return model_m[model]

        with patch("ashare_lab.paper.pipeline.timing_multiplier", side_effect=fake_timing_multiplier), \
             patch("ashare_lab.paper.signal.generate_signals", return_value={"SH600519": 0.8}):
            _step12_book_b(ctx)

        for book_id in ("none", "n1", "n2", "n3"):
            artifact_path = Path("experiments/control_books") / f"{trade_date}_{book_id}.json"
            assert artifact_path.exists(), f"missing artifact for {book_id}"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            assert artifact["date"] == trade_date
            assert artifact["book_id"] == book_id
            assert artifact["m"] == model_m[book_id]
            assert "nav_a" in artifact
            assert "nav_b" in artifact
            assert "nav_c" in artifact
            assert "excess_a" in artifact
            assert "excess_b" in artifact


# -- Contract 6: injection mapping --------------------------------------------

class TestInjectionMapping:
    """Documentation-only mapping of PM bug-injection sites to tests.

    These tests do not inject bugs themselves; they describe which test
    above fails loudly when the PM performs the corresponding injection.
    """

    def test_injection_a_m_multiplication_covered_by_r3_scaling(self):
        """(a) delete `* m` in target_value -> TestR3Canary::test_m_scaling_changes_target_value FAILS."""

    def test_injection_b_drop_artifact_covered_by_shape(self):
        """(b) drop one book's artifact -> TestArtifactShape::test_all_four_artifacts_written FAILS."""

    def test_injection_c_three_fetches_covered_by_memo(self):
        """(c) fetch called 3x -> TestMemoization::test_three_books_one_fetch FAILS."""
