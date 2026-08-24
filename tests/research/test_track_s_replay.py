"""Unit tests for Track S dynamic-slippage offline replay (R6).

Covers:
- S0, S2, S3, S1 slippage model calculations
- Gate 1 check: S0 vs Track A NAV tolerance (<= 1e-6 passes, > 1e-6 fails)
- Replay NAV calculation and trade enrichment
- Shadow DB and artifact creation
- ILLIQ calibration logic, quintile banding, and Gate 4 health checks
- TDD Bug-injection tests (0.001 vs 0.0015 slippage divergence caught by Gate 1)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.research.calibrate_illiq import (
    compute_illiq_from_df,
    compute_quintile_bands,
    gate_4_health_check,
)
from ashare_lab.research.track_s_replay import (
    DayNav,
    ReplayResult,
    S0Fixed,
    S1AlmgrenChriss,
    S2Amihud,
    S3Banded,
    TrackTrade,
    gate_1_check,
    get_model,
    init_shadow_db,
    run_model,
    write_artifact,
    write_to_shadow_db,
)

# ---------------------------------------------------------------------------
# Test Slippage Models
# ---------------------------------------------------------------------------


class TestSlippageModels:
    def test_s0_fixed_buy_sell(self) -> None:
        model = S0Fixed(slippage=0.001)
        trade_buy = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600000",
            side="buy",
            fill_qty=1000,
            orig_fill_price=10.01,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            close=10.0,
            volume=100000.0,
        )
        assert model.compute_fill_price(trade_buy) == pytest.approx(10.01)

        trade_sell = TrackTrade(
            order_id=2,
            trade_date="2026-08-18",
            symbol="SH600000",
            side="sell",
            fill_qty=1000,
            orig_fill_price=9.99,
            orig_commission=5.0,
            orig_stamp=5.0,
            orig_transfer_fee=0.1,
            close=10.0,
            volume=100000.0,
        )
        assert model.compute_fill_price(trade_sell) == pytest.approx(9.99)

    def test_s2_amihud(self) -> None:
        calib = {
            "metadata": {"calibration_end": "2026-08-17"},
            "per_stock": {"SH600000": {"illiq": 1e-8, "n_days": 20}},
        }
        model = S2Amihud(calibration=calib)
        trade = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600000",
            side="buy",
            fill_qty=1000,
            orig_fill_price=10.01,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            close=10.0,
            volume=100000.0,
        )
        # notional = 1000 * 10 = 10000
        # impact_bp = 1e-8 * 10000 * 1e4 = 1.0 bp = 0.0001
        # buy fill_price = 10.0 * (1 + 0.0001) = 10.001
        assert model.compute_fill_price(trade) == pytest.approx(10.001)

    def test_s3_banded(self) -> None:
        calib = {
            "metadata": {"calibration_end": "2026-08-17"},
            "per_stock": {
                "SH600001": {"quintile": 0},
                "SH600002": {"quintile": 1},
                "SH600003": {"quintile": 2},
                "SH600004": {"quintile": 3},
                "SH600005": {"quintile": 4},
            },
            "quintile_rates": [0.0005, 0.0008, 0.0010, 0.0015, 0.0025],
        }
        model = S3Banded(calibration=calib)
        t1 = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600001",
            side="buy",
            fill_qty=100,
            orig_fill_price=10.0,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            close=10.0,
        )
        assert model.compute_fill_price(t1) == pytest.approx(10.0 * 1.0005)

        t5 = TrackTrade(
            order_id=2,
            trade_date="2026-08-18",
            symbol="SH600005",
            side="sell",
            fill_qty=100,
            orig_fill_price=10.0,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            close=10.0,
        )
        assert model.compute_fill_price(t5) == pytest.approx(10.0 * (1.0 - 0.0025))

    def test_s1_almgren_chriss(self) -> None:
        model = S1AlmgrenChriss(k=0.1, default_sigma=0.02)
        trade = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600001",
            side="buy",
            fill_qty=10000,
            orig_fill_price=10.0,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            close=10.0,
            volume=1000000.0,  # part = 0.01, sqrt(0.01) = 0.1
            change=0.02,  # sigma = 0.02
        )
        # impact_bp = 0.1 * 0.02 * 0.1 * 1e4 = 2.0 bp = 0.0002
        # fill_price = 10.0 * (1 + 0.0002) = 10.002
        assert model.compute_fill_price(trade) == pytest.approx(10.002)

    def test_get_model_factory(self) -> None:
        m0 = get_model("S0")
        assert isinstance(m0, S0Fixed)
        calib = {"per_stock": {}, "bands": {}}
        m2 = get_model("S2", calib)
        assert isinstance(m2, S2Amihud)
        m3 = get_model("S3", calib)
        assert isinstance(m3, S3Banded)
        m1 = get_model("S1")
        assert isinstance(m1, S1AlmgrenChriss)


# ---------------------------------------------------------------------------
# Test Gate 1 Verification & Bug-Injection
# ---------------------------------------------------------------------------


class TestGate1Validation:
    def test_gate1_passes_identical(self) -> None:
        s0_res = ReplayResult(
            model_name="S0",
            trades=[],
            nav_series=[
                DayNav("2026-08-18", 100000.0, 200000.0, 300000.0, 300000.0, 300000.0),
                DayNav("2026-08-19", 100500.0, 200000.0, 300500.0, 300000.0, 300500.0),
            ],
        )
        track_a_nav = [
            {"trade_date": "2026-08-18", "total_nav": 300000.0},
            {"trade_date": "2026-08-19", "total_nav": 300500.0},
        ]
        diff = gate_1_check(s0_res, track_a_nav)
        assert diff == 0.0

    def test_gate1_fails_on_divergence_bug_injection(self) -> None:
        # Injected bug: S0 has a 0.0015 slip difference causing NAV divergence > 1e-6
        s0_res = ReplayResult(
            model_name="S0",
            trades=[],
            nav_series=[
                DayNav("2026-08-18", 100000.0, 200000.0, 300000.0, 300000.0, 300000.0),
                DayNav("2026-08-19", 100000.0, 200000.0, 300000.0, 300000.0, 300000.0),
            ],
        )
        track_a_nav = [
            {"trade_date": "2026-08-18", "total_nav": 300000.0},
            {"trade_date": "2026-08-19", "total_nav": 300500.0},  # diff ~ 1.66e-3 > 1e-6
        ]
        with pytest.raises(SystemExit) as excinfo:
            gate_1_check(s0_res, track_a_nav)
        assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Test Replay Engine NAV Reconstruction
# ---------------------------------------------------------------------------


class TestReplayEngine:
    def test_run_model_reconstructs_nav(self) -> None:
        model = S0Fixed(slippage=0.001)
        nav_dates = ["2026-08-18", "2026-08-19"]
        prices = {
            "SH600000": {
                "2026-08-18": {"close": 10.0, "volume": 100000.0, "change": 0.0, "factor": 1.0},
                "2026-08-19": {"close": 11.0, "volume": 100000.0, "change": 0.1, "factor": 1.0},
            }
        }
        # Buy on day 1
        t1 = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600000",
            side="buy",
            fill_qty=1000,
            orig_fill_price=10.01,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
        )
        # Sell on day 2
        t2 = TrackTrade(
            order_id=2,
            trade_date="2026-08-19",
            symbol="SH600000",
            side="sell",
            fill_qty=1000,
            orig_fill_price=10.989,
            orig_commission=5.0,
            orig_stamp=5.4945,
            orig_transfer_fee=0.10989,
        )

        res = run_model([t1, t2], nav_dates, prices, model, initial_cash=300000.0)
        assert len(res.nav_series) == 2
        # Day 1: cash spent = 1000 * 10.01 + 5.1001 = 10015.1001
        # cash = 300000 - 10015.1001 = 289984.8999
        # mv = 1000 * 10.0 = 10000.0
        # total_nav = 299984.8999
        assert res.nav_series[0].trade_date == "2026-08-18"
        assert res.nav_series[0].cash == pytest.approx(289984.8999)
        assert res.nav_series[0].market_value == pytest.approx(10000.0)
        assert res.nav_series[0].total_nav == pytest.approx(299984.8999)

        # Day 2: sold all shares
        assert res.nav_series[1].trade_date == "2026-08-19"
        assert res.nav_series[1].market_value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test Shadow DB & Artifacts
# ---------------------------------------------------------------------------


class TestShadowDbAndArtifacts:
    def test_init_and_write_shadow_db(self, tmp_path: Path) -> None:
        db_file = tmp_path / "paper_s_S0.db"
        conn = init_shadow_db(db_file)
        conn.close()

        # Check tables created
        conn = sqlite3.connect(str(db_file))
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "trades" in tables
        assert "nav" in tables
        conn.close()

        # Write replay result
        trades = [
            TrackTrade(
                order_id=1,
                trade_date="2026-08-18",
                symbol="SH600000",
                side="buy",
                fill_qty=1000,
                orig_fill_price=10.01,
                orig_commission=5.0,
                orig_stamp=0.0,
                orig_transfer_fee=0.1,
                replay_fill_price=10.01,
                replay_commission=5.0,
                replay_stamp=0.0,
                replay_transfer_fee=0.1,
                close=10.0,
            )
        ]
        nav_series = [
            DayNav(
                trade_date="2026-08-18",
                cash=289984.9,
                market_value=10000.0,
                total_nav=299984.9,
                pre_trade_nav=300000.0,
                post_trade_nav=299984.9,
            )
        ]
        write_to_shadow_db(
            db_path=db_file,
            model_name="S0",
            trades=trades,
            nav_series=nav_series,
            initial_cash=300000.0,
        )

        conn = sqlite3.connect(str(db_file))
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        nav_count = conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0]
        assert trade_count == 1
        assert nav_count == 1
        conn.close()

    def test_write_artifact(self, tmp_path: Path) -> None:
        art_dir = tmp_path / "experiments" / "shadow_slippage"
        t1 = TrackTrade(
            order_id=1,
            trade_date="2026-08-18",
            symbol="SH600000",
            side="buy",
            fill_qty=1000,
            orig_fill_price=10.01,
            orig_commission=5.0,
            orig_stamp=0.0,
            orig_transfer_fee=0.1,
            replay_fill_price=10.01,
            replay_commission=5.0,
            replay_stamp=0.0,
            replay_transfer_fee=0.1,
            close=10.0,
            volume=100000.0,
        )
        day_nav = DayNav(
            trade_date="2026-08-18",
            cash=290000.0,
            market_value=10000.0,
            total_nav=300000.0,
            pre_trade_nav=300000.0,
            post_trade_nav=300000.0,
        )
        path = write_artifact(
            artifact_dir=art_dir,
            model_name="S2",
            trade_date="2026-08-18",
            day_trades=[t1],
            nav=day_nav,
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == "S2"
        assert data["trade_date"] == "2026-08-18"
        assert data["nav"]["total_nav"] == 300000.0


# ---------------------------------------------------------------------------
# Test ILLIQ Computation & Gate 4
# ---------------------------------------------------------------------------


class TestIlliqComputation:
    def test_compute_illiq_basic(self) -> None:
        # Create MultiIndex DataFrame as returned by qlib
        idx = pd.MultiIndex.from_product(
            [["SH600001"], [f"2026-07-{i:02d}" for i in range(1, 31)]],
            names=["instrument", "datetime"],
        )
        df = pd.DataFrame(
            {
                "$close": [10.0] * 30,
                "$volume": [1000000.0] * 30,
                "$change": [0.01] * 30,
                "$factor": [1.0] * 30,
            },
            index=idx,
        )
        per_stock, corr = compute_illiq_from_df(df, window=20)
        assert "SH600001" in per_stock
        # |change| / (close * volume) = 0.01 / (10 * 1000000) = 1e-9
        assert per_stock["SH600001"]["illiq"] == pytest.approx(1e-9)
        assert per_stock["SH600001"]["n_days"] == 20

    def test_compute_quintile_bands(self) -> None:
        per_stock = {
            f"S{i}": {"illiq": float(i)} for i in range(1, 101)
        }
        per_stock, boundaries = compute_quintile_bands(per_stock)
        assert len(boundaries) == 4
        assert boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]
        assert per_stock["S1"]["quintile"] == 0
        assert per_stock["S100"]["quintile"] == 4

    def test_gate_4_health_check(self) -> None:
        per_stock = {f"S{i}": {"illiq": float(i)} for i in range(1, 100)}
        boundaries = [20.0, 40.0, 60.0, 80.0]
        health = gate_4_health_check(per_stock, boundaries, spearman_corr=0.92)
        assert health["pass"] is True
        assert len(health["failures"]) == 0
