"""Wiring test: signal-quality monitors are appended after _step15_graduation.

This test is intentionally not a unit test of the monitor logic; it verifies
only that run_daily calls the two monitor steps exactly once, in order, after
graduation.
"""
from __future__ import annotations

import contextlib
import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.paper.pipeline import run_daily

_MOD = "ashare_lab.paper.pipeline"


def _base_config(tmp_path: Path) -> dict:
    """Minimal config that lets the mocked pipeline steps run."""
    return {
        "paper": {
            "db_path": str(tmp_path / "paper.db"),
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
            "hedge": {"enabled": False},
        },
        "cost_model": {"risk_degree": 0.95},
        "universe": {"exclude_close_above_cny": 300},
    }


def _step1_init_stub(ctx) -> None:
    """Provide enough context for run_daily to reach graduation."""
    ctx.config = {"paper": {"hedge": {"enabled": False}}}
    ctx.paper_cfg = ctx.config["paper"]
    ctx.risk_cfg = {}
    ctx.db_path = None
    ctx.conn = MagicMock()


@pytest.fixture()
def prediction_file(tmp_path: Path) -> Path:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred_path = pred_dir / "2025-06-20.parquet"
    pred_path.write_bytes(b"")  # touched file is enough; signal gen is mocked
    return pred_path


def test_monitor_wiring_after_graduation(prediction_file, tmp_path):
    """Deleting the two signal_quality_*_step calls in pipeline.py must FAIL this test."""
    common_patches = {
        "load_config": patch(f"{_MOD}.load_config", return_value=_base_config(tmp_path)),
        "PROJECT_ROOT": patch(f"{_MOD}.PROJECT_ROOT", tmp_path),
        "PREDICTIONS_DIR": patch(f"{_MOD}.PREDICTIONS_DIR", prediction_file.parent),
        "is_day_settled": patch(f"{_MOD}.is_day_settled", return_value=False),
        "latest_trading_day": patch(f"{_MOD}.latest_trading_day", return_value=dt.date(2025, 6, 20)),
        "next_trading_day": patch(f"{_MOD}.next_trading_day", return_value=dt.date(2025, 6, 23)),
        "previous_trading_day": patch(f"{_MOD}.previous_trading_day", return_value=dt.date(2025, 6, 19)),
        "open_pipeline_run": patch(f"{_MOD}.open_pipeline_run", return_value=1),
        "_step1_init": patch(f"{_MOD}._step1_init", side_effect=_step1_init_stub),
        "_step2_idempotency": patch(f"{_MOD}._step2_idempotency", return_value=-1),
        "_step3_data_update": patch(f"{_MOD}._step3_data_update", return_value=-1),
        "_gate_data_completeness": patch(f"{_MOD}._gate_data_completeness"),
        "_step4_load_state": patch(f"{_MOD}._step4_load_state"),
        "_step5_fetch_prices_and_universe": patch(f"{_MOD}._step5_fetch_prices_and_universe", return_value=-1),
        "_gate_price_sanity": patch(f"{_MOD}._gate_price_sanity"),
        "_step6_adjustfactor": patch(f"{_MOD}._step6_adjustfactor"),
        "_step7_csi1000_exits": patch(f"{_MOD}._step7_csi1000_exits"),
        "_prefetch_hedge_prices_for_settle": patch(f"{_MOD}._prefetch_hedge_prices_for_settle"),
        "_step8_settle": patch(f"{_MOD}._step8_settle"),
        "_step9_risk_checks": patch(f"{_MOD}._step9_risk_checks"),
        "_step9b_hedge_sleeve": patch(f"{_MOD}._step9b_hedge_sleeve"),
        "_step9c_nav_hedge_split": patch(f"{_MOD}._step9c_nav_hedge_split"),
        "_step10_signal_generation": patch(f"{_MOD}._step10_signal_generation", return_value=-1),
        "_step11_ipo_processing": patch(f"{_MOD}._step11_ipo_processing"),
        "_step12_backup_and_finalize": patch(f"{_MOD}._step12_backup_and_finalize"),
        "_step12_book_b": patch(f"{_MOD}._step12_book_b"),
        "_step13_report": patch(f"{_MOD}._step13_report"),
        "_step14_record_pipeline_run": patch(f"{_MOD}._step14_record_pipeline_run"),
        "_close_pipeline_run": patch(f"{_MOD}._close_pipeline_run"),
    }

    call_order: list[str] = []

    def record_call(name: str):
        def _inner(*args, **kwargs):
            call_order.append(name)
        return _inner

    with contextlib.ExitStack() as stack:
        for p in common_patches.values():
            stack.enter_context(p)
        graduation = stack.enter_context(patch(f"{_MOD}._step15_graduation"))
        psi = stack.enter_context(patch(f"{_MOD}.signal_quality_psi_step"))
        ic = stack.enter_context(patch(f"{_MOD}.signal_quality_ic_step"))

        graduation.side_effect = record_call("graduation")
        psi.side_effect = record_call("psi")
        ic.side_effect = record_call("ic")
        hist = stack.enter_context(patch(f"{_MOD}.append_ic_history"))
        hist.side_effect = record_call("ic_history")

        rc = run_daily("2025-06-20", pred_path=prediction_file)

    assert rc == 0
    graduation.assert_called_once()
    psi.assert_called_once()
    ic.assert_called_once()
    hist.assert_called_once()
    assert call_order == ["graduation", "psi", "ic", "ic_history"]
