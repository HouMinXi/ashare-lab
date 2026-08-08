"""E2E wiring test: verify signal_quality steps are invoked by run_daily.

Deleting the two calls in pipeline.py must make this test FAIL.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


def test_signal_quality_steps_called_after_graduation(tmp_path):
    """run_daily must call psi_step and ic_step once each, after _step15_graduation."""
    from ashare_lab.paper.pipeline import run_daily

    call_log = []

    def log_call(name):
        def wrapper(*args, **kwargs):
            call_log.append(name)
        return wrapper

    def fake_init(ctx):
        ctx.conn = MagicMock()
        ctx.conn.execute.return_value.fetchone.return_value = None
        ctx.pred_path = tmp_path / "2026-08-06.parquet"
        ctx.pred_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_parquet(ctx.pred_path)

    mocks = {
        "_step1_init": patch("ashare_lab.paper.pipeline._step1_init", side_effect=fake_init),
        "_step2_idempotency": patch("ashare_lab.paper.pipeline._step2_idempotency", return_value=-1),
        "_step3_data_update": patch("ashare_lab.paper.pipeline._step3_data_update", return_value=0),
        "gate_data": patch("ashare_lab.paper.pipeline._gate_data_completeness"),
        "_step4_load_state": patch("ashare_lab.paper.pipeline._step4_load_state"),
        "_step5": patch("ashare_lab.paper.pipeline._step5_fetch_prices_and_universe", return_value=0),
        "gate_price": patch("ashare_lab.paper.pipeline._gate_price_sanity"),
        "_step6": patch("ashare_lab.paper.pipeline._step6_adjustfactor"),
        "_step7": patch("ashare_lab.paper.pipeline._step7_csi1000_exits"),
        "_step8": patch("ashare_lab.paper.pipeline._step8_settle", return_value=0),
        "_step9": patch("ashare_lab.paper.pipeline._step9_risk_checks"),
        "_step9b": patch("ashare_lab.paper.pipeline._step9b_hedge_sleeve"),
        "_step9c": patch("ashare_lab.paper.pipeline._step9c_nav_hedge_split"),
        "_step10": patch("ashare_lab.paper.pipeline._step10_signal_generation", return_value=0),
        "_step11": patch("ashare_lab.paper.pipeline._step11_ipo_processing"),
        "_step12a": patch("ashare_lab.paper.pipeline._step12_backup_and_finalize"),
        "_step12b": patch("ashare_lab.paper.pipeline._step12_book_b"),
        "_step13": patch("ashare_lab.paper.pipeline._step13_report"),
        "_step14": patch("ashare_lab.paper.pipeline._step14_record_pipeline_run"),
        "graduation": patch("ashare_lab.paper.pipeline._step15_graduation", side_effect=log_call("graduation")),
        "psi": patch("ashare_lab.paper.pipeline.signal_quality_psi_step", side_effect=log_call("psi")),
        "ic": patch("ashare_lab.paper.pipeline.signal_quality_ic_step", side_effect=log_call("ic")),
        "open_run": patch("ashare_lab.paper.pipeline.open_pipeline_run", return_value=1),
        "close_run": patch("ashare_lab.paper.pipeline._close_pipeline_run"),
    }

    with ExitStack() as stack:
        for m in mocks.values():
            stack.enter_context(m)
        run_daily(trade_date="2026-08-06")

    assert call_log.count("psi") == 1, f"psi_step called {call_log.count('psi')} times"
    assert call_log.count("ic") == 1, f"ic_step called {call_log.count('ic')} times"

    grad_idx = call_log.index("graduation")
    psi_idx = call_log.index("psi")
    ic_idx = call_log.index("ic")
    assert psi_idx > grad_idx, f"psi_step must be after graduation"
    assert ic_idx > grad_idx, f"ic_step must be after graduation"
