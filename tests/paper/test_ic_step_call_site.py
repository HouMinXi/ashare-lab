"""Drive signal_quality_ic_step end to end so the call site is covered.

The earlier tests call _write_sidecar directly, so deleting the clear= argument
from the step would leave them green. This one runs the step with the expensive
parts stubbed and asserts on the files it leaves behind.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.paper import signal_quality as sq


def _write_parquet(path, instruments: list[str]) -> None:
    pd.DataFrame({"instrument": instruments, "score": [0.1] * len(instruments)}).to_parquet(path)


@pytest.fixture()
def predictions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(sq, "previous_trading_day", lambda day: day)
    monkeypatch.setattr(sq, "compute_forward_returns", lambda *a, **k: {})
    monkeypatch.setattr(sq, "lagged_ic_for_date", lambda *a, **k: (0.123, 999))
    monkeypatch.setattr(sq, "IC_HISTORY_PATH", tmp_path / "ic_history.tsv")
    monkeypatch.delenv("ASHARE_USE_STALE", raising=False)
    return tmp_path


def test_the_step_clears_a_previous_error_on_success(predictions_dir) -> None:
    trade_date = "2026-10-30"
    pred_date = "2026-10-30"
    _write_parquet(predictions_dir / f"{pred_date}.parquet", ["SH600000"])
    t5_sidecar = predictions_dir / f"{pred_date}.meta.json"
    t5_sidecar.write_text(
        json.dumps({"lagged_ic_error": "BoomError: stale", "lagged_ic_t5_reason": "missing_t5_parquet"})
    )

    ctx = SimpleNamespace(
        steps=None,
        pred_path=predictions_dir / f"{pred_date}.parquet",
        trade_date=trade_date,
        predictions_date_str=pred_date,
    )
    sq.signal_quality_ic_step(ctx)

    meta = json.loads(t5_sidecar.read_text())
    assert meta["lagged_ic_t5"] == 0.123
    assert "lagged_ic_error" not in meta, "a success must not leave the old error"
    assert "lagged_ic_t5_reason" not in meta
