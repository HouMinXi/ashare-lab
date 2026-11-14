"""Two contracts that nothing guarded before.

* The exchange date has one implementation: the backup job's name must agree with
  the calendar's at every instant, including across the midnight boundary.
* The PSI step records its value whether or not it trips the alert. Writing only
  on the None branch would leave the file holding an older run's number, and the
  health report would read that as the current one.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.data import calendar as cal_mod
from ashare_lab.paper import backup_job
from ashare_lab.paper import signal_quality as sq


INSTANTS = [
    dt.datetime(2026, 11, 1, 15, 30, tzinfo=dt.timezone.utc),   # Shanghai 23:30, still 11-01
    dt.datetime(2026, 11, 1, 16, 30, tzinfo=dt.timezone.utc),   # Shanghai 00:30, now 11-02
    dt.datetime(2026, 6, 1, 4, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 12, 31, 16, 0, tzinfo=dt.timezone.utc),   # crosses both day and year
]


@pytest.mark.parametrize("moment", INSTANTS)
def test_the_backup_name_agrees_with_the_calendar(moment: dt.datetime) -> None:
    assert backup_job.exchange_today(moment) == cal_mod.market_today(moment)


def test_market_today_accepts_an_instant_without_patching_the_module() -> None:
    """Before this, the only way to pin the date was to replace cal_mod.dt."""
    moment = dt.datetime(2026, 11, 1, 18, 30, tzinfo=dt.timezone.utc)

    assert cal_mod.market_today(moment) == dt.date(2026, 11, 2)


def test_a_naive_instant_is_read_as_exchange_time() -> None:
    assert cal_mod.market_today(dt.datetime(2026, 11, 2, 9, 0)) == dt.date(2026, 11, 2)


# --- PSI -------------------------------------------------------------------

@pytest.fixture()
def psi_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(sq, "load_reference_scores", lambda *a, **k: (np.linspace(0, 1, 100), 30))
    monkeypatch.setattr(sq, "compute_psi", lambda ref, scores: 0.05)
    monkeypatch.delenv("ASHARE_USE_STALE", raising=False)
    monkeypatch.delenv("ASHARE_SQ_DRYRUN", raising=False)

    pred = tmp_path / "2026-10-30.parquet"
    pd.DataFrame({"score": np.linspace(0, 1, 50)}).to_parquet(pred)
    ctx = SimpleNamespace(
        steps=None,
        pred_path=pred,
        predictions_date_str="2026-10-30",
        trade_date="2026-10-30",
    )
    return ctx, pred.with_suffix(".meta.json")


def test_a_normal_psi_value_is_recorded(psi_run) -> None:
    ctx, sidecar = psi_run

    sq.signal_quality_psi_step(ctx)

    meta = json.loads(sidecar.read_text())
    assert meta["psi"] == pytest.approx(0.05), "a run below the alert threshold must still record its value"
    assert "psi_error" not in meta
    assert "psi_reason" not in meta


def test_a_recorded_value_replaces_the_previous_error(psi_run) -> None:
    ctx, sidecar = psi_run
    sidecar.write_text(json.dumps({"psi_error": "BoomError: last night", "psi": 0.4}))

    sq.signal_quality_psi_step(ctx)

    meta = json.loads(sidecar.read_text())
    assert meta["psi"] == pytest.approx(0.05)
    assert "psi_error" not in meta
