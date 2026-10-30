"""The timing multiplier must say whether 1.0 was computed or fallen back to.

Gate 3's same-day canary compared Book B against Book A and passed whenever
the two matched.  On a day the timing model failed, m fell back to 1.0, Book B
became Book A by construction, and the canary read that as agreement.  The
check could not fire on the one failure mode it existed to catch.

These tests pin the distinction at both ends: the function reports a reason,
and the daily artifact records it next to m.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from ashare_lab.paper import pipeline as pl
from ashare_lab.paper.pipeline import (
    timing_multiplier,
    timing_multiplier_detail,
)


def _series(dates, values):
    return pd.Series(values, index=pd.Index(dates, name="date"), dtype=float)


class TestDegradationIsReported:
    def test_none_model_is_not_a_degradation(self):
        # Book A runs with model="none" by design; calling that degraded would
        # mark every single Book A day and make the flag useless.
        m, degraded = timing_multiplier_detail("2026-07-25", "none")
        assert m == 1.0
        assert degraded is None

    def test_default_model_is_not_a_degradation(self):
        assert timing_multiplier_detail("2026-07-25") == (1.0, None)

    def test_unknown_model_reports_unknown_model(self):
        m, degraded = timing_multiplier_detail("2026-07-25", "n9")
        assert m == 1.0
        assert degraded == "unknown_model"

    def test_absent_index_data_reports_no_index_data(self):
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=None
        ):
            assert timing_multiplier_detail("2026-07-25", "n1") == (
                1.0,
                "no_index_data",
            )

    def test_empty_index_data_reports_no_index_data(self):
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=_series([], [])
        ):
            assert timing_multiplier_detail("2026-07-25", "n1") == (
                1.0,
                "no_index_data",
            )

    def test_date_outside_series_reports_date_not_in_series(self):
        closes = _series(["2026-07-23", "2026-07-24"], [100.0, 101.0])
        produced = _series(["2026-07-23", "2026-07-24"], [1.0, 0.5])
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=closes
        ), patch(
            "ashare_lab.research.timing_nulls.ma_trend_multiplier",
            return_value=produced,
        ):
            assert timing_multiplier_detail("2026-07-25", "n1") == (
                1.0,
                "date_not_in_series",
            )

    def test_raising_fetch_reports_computation_error(self):
        with patch.object(
            pl,
            "_fetch_csi1000_closes_for_timing",
            side_effect=RuntimeError("index feed down"),
        ):
            assert timing_multiplier_detail("2026-07-25", "n1") == (
                1.0,
                "computation_error",
            )

    def test_raising_model_reports_computation_error(self):
        closes = _series(["2026-07-24", "2026-07-25"], [100.0, 101.0])
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=closes
        ), patch(
            "ashare_lab.research.timing_nulls.vol_percentile_multiplier",
            side_effect=ValueError("not enough history"),
        ) as mock_model:
            assert timing_multiplier_detail("2026-07-25", "n2") == (
                1.0,
                "computation_error",
            )
        # A misdirected patch is caught one line above: the real model returns
        # 1.0 for a two-point series, so the reason comes back None instead of
        # computation_error. This assertion is the guard for the day that stops
        # being true -- if the real model ever raises on a short series, both
        # paths produce computation_error and only this distinguishes them.
        assert mock_model.called, "n2 does not dispatch to the patched model"


class TestRealValuesAreNotDegraded:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_model_produced_value_reports_no_reason(self, value):
        # 1.0 belongs in this list deliberately: a model that legitimately says
        # "stay fully invested" must stay distinguishable from a fallback,
        # which is the entire point of the flag.
        closes = _series(["2026-07-24", "2026-07-25"], [100.0, 101.0])
        produced = _series(["2026-07-24", "2026-07-25"], [1.0, value])
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=closes
        ), patch(
            "ashare_lab.research.timing_nulls.ma_trend_multiplier",
            return_value=produced,
        ):
            m, degraded = timing_multiplier_detail("2026-07-25", "n1")
        assert m == value
        assert degraded is None

    def test_wrapper_still_returns_a_bare_float(self):
        # Existing callers and assertions must keep working.
        result = timing_multiplier("2026-07-25", "none")
        assert isinstance(result, float)
        assert result == 1.0

    def test_wrapper_drops_the_reason_on_a_degraded_day(self):
        with patch.object(
            pl, "_fetch_csi1000_closes_for_timing", return_value=None
        ):
            result = timing_multiplier("2026-07-25", "n1")
        assert isinstance(result, float)
        assert result == 1.0


def _write_artifact(monkeypatch, work_dir, m, m_degraded):
    """Drive the real artifact writer with the context fields it reads.

    The writer builds a relative path, so the cwd has to move. monkeypatch
    does it rather than os.chdir with try/finally: pytest unwinds it even on
    an exit that would skip a finally block, so no later test inherits a
    changed working directory.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        trade_date="2026-07-25",
        total_nav=101000.0,
        paper_cfg={"initial_cash": 100000.0},
        benchmarks={"csi1000": 100000.0},
    )
    monkeypatch.chdir(work_dir)
    with patch(
        "ashare_lab.paper.ledger.get_positions_for_date", return_value={}
    ), patch(
        "ashare_lab.paper.ledger.get_latest_cash", return_value=100500.0
    ):
        pl._write_abc_artifact(ctx, None, "n1", m, m_degraded)
    path = Path("experiments/control_books/2026-07-25_n1.json")
    assert path.exists(), "artifact was not written"
    return json.loads(path.read_text(encoding="utf-8"))


class TestArtifactCarriesTheReason:
    def test_degraded_day_records_the_reason(self, tmp_path, monkeypatch):
        data = _write_artifact(
            monkeypatch, tmp_path / "degraded", 1.0, "no_index_data"
        )
        assert data["m"] == 1.0
        assert data["m_degraded"] == "no_index_data"

    def test_clean_day_records_no_reason(self, tmp_path, monkeypatch):
        data = _write_artifact(monkeypatch, tmp_path / "clean", 1.0, None)
        assert data["m"] == 1.0
        assert data["m_degraded"] is None

    def test_the_two_days_are_distinguishable(self, tmp_path, monkeypatch):
        # The regression this guards: both days carry m == 1.0, so any reader
        # keying on m alone treats them as the same observation.
        degraded = _write_artifact(
            monkeypatch, tmp_path / "a", 1.0, "computation_error"
        )
        clean = _write_artifact(monkeypatch, tmp_path / "b", 1.0, None)
        assert degraded["m"] == clean["m"]
        assert degraded["m_degraded"] != clean["m_degraded"]

    def test_a_real_multiplier_still_records_no_reason(
        self, tmp_path, monkeypatch
    ):
        data = _write_artifact(monkeypatch, tmp_path / "half", 0.5, None)
        assert data["m"] == 0.5
        assert data["m_degraded"] is None
