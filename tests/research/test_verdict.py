"""Unit tests for ashare_lab.research.verdict.

Covers compute_gate(), build_verdict(), write_verdict(), and
write_ic_csv_png(). All tests run without a qlib runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.research.verdict import (
    _nan_to_none,
    _serialize_verdict,
    build_verdict,
    compute_gate,
    write_ic_csv_png,
    write_verdict,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GATE_CFG = {
    "min_mean_rank_ic": 0.02,
    "min_positive_excess_pct": 0.60,
    "border_pass_ic_upper": 0.022,
    "border_pass_excess_upper": 0.66,
}


def _make_window(
    window_id: int = 1,
    mean_rank_ic: float | None = 0.03,
    cumulative_excess_return: float = 0.05,
    is_positive_excess: bool = True,
    lot_skip_count: int | None = 0,
) -> dict:
    """Minimal WindowResult dict for testing build_verdict."""
    return {
        "window_id": window_id,
        "mean_rank_ic": mean_rank_ic,
        "cumulative_excess_return": cumulative_excess_return,
        "is_positive_excess": is_positive_excess,
        "lot_skip_count": lot_skip_count,
        "universe": "csi500",
        "n_drop": 1,
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-06-30",
        "pred_path": "/tmp/w1_pred.parquet",
    }


# ---------------------------------------------------------------------------
# Tests for compute_gate
# ---------------------------------------------------------------------------


class TestComputeGate:
    def test_pass_boundary(self):
        """IC >= 0.022 AND pct >= 0.66 -> PASS."""
        assert compute_gate(0.022, 0.66, GATE_CFG) == "PASS"

    def test_pass_well_above_threshold(self):
        """Strong IC and high positive rate -> PASS."""
        assert compute_gate(0.05, 0.80, GATE_CFG) == "PASS"

    def test_fail_nan_ic(self):
        """NaN IC -> FAIL (rule 1: missing IC)."""
        assert compute_gate(float("nan"), 0.70, GATE_CFG) == "FAIL"

    def test_fail_none_ic(self):
        """None IC -> FAIL (rule 1: missing IC)."""
        assert compute_gate(None, 0.70, GATE_CFG) == "FAIL"

    def test_fail_ic_exactly_at_threshold(self):
        """IC == 0.02 (not strictly greater) -> FAIL (rule 2: boundary)."""
        assert compute_gate(0.02, 0.70, GATE_CFG) == "FAIL"

    def test_fail_ic_below_threshold(self):
        """IC below minimum -> FAIL."""
        assert compute_gate(0.01, 0.80, GATE_CFG) == "FAIL"

    def test_fail_pct_below_threshold(self):
        """Good IC but pos_pct below 0.60 -> FAIL."""
        assert compute_gate(0.03, 0.50, GATE_CFG) == "FAIL"

    def test_fail_pct_exactly_at_minimum(self):
        """pos_pct == 0.60 just meets the FAIL threshold -- should not FAIL
        on pct alone; must pass rule 2 pct check."""
        # IC > 0.02, pct = 0.60 (>= 0.60 passes rule 2 pct check).
        # IC = 0.021 < 0.022 -> BORDER_PASS (rule 3).
        result = compute_gate(0.021, 0.60, GATE_CFG)
        assert result == "BORDER_PASS"

    def test_border_pass_ic_below_upper(self):
        """IC > 0.02 but < 0.022, pct >= 0.66 -> BORDER_PASS (rule 3)."""
        assert compute_gate(0.021, 0.66, GATE_CFG) == "BORDER_PASS"

    def test_border_pass_pct_below_upper(self):
        """IC >= 0.022, pct >= 0.60 but < 0.66 -> BORDER_PASS (rule 3)."""
        assert compute_gate(0.022, 0.62, GATE_CFG) == "BORDER_PASS"

    def test_border_pass_both_below_upper(self):
        """IC in (0.02, 0.022) AND pct in [0.60, 0.66) -> BORDER_PASS."""
        assert compute_gate(0.021, 0.60, GATE_CFG) == "BORDER_PASS"

    def test_thresholds_read_from_config(self):
        """Thresholds are read from gate_config, not from internal defaults."""
        # Custom thresholds: min_ic=0.05, border_ic=0.06
        custom = {
            "min_mean_rank_ic": 0.05,
            "min_positive_excess_pct": 0.50,
            "border_pass_ic_upper": 0.06,
            "border_pass_excess_upper": 0.70,
        }
        # IC=0.03 <= 0.05 -> FAIL (even though > default 0.02)
        assert compute_gate(0.03, 0.80, custom) == "FAIL"
        # IC=0.055 > 0.05, pct=0.65 < 0.70 -> BORDER_PASS
        assert compute_gate(0.055, 0.65, custom) == "BORDER_PASS"
        # IC=0.06, pct=0.70 -> PASS
        assert compute_gate(0.06, 0.70, custom) == "PASS"


# ---------------------------------------------------------------------------
# Tests for build_verdict
# ---------------------------------------------------------------------------


class TestBuildVerdict:
    def _call(self, windows: list[dict], gate_cfg: dict = GATE_CFG) -> dict:
        return build_verdict(windows, track="topk15_ndrop1", universe="csi500", gate_config=gate_cfg)

    def test_schema_completeness(self):
        """Verdict dict contains all 12 fields (11 original + failed_windows)."""
        windows = [_make_window(1, 0.03, 0.05, True, 2)]
        result = self._call(windows)
        expected_keys = {
            "gate", "universe", "track", "mean_rank_ic", "n_windows",
            "n_positive_excess_windows", "positive_excess_pct",
            "window_details", "rejected_orders_lot_skip",
            "slippage_sensitivity", "failed_windows", "note",
        }
        assert set(result.keys()) == expected_keys

    def test_pass_verdict(self):
        """All windows positive, high IC -> gate=PASS."""
        windows = [
            _make_window(i, 0.025, 0.05, True, 1) for i in range(1, 7)
        ]
        result = self._call(windows)
        assert result["gate"] == "PASS"
        assert result["n_windows"] == 6
        assert result["n_positive_excess_windows"] == 6
        assert result["positive_excess_pct"] == pytest.approx(1.0)

    def test_fail_verdict_low_ic(self):
        """Low IC -> gate=FAIL."""
        windows = [_make_window(1, 0.01, 0.05, True, 0)]
        result = self._call(windows)
        assert result["gate"] == "FAIL"

    def test_fail_verdict_all_nan_ic(self):
        """All windows have None IC -> mean_rank_ic=None -> gate=FAIL."""
        windows = [
            _make_window(i, None, 0.03, True, 0) for i in range(1, 4)
        ]
        result = self._call(windows)
        assert result["gate"] == "FAIL"
        assert result["mean_rank_ic"] is None

    def test_border_pass_verdict(self):
        """IC in (0.02, 0.022) with sufficient pos_pct -> gate=BORDER_PASS."""
        windows = [
            _make_window(i, 0.021, 0.03, True, 0) for i in range(1, 5)
        ]
        result = self._call(windows)
        assert result["gate"] == "BORDER_PASS"

    def test_note_field_format(self):
        """note field contains gate label, IC, and pct in human-readable form."""
        windows = [_make_window(1, 0.03, 0.05, True, 0)]
        result = self._call(windows)
        assert "Gate" in result["note"]
        assert result["gate"] in result["note"]
        assert "mean_rank_ic" in result["note"]
        assert "positive_excess_pct" in result["note"]

    def test_track_field_stored_as_passed(self):
        """track field is the caller's string, not reconstructed."""
        windows = [_make_window(1, 0.03, 0.05, True, 0)]
        result = build_verdict(
            windows, track="topk20_ndrop3", universe="csi300", gate_config=GATE_CFG
        )
        assert result["track"] == "topk20_ndrop3"
        assert result["universe"] == "csi300"

    def test_slippage_sensitivity_is_none(self):
        """slippage_sensitivity is always None (backfilled by 02-04)."""
        windows = [_make_window(1, 0.03, 0.05, True, 0)]
        result = self._call(windows)
        assert result["slippage_sensitivity"] is None

    def test_empty_window_results(self):
        """Empty list -> n_windows=0, pos_pct=0, mean_ic=None, gate=FAIL."""
        result = self._call([])
        assert result["n_windows"] == 0
        assert result["positive_excess_pct"] == 0.0
        assert result["mean_rank_ic"] is None
        assert result["gate"] == "FAIL"
        assert result["window_details"] == []

    def test_none_ic_windows_in_denominator(self):
        """None-IC windows count in n_windows and pos_pct denominator."""
        # 4 windows: 3 positive, 1 None-IC but also positive
        windows = [
            _make_window(1, 0.03, 0.05, True, 0),
            _make_window(2, 0.03, 0.05, True, 0),
            _make_window(3, 0.03, 0.05, True, 0),
            _make_window(4, None, 0.01, True, 0),  # None IC, positive
        ]
        result = self._call(windows)
        assert result["n_windows"] == 4
        # mean IC computed from 3 valid windows only
        assert result["mean_rank_ic"] == pytest.approx(0.03, rel=1e-6)
        # pos_pct denominator = 4; all 4 are positive
        assert result["positive_excess_pct"] == pytest.approx(1.0)
        assert result["n_positive_excess_windows"] == 4

    def test_rejected_orders_lot_skip_sum(self):
        """rejected_orders_lot_skip sums non-None lot_skip_count values."""
        windows = [
            _make_window(1, 0.03, 0.05, True, 5),
            _make_window(2, 0.03, 0.05, True, 3),
            _make_window(3, 0.03, 0.05, True, 2),
        ]
        result = self._call(windows)
        assert result["rejected_orders_lot_skip"] == 10

    def test_rejected_orders_lot_skip_all_none(self):
        """rejected_orders_lot_skip=None when all lot_skip_count are None."""
        windows = [
            _make_window(1, 0.03, 0.05, True, None),
            _make_window(2, 0.03, 0.05, True, None),
        ]
        result = self._call(windows)
        assert result["rejected_orders_lot_skip"] is None

    def test_window_details_none_safe_ic(self):
        """window_details preserves None mean_rank_ic without error."""
        windows = [
            _make_window(1, None, 0.03, False, 0),
            _make_window(2, 0.04, 0.05, True, 1),
        ]
        result = self._call(windows)
        details = result["window_details"]
        assert len(details) == 2
        assert details[0]["mean_rank_ic"] is None
        assert details[1]["mean_rank_ic"] == pytest.approx(0.04)

    def test_mixed_nan_ic_correct_mean(self):
        """Mixed None and valid IC -> mean computed from valid values only."""
        windows = [
            _make_window(1, None, 0.03, True, 0),
            _make_window(2, 0.04, 0.05, True, 0),
            _make_window(3, 0.06, 0.08, True, 0),
        ]
        result = self._call(windows)
        # mean of [0.04, 0.06] = 0.05
        assert result["mean_rank_ic"] == pytest.approx(0.05, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests for write_verdict (file creation and JSON content)
# ---------------------------------------------------------------------------


class TestWriteVerdict:
    def test_file_created(self, tmp_path: Path):
        """write_verdict creates verdict.json in exp_dir."""
        verdict = build_verdict(
            [_make_window(1, 0.03, 0.05, True, 2)],
            track="topk15_ndrop1",
            universe="csi500",
            gate_config=GATE_CFG,
        )
        out = write_verdict(verdict, tmp_path)
        assert out == tmp_path / "verdict.json"
        assert out.exists()

    def test_json_parseable(self, tmp_path: Path):
        """verdict.json must be valid JSON."""
        verdict = build_verdict(
            [_make_window(1, 0.03, 0.05, True, 0)],
            track="topk15_ndrop1",
            universe="csi500",
            gate_config=GATE_CFG,
        )
        write_verdict(verdict, tmp_path)
        data = json.loads((tmp_path / "verdict.json").read_text())
        assert data["gate"] in {"PASS", "FAIL", "BORDER_PASS"}

    def test_nan_serialized_as_null(self, tmp_path: Path):
        """NaN mean_rank_ic must appear as JSON null (not literal NaN)."""
        verdict = build_verdict(
            [_make_window(1, None, 0.03, False, 0)],
            track="topk15_ndrop1",
            universe="csi500",
            gate_config=GATE_CFG,
        )
        write_verdict(verdict, tmp_path)
        text = (tmp_path / "verdict.json").read_text()
        # JSON standard: NaN -> null (not NaN literal, which is invalid JSON)
        data = json.loads(text)
        assert data["mean_rank_ic"] is None

    def test_all_11_fields_present(self, tmp_path: Path):
        """Loaded JSON has all 11 required fields."""
        verdict = build_verdict(
            [_make_window(1, 0.025, 0.05, True, 1)],
            track="topk15_ndrop1",
            universe="csi500",
            gate_config=GATE_CFG,
        )
        write_verdict(verdict, tmp_path)
        data = json.loads((tmp_path / "verdict.json").read_text())
        for field in [
            "gate", "universe", "track", "mean_rank_ic", "n_windows",
            "n_positive_excess_windows", "positive_excess_pct",
            "window_details", "rejected_orders_lot_skip",
            "slippage_sensitivity", "note",
        ]:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests for write_ic_csv_png
# ---------------------------------------------------------------------------


class TestWriteIcCsvPng:
    def test_creates_both_files(self, tmp_path: Path):
        """write_ic_csv_png creates ic_windows.csv and ic_windows.png."""
        windows = [_make_window(i, 0.03, 0.05, True, 0) for i in range(1, 4)]
        csv_path, png_path = write_ic_csv_png(windows, tmp_path, GATE_CFG)
        assert csv_path.exists()
        assert png_path.exists()
        assert csv_path == tmp_path / "ic_windows.csv"
        assert png_path == tmp_path / "ic_windows.png"

    def test_csv_columns(self, tmp_path: Path):
        """CSV has exactly: window_id, mean_rank_ic, cumulative_excess_return, is_positive_excess."""
        windows = [_make_window(1, 0.03, 0.05, True, 2)]
        csv_path, _ = write_ic_csv_png(windows, tmp_path, GATE_CFG)
        df = pd.read_csv(csv_path)
        assert list(df.columns) == [
            "window_id",
            "mean_rank_ic",
            "cumulative_excess_return",
            "is_positive_excess",
        ]

    def test_empty_window_results(self, tmp_path: Path):
        """Empty window_results -> CSV with header only, PNG created."""
        csv_path, png_path = write_ic_csv_png([], tmp_path, GATE_CFG)
        assert csv_path.exists()
        assert png_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 0
        assert list(df.columns) == [
            "window_id",
            "mean_rank_ic",
            "cumulative_excess_return",
            "is_positive_excess",
        ]

    def test_csv_no_index_column(self, tmp_path: Path):
        """CSV must not include an unnamed index column (index=False)."""
        windows = [_make_window(1, 0.03, 0.05, True, 0)]
        csv_path, _ = write_ic_csv_png(windows, tmp_path, GATE_CFG)
        df = pd.read_csv(csv_path)
        # If index was written, columns would include "Unnamed: 0"
        assert not any("Unnamed" in c for c in df.columns)


# ---------------------------------------------------------------------------
# Tests for _serialize_verdict / NaN handling
# ---------------------------------------------------------------------------


class TestSerializeVerdict:
    def test_nan_float_to_none(self):
        """Python float NaN -> None via _nan_to_none."""
        result = _nan_to_none({"a": float("nan"), "b": 1.0})
        assert result["a"] is None
        assert result["b"] == 1.0

    def test_nested_nan_to_none(self):
        """Nested NaN in lists and dicts -> None."""
        obj = {"details": [{"ic": float("nan")}, {"ic": 0.03}]}
        result = _nan_to_none(obj)
        assert result["details"][0]["ic"] is None
        assert result["details"][1]["ic"] == pytest.approx(0.03)

    def test_serialize_verdict_nan_to_null_json(self, tmp_path: Path):
        """Full round-trip: verdict with None IC -> verdict.json with null."""
        windows = [_make_window(1, None, -0.01, False, 0)]
        verdict = build_verdict(
            windows, track="topk15_ndrop1", universe="csi500", gate_config=GATE_CFG
        )
        write_verdict(verdict, tmp_path)
        data = json.loads((tmp_path / "verdict.json").read_text())
        # mean_rank_ic was None -> should serialize to null
        assert data["mean_rank_ic"] is None
        # window_details[0].mean_rank_ic was None -> null
        assert data["window_details"][0]["mean_rank_ic"] is None

    def test_serialize_verdict_numpy_types(self):
        """numpy integer/float types are converted to Python natives."""
        import numpy as np

        raw = {
            "gate": "PASS",
            "mean_rank_ic": np.float64(0.03),
            "n_windows": np.int64(6),
            "nested": [np.float64(0.01)],
        }
        serialized = _serialize_verdict(raw)
        assert isinstance(serialized["mean_rank_ic"], float)
        assert not isinstance(serialized["mean_rank_ic"], np.floating)
        assert isinstance(serialized["n_windows"], int)
        assert not isinstance(serialized["n_windows"], np.integer)
        # JSON round-trip must not raise
        json.dumps(serialized)

    def test_window_details_nan_to_none_in_json(self, tmp_path: Path):
        """window_details with NaN mean_rank_ic serializes to null in JSON."""
        # Manually construct verdict without passing through build_verdict's
        # None-gate (build_verdict filters via pd.notna; float NaN is also
        # filtered, so mean_rank_ic would be None from build_verdict already).
        # Test the serialization path directly.
        verdict = {
            "gate": "PASS",
            "universe": "csi500",
            "track": "topk15_ndrop1",
            "mean_rank_ic": float("nan"),
            "n_windows": 1,
            "n_positive_excess_windows": 1,
            "positive_excess_pct": 1.0,
            "window_details": [{"window_id": 1, "mean_rank_ic": float("nan")}],
            "rejected_orders_lot_skip": None,
            "slippage_sensitivity": None,
            "note": "test",
        }
        write_verdict(verdict, tmp_path)
        data = json.loads((tmp_path / "verdict.json").read_text())
        assert data["mean_rank_ic"] is None
        assert data["window_details"][0]["mean_rank_ic"] is None

    def test_serialize_does_not_raise_on_math_nan(self):
        """_serialize_verdict on a dict with math.nan must not raise."""
        obj = {"x": math.nan}
        result = _serialize_verdict(obj)
        assert result["x"] is None
        # Must be JSON-serializable
        json.dumps(result)
