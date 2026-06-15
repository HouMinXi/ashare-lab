"""Unit tests for ashare_lab.research.supplementary.

All tests run without a qlib runtime (qlib imports deferred in implementation).
Tests use mocks and temporary directories to verify behavior without I/O side
effects on the real filesystem.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ashare_lab.research.supplementary import (
    finalize_artifacts,
    run_csi300_reference,
    run_slippage_sensitivity,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_window_result(
    window_id: int = 1,
    cumulative_excess_return: float = 0.05,
    is_positive_excess: bool = True,
    mean_rank_ic: float | None = 0.03,
    n_drop: int = 1,
    pred_path: str | None = None,
) -> dict:
    """Minimal WindowResult dict for testing supplementary functions."""
    return {
        "window_id": window_id,
        "universe": "csi500",
        "n_drop": n_drop,
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-06-30",
        "mean_rank_ic": mean_rank_ic,
        "cumulative_excess_return": cumulative_excess_return,
        "is_positive_excess": is_positive_excess,
        "lot_skip_count": 0,
        "pred_path": pred_path or "/tmp/w1_pred.parquet",
    }


@pytest.fixture()
def tmp_exp_dir():
    """Create a temporary experiment directory, clean up after test."""
    d = tempfile.mkdtemp(prefix="test_exp_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def models_dir(tmp_exp_dir):
    """Create models/ subdirectory under tmp_exp_dir."""
    m = tmp_exp_dir / "models"
    m.mkdir()
    return m


def _write_verdict_json(exp_dir: Path, gate: str = "PASS") -> None:
    """Write a minimal verdict.json to exp_dir for backfill tests."""
    verdict = {
        "gate": gate,
        "universe": "csi500",
        "track": "topk15_ndrop1",
        "mean_rank_ic": 0.025,
        "n_windows": 6,
        "n_positive_excess_windows": 4,
        "positive_excess_pct": 0.666,
        "window_details": [],
        "rejected_orders_lot_skip": 0,
        "slippage_sensitivity": None,
        "note": "Gate PASS: mean_rank_ic=0.025, positive_excess_pct=67%",
    }
    with (exp_dir / "verdict.json").open("w") as f:
        json.dump(verdict, f)


# ---------------------------------------------------------------------------
# Test 1: slippage baseline row has correct impact_cost and is_baseline=True
# ---------------------------------------------------------------------------


class TestSlippageSensitivityBaseline:
    def test_baseline_row_uses_config_slippage(self, tmp_exp_dir):
        """Baseline row must have impact_cost = cost_model.slippage, is_baseline=True."""
        _write_verdict_json(tmp_exp_dir)

        windows = [
            _make_window_result(window_id=1, cumulative_excess_return=0.04, pred_path="/tmp/w1.parquet"),
            _make_window_result(window_id=2, cumulative_excess_return=0.06, pred_path="/tmp/w2.parquet"),
        ]

        # Patch _rebacktest_window so no actual qlib call happens.
        def fake_rebacktest(window_result, n_drop, slippage_override):
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.03}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            from ashare_lab.config import load_config
            cfg = load_config()
            expected_baseline_slippage = cfg["cost_model"]["slippage"]

            out_path = run_slippage_sensitivity(windows, tmp_exp_dir)

        assert out_path.exists(), "slippage_sensitivity.csv should be written"
        df = pd.read_csv(out_path)

        baseline_rows = df[df["is_baseline"]]
        assert len(baseline_rows) == 1, "exactly one baseline row"
        assert abs(baseline_rows.iloc[0]["impact_cost"] - expected_baseline_slippage) < 1e-9
        assert bool(baseline_rows.iloc[0]["is_baseline"]) is True
        # Mean of 0.04 and 0.06 = 0.05
        assert abs(baseline_rows.iloc[0]["mean_cumulative_excess"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: non-baseline row triggers re-backtest and uses level's impact_cost
# ---------------------------------------------------------------------------


class TestSlippageSensitivityNonBaseline:
    def test_non_baseline_rows_use_level_values(self, tmp_exp_dir):
        """Non-baseline rows must have impact_cost from config slippage_sensitivity."""
        _write_verdict_json(tmp_exp_dir)

        windows = [
            _make_window_result(window_id=1, cumulative_excess_return=0.05, pred_path="/tmp/w1.parquet"),
        ]

        call_log = []

        def fake_rebacktest(window_result, n_drop, slippage_override):
            call_log.append(slippage_override)
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.03}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            from ashare_lab.config import load_config
            cfg = load_config()
            levels = cfg["slippage_sensitivity"]  # [0.0015, 0.002]

            out_path = run_slippage_sensitivity(windows, tmp_exp_dir)

        assert out_path.exists()
        # _rebacktest_window called once per window per level.
        assert len(call_log) == len(levels), "one re-backtest call per level"
        assert set(call_log) == set(levels), "levels passed as slippage_override"

        df = pd.read_csv(out_path)
        non_baseline = df[~df["is_baseline"]]
        assert len(non_baseline) == len(levels)
        assert set(non_baseline["impact_cost"].tolist()) == set(levels)


# ---------------------------------------------------------------------------
# Test 3: fragility positive -- rel_drop > threshold -> is_fragile=True
# ---------------------------------------------------------------------------


class TestFragilityPositive:
    def test_fragility_true_when_excess_drops_beyond_threshold(self, tmp_exp_dir):
        """is_fragile=True when excess drops > rel_drop_threshold vs baseline."""
        _write_verdict_json(tmp_exp_dir)

        # baseline excess = 0.10; level excess = 0.04
        # rel_drop = (0.10 - 0.04) / max(0.10, 0.0005) = 0.60 > 0.50 threshold
        windows = [
            _make_window_result(window_id=1, cumulative_excess_return=0.10, pred_path="/tmp/w1.parquet"),
        ]

        def fake_rebacktest(window_result, n_drop, slippage_override):
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.04}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            run_slippage_sensitivity(windows, tmp_exp_dir)

        # Check verdict.json backfill.
        with (tmp_exp_dir / "verdict.json").open() as f:
            verdict = json.load(f)

        assert verdict["slippage_sensitivity"]["is_fragile"] is True


# ---------------------------------------------------------------------------
# Test 4: fragility negative -- rel_drop <= threshold -> is_fragile=False
# ---------------------------------------------------------------------------


class TestFragilityNegative:
    def test_fragility_false_when_excess_within_threshold(self, tmp_exp_dir):
        """is_fragile=False when rel_drop <= rel_drop_threshold."""
        _write_verdict_json(tmp_exp_dir)

        # baseline excess = 0.10; level excess = 0.08
        # rel_drop = (0.10 - 0.08) / 0.10 = 0.20 <= 0.50 threshold
        windows = [
            _make_window_result(window_id=1, cumulative_excess_return=0.10, pred_path="/tmp/w1.parquet"),
        ]

        def fake_rebacktest(window_result, n_drop, slippage_override):
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.08}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            run_slippage_sensitivity(windows, tmp_exp_dir)

        with (tmp_exp_dir / "verdict.json").open() as f:
            verdict = json.load(f)

        assert verdict["slippage_sensitivity"]["is_fragile"] is False


# ---------------------------------------------------------------------------
# Test 5: finalize_artifacts raises FileNotFoundError when no pkl found
# ---------------------------------------------------------------------------


class TestFinalizeArtifacts:
    def test_no_pkl_raises_file_not_found(self, models_dir):
        """finalize_artifacts raises FileNotFoundError when models_dir has no w*.pkl."""
        with pytest.raises(FileNotFoundError):
            finalize_artifacts(models_dir)

    # Test 6: non-contiguous w3, w7 picks w7; ignores backup.pkl.
    def test_non_contiguous_picks_highest_ignores_non_pattern(self, models_dir, tmp_path):
        """w3.pkl and w7.pkl: picks w7; backup.pkl must be ignored."""
        (models_dir / "w3.pkl").write_bytes(b"dummy3")
        (models_dir / "w7.pkl").write_bytes(b"dummy7")
        (models_dir / "backup.pkl").write_bytes(b"shouldignore")

        # Patch MODELS_DIR to avoid writing to real models/ directory.
        fake_models_dir = tmp_path / "models_output"
        with patch("ashare_lab.research.supplementary.MODELS_DIR", fake_models_dir):
            dest = finalize_artifacts(models_dir)

        assert dest == fake_models_dir / "latest.pkl"
        assert dest.exists()
        # Content should match w7.pkl (b"dummy7"), not w3 or backup.
        assert dest.read_bytes() == b"dummy7"


# ---------------------------------------------------------------------------
# Test 7: verdict.json slippage backfill preserves the other 10 fields
# ---------------------------------------------------------------------------


class TestVerdictBackfill:
    def test_slippage_backfill_preserves_other_fields(self, tmp_exp_dir):
        """After backfill, all 11 verdict fields present; non-slippage fields unchanged."""
        original_fields = {
            "gate": "PASS",
            "universe": "csi500",
            "track": "topk15_ndrop1",
            "mean_rank_ic": 0.025,
            "n_windows": 6,
            "n_positive_excess_windows": 4,
            "positive_excess_pct": 0.666,
            "window_details": [{"window_id": 1}],
            "rejected_orders_lot_skip": 3,
            "slippage_sensitivity": None,
            "note": "Gate PASS: mean_rank_ic=0.025, positive_excess_pct=67%",
        }
        with (tmp_exp_dir / "verdict.json").open("w") as f:
            json.dump(original_fields, f)

        windows = [
            _make_window_result(window_id=1, cumulative_excess_return=0.05, pred_path="/tmp/w1.parquet"),
        ]

        def fake_rebacktest(window_result, n_drop, slippage_override):
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.04}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            run_slippage_sensitivity(windows, tmp_exp_dir)

        with (tmp_exp_dir / "verdict.json").open() as f:
            result = json.load(f)

        # All 11 fields must be present.
        expected_keys = {
            "gate", "universe", "track", "mean_rank_ic", "n_windows",
            "n_positive_excess_windows", "positive_excess_pct", "window_details",
            "rejected_orders_lot_skip", "slippage_sensitivity", "note",
        }
        assert set(result.keys()) == expected_keys, f"unexpected keys: {set(result.keys())}"

        # Non-slippage fields must be unchanged.
        for key in expected_keys - {"slippage_sensitivity"}:
            assert result[key] == original_fields[key], f"field {key!r} was mutated"

        # slippage_sensitivity must now be a dict (not None).
        assert isinstance(result["slippage_sensitivity"], dict)
        assert "levels" in result["slippage_sensitivity"]
        assert "is_fragile" in result["slippage_sensitivity"]


# ---------------------------------------------------------------------------
# Test 8: pred_path missing -> window skipped with warning, not exception
# ---------------------------------------------------------------------------


class TestMissingPredPath:
    def test_missing_pred_path_skips_with_warning_not_exception(self, tmp_exp_dir):
        """Window with nonexistent pred_path is skipped (logged), not raised."""
        _write_verdict_json(tmp_exp_dir)

        # Provide two windows: one with a missing pred_path, one with a mock-backed path.
        windows = [
            _make_window_result(
                window_id=1,
                cumulative_excess_return=0.05,
                pred_path="/nonexistent/path/w1.parquet",
            ),
            _make_window_result(
                window_id=2,
                cumulative_excess_return=0.06,
                pred_path="/tmp/w2.parquet",
            ),
        ]

        # Only window 2 will succeed; window 1's pred_path does not exist.
        def fake_rebacktest(window_result, n_drop, slippage_override):
            if window_result["window_id"] == 1:
                return None  # simulates missing pred_path -> skipped
            return {**window_result, "n_drop": n_drop, "cumulative_excess_return": 0.04}

        with patch(
            "ashare_lab.research.supplementary._rebacktest_window",
            side_effect=fake_rebacktest,
        ):
            # Must NOT raise; must return the output path.
            out_path = run_slippage_sensitivity(windows, tmp_exp_dir)

        assert isinstance(out_path, Path)
        # CSV must still be written (partial success is acceptable).
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Test 9: run_csi300_reference writes verdict_csi300.json and
#          verdict_csi300_control.json to separate files (NOT verdict.json)
# ---------------------------------------------------------------------------


class TestCsi300Reference:
    def test_csi300_reference_writes_two_separate_files(self, tmp_exp_dir):
        """run_csi300_reference writes verdict_csi300.json + verdict_csi300_control.json."""
        from ashare_lab.config import load_config
        cfg = load_config()
        topk = cfg["strategy"]["topk"]
        main_n_drop = cfg["strategy"]["main"]["n_drop"]
        control_n_drop = cfg["strategy"]["control"]["n_drop"]
        ref_universe = cfg["universe"]["reference"]

        # Fake walk-forward returns one window result for CSI300.
        fake_csi300_window = _make_window_result(
            window_id=1,
            cumulative_excess_return=0.04,
            pred_path=str(tmp_exp_dir / ref_universe / "predictions" / "w1_pred.parquet"),
        )
        # Update universe to ref_universe.
        fake_csi300_window["universe"] = ref_universe

        def fake_rebacktest(window_result, n_drop, slippage_override):
            return {
                **window_result,
                "n_drop": n_drop,
                "cumulative_excess_return": 0.03,
                "is_positive_excess": True,
            }

        with (
            patch(
                "ashare_lab.research.rolling.run_full_walk_forward",
                return_value=[fake_csi300_window],
            ),
            patch(
                "ashare_lab.research.supplementary._rebacktest_window",
                side_effect=fake_rebacktest,
            ),
        ):
            v_path, vc_path = run_csi300_reference(tmp_exp_dir)

        # Must be separate files (NOT verdict.json).
        assert v_path.name == "verdict_csi300.json"
        assert vc_path.name == "verdict_csi300_control.json"
        assert v_path != vc_path

        # verdict.json must NOT be written/modified by this function.
        assert not (tmp_exp_dir / "verdict.json").exists()

        # Both files must be valid JSON.
        with v_path.open() as f:
            v = json.load(f)
        with vc_path.open() as f:
            vc = json.load(f)

        # Check track names and universe.
        assert v["track"] == f"topk{topk}_ndrop{main_n_drop}"
        assert vc["track"] == f"topk{topk}_ndrop{control_n_drop}"
        assert v["universe"] == ref_universe
        assert vc["universe"] == ref_universe

        # Note fields must be prefixed with REFERENCE TRACK.
        assert v["note"].startswith(f"REFERENCE TRACK ({ref_universe})")
        assert vc["note"].startswith(f"REFERENCE TRACK ({ref_universe})")
