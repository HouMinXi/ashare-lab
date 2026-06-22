"""Tests for track_verdict.py -- generic shadow-track verdict driver.

Covers:
  1. run_track_verdict calls run_full_walk_forward with the given signal_fn
  2. signal_fn=None (R14 identity) passes None to walk-forward
  3. Output filename is verdict_{track_key}.json (not verdict.json)
  4. Verdict mean_rank_ic matches walk-forward window ICs
  5. write_verdict name= parameter backward compatibility (verdict.json default)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: fake config matching baseline.yaml schema
# ---------------------------------------------------------------------------

_FAKE_CONFIG = {
    "universe": {"primary": "csi1000"},
    "strategy": {"main": {"n_drop": 1}, "topk": 15},
    "gate": {
        "min_mean_rank_ic": 0.02,
        "min_positive_excess_pct": 0.60,
        "border_pass_ic_upper": 0.05,
        "border_pass_excess_upper": 0.70,
    },
}


def _make_window_results(n: int = 4) -> list[dict]:
    """Build synthetic window results with ascending IC values."""
    return [
        {
            "window_id": i + 1,
            "mean_rank_ic": 0.03 + i * 0.01,
            "cumulative_excess_return": 0.05 + i * 0.02,
            "is_positive_excess": True,
            "lot_skip_count": 0,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test: signal_fn is forwarded correctly
# ---------------------------------------------------------------------------


class TestSignalFnForwarding:
    """run_track_verdict must pass signal_fn to run_full_walk_forward."""

    @patch("ashare_lab.research.track_verdict.write_verdict")
    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_signal_fn_forwarded(
        self, mock_config, mock_walk, mock_write, tmp_path
    ):
        mock_config.return_value = _FAKE_CONFIG
        mock_walk.return_value = (_make_window_results(), [])

        def my_transform(pred, window):
            return pred

        from ashare_lab.research.track_verdict import run_track_verdict

        run_track_verdict(
            track_key="regime",
            exp_dir=tmp_path,
            signal_fn=my_transform,
            results_dir=tmp_path,
        )

        _, kwargs = mock_walk.call_args
        assert kwargs["signal_transform"] is my_transform

    @patch("ashare_lab.research.track_verdict.write_verdict")
    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_none_signal_fn_for_r14(
        self, mock_config, mock_walk, mock_write, tmp_path
    ):
        """signal_fn=None -> walk-forward gets signal_transform=None."""
        mock_config.return_value = _FAKE_CONFIG
        mock_walk.return_value = (_make_window_results(), [])

        from ashare_lab.research.track_verdict import run_track_verdict

        run_track_verdict(
            track_key="r14",
            exp_dir=tmp_path,
            signal_fn=None,
            results_dir=tmp_path,
        )

        _, kwargs = mock_walk.call_args
        assert kwargs["signal_transform"] is None


# ---------------------------------------------------------------------------
# Test: output filename uses track_key
# ---------------------------------------------------------------------------


class TestOutputFilename:
    """verdict_{track_key}.json, not verdict.json."""

    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_regime_filename(self, mock_config, mock_walk, tmp_path):
        mock_config.return_value = _FAKE_CONFIG
        mock_walk.return_value = (_make_window_results(), [])

        from ashare_lab.research.track_verdict import run_track_verdict

        run_track_verdict(
            track_key="regime",
            exp_dir=tmp_path,
            results_dir=tmp_path,
        )

        assert (tmp_path / "verdict_regime.json").exists()
        assert not (tmp_path / "verdict.json").exists()

    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_r14_filename(self, mock_config, mock_walk, tmp_path):
        mock_config.return_value = _FAKE_CONFIG
        mock_walk.return_value = (_make_window_results(), [])

        from ashare_lab.research.track_verdict import run_track_verdict

        run_track_verdict(
            track_key="r14",
            exp_dir=tmp_path,
            signal_fn=None,
            results_dir=tmp_path,
        )

        assert (tmp_path / "verdict_r14.json").exists()


# ---------------------------------------------------------------------------
# Test: verdict IC matches walk-forward
# ---------------------------------------------------------------------------


class TestVerdictIcAccuracy:
    """mean_rank_ic in verdict must equal the mean of window ICs."""

    @patch("ashare_lab.research.track_verdict.write_verdict")
    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_ic_from_walk_forward(
        self, mock_config, mock_walk, mock_write, tmp_path
    ):
        mock_config.return_value = _FAKE_CONFIG
        results = _make_window_results(3)
        mock_walk.return_value = (results, [])

        from ashare_lab.research.track_verdict import run_track_verdict

        verdict = run_track_verdict(
            track_key="test",
            exp_dir=tmp_path,
            results_dir=tmp_path,
        )

        expected = sum(r["mean_rank_ic"] for r in results) / len(results)
        assert verdict["mean_rank_ic"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test: track string format
# ---------------------------------------------------------------------------


class TestTrackString:
    """Track string must be {track_key}_topk{topk}_ndrop{n_drop}."""

    @patch("ashare_lab.research.track_verdict.write_verdict")
    @patch("ashare_lab.research.track_verdict.run_full_walk_forward")
    @patch("ashare_lab.research.track_verdict.load_config")
    def test_track_format(
        self, mock_config, mock_walk, mock_write, tmp_path
    ):
        mock_config.return_value = _FAKE_CONFIG
        mock_walk.return_value = (_make_window_results(), [])

        from ashare_lab.research.track_verdict import run_track_verdict

        verdict = run_track_verdict(
            track_key="regime",
            exp_dir=tmp_path,
            results_dir=tmp_path,
        )

        assert verdict["track"] == "regime_topk15_ndrop1"


# ---------------------------------------------------------------------------
# Test: write_verdict name= backward compatibility
# ---------------------------------------------------------------------------


class TestWriteVerdictNameParam:
    """write_verdict(name=) defaults to 'verdict' for backward compat."""

    def test_default_name_is_verdict(self, tmp_path):
        from ashare_lab.research.verdict import write_verdict

        verdict_dict = {"gate": "PASS", "note": "test"}
        path = write_verdict(verdict_dict, tmp_path)
        assert path == tmp_path / "verdict.json"
        assert path.exists()

    def test_custom_name(self, tmp_path):
        from ashare_lab.research.verdict import write_verdict

        verdict_dict = {"gate": "FAIL", "note": "test"}
        path = write_verdict(verdict_dict, tmp_path, name="verdict_regime")
        assert path == tmp_path / "verdict_regime.json"
        assert path.exists()
        assert not (tmp_path / "verdict.json").exists()

    @pytest.mark.parametrize("bad_name", [
        "../escape",
        "sub/dir",
        "back\\slash",
        "..\\winesc",
    ])
    def test_rejects_path_traversal(self, tmp_path, bad_name):
        from ashare_lab.research.verdict import write_verdict

        with pytest.raises(ValueError, match="simple filename stem"):
            write_verdict({"gate": "PASS"}, tmp_path, name=bad_name)
