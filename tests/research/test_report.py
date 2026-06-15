"""Stub unit tests for ashare_lab.research.report.

These cover the three public functions in report.py at a basic smoke level.
Full integration tests require a live qlib runtime and are out of scope
for unit testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest


class TestMakeExpDir:
    def test_creates_directory(self, tmp_path):
        """make_exp_dir creates a timestamped subdirectory under experiments/."""
        from ashare_lab.research.report import make_exp_dir
        from ashare_lab.config import PROJECT_ROOT

        # Redirect experiments/ to tmp_path so we don't pollute the real tree.
        fake_experiments = tmp_path / "experiments"
        with mock.patch("ashare_lab.research.report.PROJECT_ROOT", tmp_path):
            result = make_exp_dir()

        assert result.exists()
        assert result.is_dir()
        assert result.name.startswith("baseline_")
        assert len(result.name) == len("baseline_YYYYMMDD_HHMMSS")

    def test_returns_path(self, tmp_path):
        """make_exp_dir returns a Path."""
        from ashare_lab.research.report import make_exp_dir

        with mock.patch("ashare_lab.research.report.PROJECT_ROOT", tmp_path):
            result = make_exp_dir()
        assert isinstance(result, Path)


class TestRunVerdictPhase:
    def test_fail_gate_calls_sys_exit(self, tmp_path):
        """run_verdict_phase calls sys.exit(1) when gate is FAIL."""
        import json
        from ashare_lab.research.report import run_verdict_phase

        cfg = {
            "gate": {"min_mean_rank_ic": 0.02, "min_positive_excess_pct": 0.60,
                     "border_pass_ic_upper": 0.022, "border_pass_excess_upper": 0.66},
            "universe": {"primary": "csi500"},
            "strategy": {"topk": 15, "main": {"n_drop": 1}},
        }
        with mock.patch("ashare_lab.research.report.load_config", return_value=cfg), \
             mock.patch("ashare_lab.research.report.write_verdict"), \
             mock.patch("ashare_lab.research.report.write_ic_csv_png"), \
             mock.patch("ashare_lab.research.report.shutil.copy"), \
             pytest.raises(SystemExit) as exc_info:
            run_verdict_phase([], tmp_path, n_drop=1)

        assert exc_info.value.code == 1

    def test_pass_gate_returns_verdict(self, tmp_path):
        """run_verdict_phase returns verdict dict on PASS gate."""
        from ashare_lab.research.report import run_verdict_phase

        windows = [
            {"window_id": i, "mean_rank_ic": 0.025, "cumulative_excess_return": 0.01,
             "is_positive_excess": True, "lot_skip_count": 0}
            for i in range(1, 7)
        ]
        cfg = {
            "gate": {"min_mean_rank_ic": 0.02, "min_positive_excess_pct": 0.60,
                     "border_pass_ic_upper": 0.022, "border_pass_excess_upper": 0.66},
            "universe": {"primary": "csi500"},
            "strategy": {"topk": 15, "main": {"n_drop": 1}},
        }
        with mock.patch("ashare_lab.research.report.load_config", return_value=cfg), \
             mock.patch("ashare_lab.research.report.write_verdict"), \
             mock.patch("ashare_lab.research.report.write_ic_csv_png"), \
             mock.patch("ashare_lab.research.report.shutil.copy"), \
             mock.patch("ashare_lab.research.report.pd.isna", return_value=False):
            result = run_verdict_phase(windows, tmp_path, n_drop=1)

        assert result["gate"] in ("PASS", "BORDER_PASS")
        assert "mean_rank_ic" in result
