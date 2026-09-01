"""Tests for the train-window sweep plumbing (Phase 12).

Covers the two new matrix_runner seams (candidate walk_forward overlay,
portfolio turnover extraction), the run_cell call-site wiring that feeds
them, and analyze_matrix's tolerance for jsonl rows with and without the
optional turnover key.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ashare_lab import config as cfg
from ashare_lab.research.analyze_matrix import generate_gate_decision
from ashare_lab.research.matrix_runner import (
    extract_mean_turnover,
    merge_candidate_config,
    run_cell,
)
from ashare_lab.research.metrics import is_valid_turnover
from ashare_lab.research.smoke_test import get_window

BASELINE_PATH = Path(__file__).resolve().parents[2] / "configs" / "baseline.yaml"


def _load_baseline() -> dict:
    with BASELINE_PATH.open() as f:
        return yaml.safe_load(f)


class TestIsValidTurnover:
    def test_accepts_python_and_numpy_numerics(self) -> None:
        assert is_valid_turnover(0.25)
        assert is_valid_turnover(2)
        assert is_valid_turnover(np.float64(0.25))
        assert is_valid_turnover(np.int64(3))

    def test_rejects_bool_nan_str_none(self) -> None:
        assert not is_valid_turnover(True)
        assert not is_valid_turnover(np.bool_(True))
        assert not is_valid_turnover(float("nan"))
        assert not is_valid_turnover(np.float64("nan"))
        assert not is_valid_turnover("0.25")
        assert not is_valid_turnover(None)


class TestMergeCandidateConfig:
    def test_model_only_overlay_preserves_base_content(self) -> None:
        base = _load_baseline()
        candidate = {"model": {"type": "tra", "tag": "x"}, "matrix": {"tag": "x"}}
        merged = merge_candidate_config(base, candidate)
        # Everything except the model section is identical to the baseline.
        expect = copy.deepcopy(base)
        expect["model"] = {"type": "tra", "tag": "x"}
        assert merged == expect

    def test_walk_forward_overlay_creates_missing_section(self) -> None:
        base = _load_baseline()
        del base["walk_forward"]
        candidate = {
            "model": {"type": "tra", "tag": "x"},
            "walk_forward": {"train_window_years": 1},
        }
        merged = merge_candidate_config(base, candidate)
        assert merged["walk_forward"] == {"train_window_years": 1}

    def test_candidate_model_section_is_copied(self) -> None:
        base = _load_baseline()
        candidate = {"model": {"type": "tra", "tag": "x"}}
        merged = merge_candidate_config(base, candidate)
        candidate["model"]["tag"] = "mutated"
        assert merged["model"]["tag"] == "x"

    def test_explicit_none_walk_forward_preserves_base(self) -> None:
        # walk_forward: null in YAML lands as an explicit None, which the
        # merge treats the same as an absent key: base section kept.
        base = _load_baseline()
        candidate = {
            "model": {"type": "tra", "tag": "x"},
            "walk_forward": None,
        }
        merged = merge_candidate_config(base, candidate)
        assert merged["walk_forward"] == base["walk_forward"]

    def test_walk_forward_overlay_merges_keywise(self) -> None:
        base = _load_baseline()
        base_before = copy.deepcopy(base)
        candidate = {
            "model": {"type": "tra", "tag": "win1_tra"},
            "walk_forward": {"train_window_years": 1},
        }
        merged = merge_candidate_config(base, candidate)
        assert merged["walk_forward"]["train_window_years"] == 1
        # Other walk_forward keys survive the key-wise merge.
        assert merged["walk_forward"]["step_months"] == 6
        assert merged["walk_forward"]["train_start"] == "2018-01-01"
        # The input dict is not mutated.
        assert base == base_before

    def test_overlay_shifts_get_window_train_start(self, tmp_path, monkeypatch) -> None:
        """Merged config with train_window_years=1 moves W1 train_start.

        train_end is base_train_end (2020-12-31); minus 1 year gives
        2019-12-31, above the 2018-01-01 safety floor.
        """
        base = _load_baseline()
        candidate = {
            "model": dict(base["model"]),
            "walk_forward": {"train_window_years": 1},
        }
        merged = merge_candidate_config(base, candidate)
        tmp_cfg = tmp_path / "merged.yaml"
        tmp_cfg.write_text(yaml.safe_dump(merged))

        monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_cfg)
        cfg.load_config.cache_clear()
        try:
            w = get_window(0)
        finally:
            cfg.load_config.cache_clear()
        assert w["train_start"] == "2019-12-31"
        assert w["train_end"] == "2020-12-31"


class TestRunCellWiring:
    """Proves run_cell actually calls the new seams (not just that the
    helpers work in isolation). All heavy/GPU stages are faked; the config
    load path and window resolution run for real."""

    def test_overlay_and_turnover_reach_record(self, tmp_path, monkeypatch) -> None:
        import ashare_lab.research.backtest as backtest_mod  # noqa: PLC0415
        import ashare_lab.research.train as train_mod  # noqa: PLC0415
        from ashare_lab.research import smoke_test as st  # noqa: PLC0415

        baseline = _load_baseline()
        candidate = {
            "model": dict(baseline["model"], tag="win1_tra"),
            "matrix": {"tag": "win1_tra"},
            "walk_forward": {"train_window_years": 1},
        }
        cand_path = tmp_path / "cand.yaml"
        cand_path.write_text(yaml.safe_dump(candidate))

        # Spy on window resolution: captures the walk_forward section of the
        # config that is LIVE at the moment get_window runs.
        seen: dict = {}
        real_get_window = st.get_window

        def spy_get_window(step: int) -> dict:
            seen["walk_forward"] = dict(cfg.load_config()["walk_forward"])
            return real_get_window(step)

        monkeypatch.setattr(st, "get_window", spy_get_window)

        # Fake the GPU stages. pred/label: 2 dates x 5 instruments so
        # daily_rank_ic has enough cross-section per date.
        dates = pd.date_range("2021-07-01", periods=2, freq="B")
        instruments = [f"SH60000{i}" for i in range(5)]
        midx = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred = pd.Series(range(len(midx)), index=midx, dtype=float)
        label = pd.Series(range(len(midx)), index=midx, dtype=float) * 0.01

        def fake_train_window(window, exp_dir, universe, seed=None):
            return tmp_path / "model.pkl", pred, label

        bench_idx = pd.date_range("2021-07-01", periods=2, freq="B")
        portfolio_df = pd.DataFrame(
            {"return": [0.01, -0.005], "turnover": [0.20, 0.40]},
            index=bench_idx,
        )
        bench_close = pd.Series([100.0, 101.0], index=bench_idx)

        def fake_run_backtest(window, pred_arg, n_drop, slippage_override=None):
            return portfolio_df, bench_close, 0

        monkeypatch.setattr(train_mod, "train_window", fake_train_window)
        monkeypatch.setattr(backtest_mod, "run_backtest", fake_run_backtest)
        # Pin the baseline explicitly instead of relying on the module default.
        monkeypatch.setattr(cfg, "CONFIG_PATH", BASELINE_PATH)

        record = run_cell(
            config_path=str(cand_path),
            window_id=1,
            output_dir=str(tmp_path / "out"),
        )

        # The overlay reached the live config before window resolution.
        assert seen["walk_forward"]["train_window_years"] == 1
        # Base walk_forward keys survive the key-wise merge.
        assert seen["walk_forward"]["step_months"] == 6
        assert seen["walk_forward"]["train_start"] == "2018-01-01"
        # The turnover column reached the record.
        assert record["turnover"] == pytest.approx(0.30)
        assert record["model"] == "win1_tra"
        assert record["window"] == 1
        # pred and label are perfectly monotone, so rank IC is 1.0.
        assert record["ic"] == pytest.approx(1.0)
        for key in ("excess", "maxdd", "completed_at"):
            assert key in record

    def test_turnover_key_omitted_when_unavailable(
        self, tmp_path, monkeypatch
    ) -> None:
        import ashare_lab.research.backtest as backtest_mod  # noqa: PLC0415
        import ashare_lab.research.train as train_mod  # noqa: PLC0415

        baseline = _load_baseline()
        candidate = {
            "model": dict(baseline["model"], tag="win1_tra"),
            "matrix": {"tag": "win1_tra"},
        }
        cand_path = tmp_path / "cand.yaml"
        cand_path.write_text(yaml.safe_dump(candidate))

        dates = pd.date_range("2021-07-01", periods=2, freq="B")
        instruments = [f"SH60000{i}" for i in range(5)]
        midx = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred = pd.Series(range(len(midx)), index=midx, dtype=float)
        label = pd.Series(range(len(midx)), index=midx, dtype=float) * 0.01

        def fake_train_window(window, exp_dir, universe, seed=None):
            return tmp_path / "model.pkl", pred, label

        bench_idx = pd.date_range("2021-07-01", periods=2, freq="B")
        # No turnover column at all.
        portfolio_df = pd.DataFrame({"return": [0.01, -0.005]}, index=bench_idx)
        bench_close = pd.Series([100.0, 101.0], index=bench_idx)

        def fake_run_backtest(window, pred_arg, n_drop, slippage_override=None):
            return portfolio_df, bench_close, 0

        monkeypatch.setattr(train_mod, "train_window", fake_train_window)
        monkeypatch.setattr(backtest_mod, "run_backtest", fake_run_backtest)
        monkeypatch.setattr(cfg, "CONFIG_PATH", BASELINE_PATH)

        record = run_cell(
            config_path=str(cand_path),
            window_id=1,
            output_dir=str(tmp_path / "out"),
        )

        assert "turnover" not in record


class TestExtractMeanTurnover:
    def test_normal_column(self) -> None:
        df = pd.DataFrame({"turnover": [0.10, 0.20, 0.30]})
        assert extract_mean_turnover(df) == pytest.approx(0.20)

    def test_single_element_column(self) -> None:
        df = pd.DataFrame({"turnover": [0.50]})
        assert extract_mean_turnover(df) == pytest.approx(0.50)

    def test_missing_column_returns_none(self) -> None:
        df = pd.DataFrame({"return": [0.01, 0.02]})
        assert extract_mean_turnover(df) is None

    def test_empty_frame_returns_none(self) -> None:
        df = pd.DataFrame({"turnover": pd.Series(dtype=float)})
        assert extract_mean_turnover(df) is None

    def test_nan_only_column_returns_none(self) -> None:
        df = pd.DataFrame({"turnover": [float("nan"), float("nan")]})
        assert extract_mean_turnover(df) is None

    def test_non_numeric_column_returns_none(self) -> None:
        df = pd.DataFrame({"turnover": ["n/a", "n/a"]})
        assert extract_mean_turnover(df) is None

    def test_bool_column_returns_none(self) -> None:
        # The consumer side (generate_gate_decision) drops bools too.
        df = pd.DataFrame({"turnover": [True, False]})
        assert extract_mean_turnover(df) is None


class TestAnalyzeMatrixTurnoverTolerance:
    @staticmethod
    def _record(tag: str, window: int, turnover: float | None = None) -> dict:
        rec = {
            "model": tag,
            "window": window,
            "ic": 0.05,
            "excess": 0.03,
            "maxdd": -0.10,
            "completed_at": "2026-09-01T00:00:00+00:00",
        }
        if turnover is not None:
            rec["turnover"] = turnover
        return rec

    def test_mixed_rows_old_and_new(self, tmp_path) -> None:
        """Rows without the turnover key parse alongside rows with it."""
        records = [
            self._record("old_arm", 1),
            self._record("old_arm", 2),
            self._record("new_arm", 1, turnover=0.20),
            self._record("new_arm", 2, turnover=0.40),
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        md_path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )
        md = md_path.read_text()
        assert "Mean Turnover" in md
        # new_arm: mean of 0.20 and 0.40, on the arm's own table row.
        new_arm_row = next(
            line for line in md.splitlines() if "new_arm" in line
        )
        assert "0.3000" in new_arm_row
        # old_arm has no turnover data.
        assert "n/a" in md

    def test_gate_report_all_turnover_missing(self, tmp_path) -> None:
        records = [self._record("old_arm", 1), self._record("old_arm", 2)]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        md_path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )
        md = md_path.read_text()
        old_arm_row = next(
            line for line in md.splitlines() if "old_arm" in line
        )
        assert "n/a" in old_arm_row

    def test_gate_report_non_numeric_turnover_dropped(self, tmp_path) -> None:
        rec = self._record("new_arm", 1, turnover=0.25)
        bad = self._record("new_arm", 2)
        bad["turnover"] = "n/a"
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in (rec, bad)) + "\n"
        )

        md_path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )
        md = md_path.read_text()
        new_arm_row = next(
            line for line in md.splitlines() if "new_arm" in line
        )
        # The string row is dropped; the mean reflects only the numeric row.
        assert "0.2500" in new_arm_row

    def test_gate_report_nan_turnover_dropped(self, tmp_path) -> None:
        # json.loads accepts a bare NaN literal (Python extension over strict
        # JSON); a NaN turnover must not poison the mean or render as "nan".
        rec = self._record("nan_arm", 1, turnover=0.25)
        nan_line = (
            '{"model": "nan_arm", "window": 2, "ic": 0.05, "excess": 0.03,'
            ' "maxdd": -0.10, "completed_at": "2026-09-01T00:00:00+00:00",'
            ' "turnover": NaN}'
        )
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(json.dumps(rec) + "\n" + nan_line + "\n")

        md_path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )
        md = md_path.read_text()
        row = next(line for line in md.splitlines() if "nan_arm" in line)
        assert "0.2500" in row
        assert "nan" not in row.lower().replace("nan_arm", "")

    def test_gate_report_all_nan_turnover_shows_na(self, tmp_path) -> None:
        nan_line_1 = (
            '{"model": "allnan_arm", "window": 1, "ic": 0.05, "excess": 0.03,'
            ' "maxdd": -0.10, "completed_at": "2026-09-01T00:00:00+00:00",'
            ' "turnover": NaN}'
        )
        nan_line_2 = nan_line_1.replace('"window": 1', '"window": 2')
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(nan_line_1 + "\n" + nan_line_2 + "\n")

        md_path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )
        md = md_path.read_text()
        row = next(line for line in md.splitlines() if "allnan_arm" in line)
        assert "n/a" in row
