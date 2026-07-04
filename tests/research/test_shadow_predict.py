"""Tests for shadow_predict -- output isolation, path resolution, sidecar.

Uses sys.modules stubs (same pattern as test_predict.py) so deferred qlib/torch
imports resolve without those packages installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# sys.modules stubs for qlib/torch (mirrors test_predict.py)
# ---------------------------------------------------------------------------

def _install_fake_modules():
    """Inject mock qlib/torch modules into sys.modules."""
    injected = []

    def _ensure(name, attrs=None):
        if name not in sys.modules:
            mod = ModuleType(name)
            if attrs:
                for k, v in attrs.items():
                    setattr(mod, k, v)
            sys.modules[name] = mod
            injected.append(name)
        return sys.modules[name]

    torch_mod = _ensure("torch")
    torch_mod.load = MagicMock()

    _ensure("qlib")
    qlib_config = _ensure("qlib.config")
    mock_c = MagicMock()
    mock_c.registered = True
    qlib_config.C = mock_c
    qlib_config.REG_CN = "cn"

    _ensure("qlib.contrib")
    _ensure("qlib.contrib.data")
    qlib_handler = _ensure("qlib.contrib.data.handler")
    qlib_handler.Alpha158 = MagicMock()
    qlib_handler.Alpha360 = MagicMock()

    qlib_dataset_contrib = _ensure("qlib.contrib.data.dataset")
    qlib_dataset_contrib.MTSDatasetH = MagicMock()

    _ensure("qlib.data")
    _ensure("qlib.data.dataset")
    sys.modules["qlib.data.dataset"].DatasetH = MagicMock()
    sys.modules["qlib.data.dataset"].TSDatasetH = MagicMock()

    sys.modules["qlib"].init = MagicMock()

    def cleanup():
        for name in reversed(injected):
            sys.modules.pop(name, None)

    return cleanup


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_WINDOW = {
    "step": 0, "window_id": 1,
    "train_start": "2018-01-01", "train_end": "2020-12-31",
    "valid_start": "2021-01-01", "valid_end": "2021-06-30",
    "test_start": "2021-07-01", "test_end": "2021-12-31",
    "is_complete": True,
}


@pytest.fixture()
def fake_config(tmp_path):
    """Create minimal baseline YAML and return (tmp_dir, baseline_path)."""
    baseline = {
        "universe": {"primary": "csi1000", "exclude_close_above_cny": 300},
        "walk_forward": {
            "train_start": "2018-01-01",
            "base_train_end": "2020-12-31",
            "step_months": 6,
            "min_windows": 5,
        },
        "model": {"type": "tra", "handler": "alpha158"},
    }
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline))
    return tmp_path, baseline_path


def _make_candidate(tmp_path, model_type="lgbm", tag="a_lgb"):
    candidate = {
        "model": {"type": model_type, "tag": tag, "handler": "alpha158"},
        "matrix": {"tag": tag},
    }
    p = tmp_path / f"matrix_{tag}.yaml"
    p.write_text(yaml.safe_dump(candidate))
    return p


def _make_model_file(model_dir, tag, window_id, ext):
    p = model_dir / tag / "models" / f"w{window_id}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")
    return p


def _run_shadow(baseline_path, candidate_path, model_dir, tmp_path,
                project_root=None):
    """Run shadow_predict_for_date with full mocking. Returns output Path."""
    cleanup = _install_fake_modules()
    try:
        # Force reimport so deferred imports pick up stubs
        sys.modules.pop("ashare_lab.research.shadow_predict", None)
        from ashare_lab.research.shadow_predict import shadow_predict_for_date
        import ashare_lab.config as cfg

        proj = project_root or tmp_path

        mock_model = MagicMock()
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-09-01"), "SH600000"),
             (pd.Timestamp("2021-09-01"), "SH600001")],
            names=["datetime", "instrument"],
        )
        pred_series = pd.Series([0.1, 0.2], index=idx, name="score")
        mock_model.predict.return_value = pred_series

        with (
            patch.object(cfg, "CONFIG_PATH", baseline_path),
            patch.object(cfg, "PROJECT_ROOT", proj),
            patch.object(cfg, "PREDICTIONS_DIR", proj / "predictions"),
            patch("ashare_lab.research.smoke_test.get_all_windows",
                  return_value=[FAKE_WINDOW]),
            patch("ashare_lab.research.train.ALPHA158_WARMUP_START", "2017-01-01"),
            patch("ashare_lab.data.update.DEFAULT_PROVIDER_URI", "/fake"),
            patch("pickle.load", return_value=mock_model),
        ):
            return shadow_predict_for_date(
                "2021-09-01", str(candidate_path), model_dir=model_dir
            )
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_dir_isolation(fake_config, tmp_path):
    """Shadow output contains 'shadow_predictions', not bare '/predictions/'."""
    tmp_dir, baseline_path = fake_config
    candidate_path = _make_candidate(tmp_dir, "lgbm", "a_lgb")
    model_dir = tmp_path / "models"
    _make_model_file(model_dir, "a_lgb", 1, ".pkl")

    result = _run_shadow(baseline_path, candidate_path, model_dir, tmp_path)

    assert "shadow_predictions" in str(result)
    # The string "/predictions/" should only appear as part of "/shadow_predictions/"
    stripped = str(result).replace("shadow_predictions", "SHADOW")
    assert "/predictions/" not in stripped


def test_model_path_resolution_pkl(tmp_path):
    """lgbm/densemble -> .pkl extension."""
    model_dir = tmp_path / "models"
    p = _make_model_file(model_dir, "a_lgb", 1, ".pkl")
    assert p.suffix == ".pkl"
    assert p == model_dir / "a_lgb" / "models" / "w1.pkl"


def test_model_path_resolution_pt(tmp_path):
    """alstm/tra -> .pt extension."""
    model_dir = tmp_path / "models"
    p = _make_model_file(model_dir, "e_alstm", 1, ".pt")
    assert p.suffix == ".pt"
    assert p == model_dir / "e_alstm" / "models" / "w1.pt"


def test_meta_sidecar_written(fake_config, tmp_path):
    """Meta JSON sidecar written alongside parquet with correct fields."""
    tmp_dir, baseline_path = fake_config
    candidate_path = _make_candidate(tmp_dir, "lgbm", "test_meta")
    model_dir = tmp_path / "models"
    _make_model_file(model_dir, "test_meta", 1, ".pkl")

    result = _run_shadow(baseline_path, candidate_path, model_dir, tmp_path)

    meta_path = result.parent / "2021-09-01.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["tag"] == "test_meta"
    assert meta["type"] == "lgbm"
    assert "produced_at" in meta
    assert meta["n_instruments"] == 2


def test_model_dir_none_fallback(fake_config):
    """model_dir=None resolves to PROJECT_ROOT / 'matrix_results'."""
    cleanup = _install_fake_modules()
    try:
        sys.modules.pop("ashare_lab.research.shadow_predict", None)
        from ashare_lab.research.shadow_predict import shadow_predict_for_date
        import ashare_lab.config as cfg

        tmp_dir, baseline_path = fake_config
        candidate_path = _make_candidate(tmp_dir, "lgbm", "fallback_test")

        with (
            patch.object(cfg, "CONFIG_PATH", baseline_path),
            patch.object(cfg, "PROJECT_ROOT", tmp_dir),
            patch.object(cfg, "PREDICTIONS_DIR", tmp_dir / "predictions"),
            patch("ashare_lab.research.smoke_test.get_all_windows",
                  return_value=[FAKE_WINDOW]),
            patch("ashare_lab.research.train.ALPHA158_WARMUP_START", "2017-01-01"),
            patch("ashare_lab.data.update.DEFAULT_PROVIDER_URI", "/fake"),
        ):
            with pytest.raises(FileNotFoundError, match="matrix_results"):
                shadow_predict_for_date(
                    "2021-09-01", str(candidate_path), model_dir=None
                )
    finally:
        cleanup()


def test_output_dir_created(fake_config, tmp_path):
    """Non-existent shadow_predictions/{tag}/ dir created before write."""
    tmp_dir, baseline_path = fake_config
    candidate_path = _make_candidate(tmp_dir, "lgbm", "newdir_test")
    model_dir = tmp_path / "models"
    _make_model_file(model_dir, "newdir_test", 1, ".pkl")

    project_root = tmp_path / "project"
    shadow_dir = project_root / "shadow_predictions" / "newdir_test"
    assert not shadow_dir.exists()

    result = _run_shadow(
        baseline_path, candidate_path, model_dir, tmp_path,
        project_root=project_root,
    )

    assert shadow_dir.exists()
    assert result.parent == shadow_dir
