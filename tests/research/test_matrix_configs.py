"""Tests for matrix candidate config YAMLs.

Verifies all configs/matrix_*.yaml files load, have required keys,
and use correct hyperparameters per qlib benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
VALID_MODEL_TYPES = {"lgbm", "alstm", "tra", "densemble"}


def _load_all_matrix_configs():
    """Load all matrix config YAMLs and return as list of (path, dict)."""
    files = sorted(CONFIGS_DIR.glob("matrix_*.yaml"))
    assert len(files) >= 4, f"Expected >= 4 matrix configs, found {len(files)}"
    result = []
    for f in files:
        with open(f) as fh:
            cfg = yaml.safe_load(fh)
        result.append((f, cfg))
    return result


class TestMatrixConfigs:
    def test_all_configs_load_and_have_required_keys(self):
        """Every matrix config has model.tag, model.type, and matrix.tag."""
        for path, cfg in _load_all_matrix_configs():
            model = cfg.get("model", {})
            assert "tag" in model, f"{path.name}: missing model.tag"
            assert "type" in model, f"{path.name}: missing model.type"
            assert model["type"] in VALID_MODEL_TYPES, (
                f"{path.name}: model.type={model['type']!r} not in {VALID_MODEL_TYPES}"
            )
            matrix = cfg.get("matrix", {})
            assert "tag" in matrix, f"{path.name}: missing matrix.tag"
            assert model["tag"] == matrix["tag"], (
                f"{path.name}: model.tag={model['tag']!r} != matrix.tag={matrix['tag']!r}"
            )

    def test_tag_matches_filename(self):
        """model.tag matches the filename pattern matrix_{tag}.yaml."""
        for path, cfg in _load_all_matrix_configs():
            expected_tag = path.stem.replace("matrix_", "")
            actual_tag = cfg["model"]["tag"]
            assert actual_tag == expected_tag, (
                f"{path.name}: tag={actual_tag!r} != filename-derived={expected_tag!r}"
            )

    def test_alpha360_configs_input_size(self):
        """Alpha360 configs must have input_size=6 or d_feat=6, not 360."""
        for path, cfg in _load_all_matrix_configs():
            model = cfg["model"]
            if model.get("handler") != "alpha360":
                continue
            if model["type"] == "tra":
                backbone = model.get("backbone", {})
                assert backbone.get("input_size") == 6, (
                    f"{path.name}: TRA Alpha360 backbone.input_size must be 6"
                )
            elif model["type"] == "alstm":
                assert model.get("d_feat") == 6, (
                    f"{path.name}: ALSTM Alpha360 d_feat must be 6"
                )

    def test_densemble_benchmark_params(self):
        """Candidate B uses benchmark params, not defaults."""
        for path, cfg in _load_all_matrix_configs():
            if cfg["model"].get("type") != "densemble":
                continue
            model = cfg["model"]
            assert model["learning_rate"] == 0.2, "densemble lr should be 0.2 (benchmark)"
            assert model["epochs"] == 28, "densemble epochs should be 28 (benchmark)"
            assert model["num_models"] == 6
