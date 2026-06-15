"""Canonical config loader for ashare-lab Phase 2 research modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# PROJECT_ROOT is the repo root, anchored to this file's location.
# ashare_lab/config.py lives one level inside the repo root.
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "baseline.yaml"

MODELS_DIR: Path = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read and return baseline.yaml as a dict, cached after first call."""
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)
