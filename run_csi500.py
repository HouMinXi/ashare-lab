"""Standalone CSI500+TRA walk-forward run script for the GPU host.

Bypasses ashare_lab.config.load_config() (which hardcodes baseline.yaml with
LRU cache) by seeding the cache with configs/baseline_csi500.yaml before any
pipeline module is imported.

Usage (Windows GPU host):
    cd C:\\Users\\admin\\ashare-lab
    .venv\\Scripts\\python run_csi500.py

Output: experiments/csi500_tra/ with verdict.json, predictions/, models/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 1. Load CSI500 config and seed the LRU cache BEFORE importing pipeline
#    modules.  Every module in the pipeline (rolling, train, backtest,
#    smoke_test) calls ashare_lab.config.load_config() which is
#    @lru_cache(maxsize=1) pointing at the hardcoded baseline.yaml path.
#    By importing config first and populating its cache with our YAML,
#    all downstream load_config() calls return the CSI500 config.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CSI500_CONFIG = SCRIPT_DIR / "configs" / "baseline_csi500.yaml"

with open(CSI500_CONFIG, encoding="utf-8") as f:
    _csi500_cfg: dict = yaml.safe_load(f)

# Seed the LRU cache: import config module, then call the cached function
# after replacing its underlying logic.  The simplest correct approach is
# to clear the cache and make the function return our dict on first call.
import ashare_lab.config as _cfg_mod  # noqa: E402

_cfg_mod.load_config.cache_clear()

# Replace the cached function with one that returns CSI500 config.
_original_load = _cfg_mod.load_config.__wrapped__


def _load_csi500() -> dict:
    return _csi500_cfg


# Rebind: wrap with lru_cache so downstream code sees the same interface.
from functools import lru_cache  # noqa: E402

_cfg_mod.load_config = lru_cache(maxsize=1)(_load_csi500)

# ---------------------------------------------------------------------------
# 2. Now safe to import pipeline modules (they will get CSI500 config).
# ---------------------------------------------------------------------------

from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: E402
from ashare_lab.research.rolling import run_full_walk_forward  # noqa: E402
from ashare_lab.research.verdict import (  # noqa: E402
    build_verdict,
    write_ic_csv_png,
    write_verdict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = _csi500_cfg
    universe: str = cfg["universe"]["primary"]
    n_drop: int = cfg["strategy"]["main"]["n_drop"]
    topk: int = cfg["strategy"]["topk"]
    gate_config: dict = cfg["gate"]

    exp_dir = SCRIPT_DIR / "experiments" / "csi500_tra"
    exp_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "CSI500+TRA walk-forward: universe=%s topk=%d n_drop=%d exp_dir=%s",
        universe,
        topk,
        n_drop,
        exp_dir,
    )

    # Run the full walk-forward pipeline (train -> predict -> backtest).
    window_results = run_full_walk_forward(
        exp_dir=exp_dir,
        n_drop=n_drop,
        universe=universe,
    )

    # Build and write verdict.
    track = f"topk{topk}_ndrop{n_drop}"
    verdict = build_verdict(window_results, track, universe, gate_config)
    write_verdict(verdict, exp_dir)
    write_ic_csv_png(window_results, exp_dir, gate_config)

    log.info(
        "CSI500+TRA complete: gate=%s mean_ic=%s n_windows=%d",
        verdict["gate"],
        verdict["mean_rank_ic"],
        verdict["n_windows"],
    )


if __name__ == "__main__":
    main()
