"""Blend verdict driver: run walk-forward with 60/40 blend and produce verdict.

Replaces out-of-band verdict assembly (Finding 4) with a committed,
reproducible producer. Wires blend_tra_ntra as the signal_transform
into run_full_walk_forward, builds the verdict on the REAL blend IC,
and writes results/verdict.json.

All qlib imports are deferred via the functions this module calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ashare_lab.config import load_config
from ashare_lab.research.blend import blend_tra_ntra
from ashare_lab.research.rolling import run_full_walk_forward
from ashare_lab.research.verdict import build_verdict, write_verdict

log = logging.getLogger(__name__)


def run_blend_verdict(
    exp_dir: Path,
    results_dir: Path | None = None,
) -> dict:
    """Run the full blend walk-forward and produce verdict.json.

    Executes run_full_walk_forward with blend_tra_ntra as the
    signal_transform, builds a verdict with track name derived from
    the blend ratio and strategy params, and writes to results_dir.

    The IC in the resulting verdict is the REAL blend IC (computed on
    the blended signal), not a proxy.

    Args:
        exp_dir: Experiment output directory for model weights and
            prediction persistence.
        results_dir: Directory for verdict.json output. Defaults to
            exp_dir / "results" if None.

    Returns:
        The verdict dict (same as build_verdict output with 12 fields).
    """
    cfg = load_config()

    universe: str = cfg["universe"]["primary"]
    n_drop: int = cfg["strategy"]["main"]["n_drop"]
    topk: int = cfg["strategy"]["topk"]
    gate_config: dict = cfg["gate"]

    # Fixed blend ratio from the locked default.
    tra_weight = 0.60

    log.info(
        "blend_verdict: universe=%s, topk=%d, n_drop=%d, tra_weight=%.2f",
        universe,
        topk,
        n_drop,
        tra_weight,
    )

    window_results, failed_windows = run_full_walk_forward(
        exp_dir=exp_dir,
        n_drop=n_drop,
        universe=universe,
        signal_transform=blend_tra_ntra,
    )

    # Track name built in code (never hand-set).
    ntra_weight = 1.0 - tra_weight
    track = (
        f"blend{int(tra_weight * 100)}_{int(ntra_weight * 100)}"
        f"_topk{topk}_ndrop{n_drop}"
    )

    verdict = build_verdict(
        window_results=window_results,
        track=track,
        universe=universe,
        gate_config=gate_config,
        failed_windows=failed_windows,
    )

    # Write verdict.json.
    if results_dir is None:
        results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_verdict(verdict, results_dir)

    log.info(
        "blend_verdict: gate=%s, track=%s, mean_ic=%s, n_windows=%d",
        verdict["gate"],
        verdict["track"],
        verdict.get("mean_rank_ic"),
        verdict["n_windows"],
    )

    return verdict
