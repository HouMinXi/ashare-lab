"""Generic track verdict driver for shadow tracks (regime, raw R14).

Runs walk-forward with an arbitrary signal_transform and produces
verdict_{track_key}.json.  Complements blend_verdict.py (which stays
unchanged as the forge-passed primary-track producer) by giving
regime and r14 shadow tracks their own committed producers.

Blend uses blend_verdict.py (dedicated, forge-reviewed).
Regime and R14 use this module (generic, parameterised by signal_fn).

All qlib imports are deferred via the functions this module calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ashare_lab.config import load_config
from ashare_lab.research.rolling import run_full_walk_forward
from ashare_lab.research.verdict import build_verdict, write_verdict

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def run_track_verdict(
    track_key: str,
    exp_dir: Path,
    signal_fn: Callable[[pd.Series, dict], pd.Series] | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run walk-forward with *signal_fn* and produce verdict_{track_key}.json.

    Args:
        track_key: Short identifier used in the output filename
            (``verdict_{track_key}.json``) and in the track string
            stored inside the verdict.  Examples: ``"regime"``,
            ``"r14"``.
        exp_dir: Experiment output directory for model weights and
            prediction persistence.
        signal_fn: Optional signal transform ``(pred, window) -> pred``.
            ``None`` means identity (raw predictions, i.e. the R14
            baseline track).
        results_dir: Directory for verdict output.  Defaults to
            ``exp_dir / "results"`` when ``None``.

    Returns:
        The verdict dict (same 12-field schema as build_verdict).
    """
    cfg = load_config()

    universe: str = cfg["universe"]["primary"]
    n_drop: int = cfg["strategy"]["main"]["n_drop"]
    topk: int = cfg["strategy"]["topk"]
    gate_config: dict = cfg["gate"]

    log.info(
        "track_verdict[%s]: universe=%s, topk=%d, n_drop=%d, "
        "signal_fn=%s",
        track_key,
        universe,
        topk,
        n_drop,
        getattr(signal_fn, "__name__", repr(signal_fn)) if signal_fn is not None else "None(identity)",
    )

    window_results, failed_windows = run_full_walk_forward(
        exp_dir=exp_dir,
        n_drop=n_drop,
        universe=universe,
        signal_transform=signal_fn,
    )

    track = f"{track_key}_topk{topk}_ndrop{n_drop}"

    verdict = build_verdict(
        window_results=window_results,
        track=track,
        universe=universe,
        gate_config=gate_config,
        failed_windows=failed_windows,
    )

    if results_dir is None:
        results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_verdict(verdict, results_dir, name=f"verdict_{track_key}")

    log.info(
        "track_verdict[%s]: gate=%s, track=%s, mean_ic=%s, n_windows=%d",
        track_key,
        verdict["gate"],
        verdict["track"],
        verdict.get("mean_rank_ic"),
        verdict["n_windows"],
    )

    return verdict
