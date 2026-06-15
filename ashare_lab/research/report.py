"""CLI entry point and top-level orchestration for the baseline research pipeline.

Provides make_exp_dir(), run_verdict_phase(), and main() -- the three
functions that wire together the walk-forward loop (plan 02-02) with the
gate verdict (plan 02-03) and optional supplementary analysis (plan 02-04).

Registered as the ashare-baseline console script in pyproject.toml.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from ashare_lab.config import CONFIG_PATH, PROJECT_ROOT, load_config
from ashare_lab.research.verdict import (
    build_verdict,
    write_ic_csv_png,
    write_verdict,
)

log = logging.getLogger(__name__)


def make_exp_dir() -> Path:
    """Create a timestamped experiment directory under PROJECT_ROOT/experiments/.

    Directory name: ``baseline_YYYYMMDD_HHMMSS``. The timestamp is
    lexicographically sortable and also agrees with mtime sort (``ls -dt``)
    since directories are created in ascending time order.

    The "baseline_" prefix is required by the plan 02-04 verification glob
    (``experiments/baseline_*/verdict.json``).

    Returns:
        Path to the created directory (parents created as needed).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / f"baseline_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log.info("experiment directory created: %s", exp_dir)
    return exp_dir


def run_verdict_phase(
    window_results: list[dict],
    exp_dir: Path,
    n_drop: int,
) -> dict:
    """Aggregate walk-forward results into a gate verdict and write outputs.

    Loads config, constructs the track name, calls build_verdict and the
    two file writers, copies the config YAML into the experiment directory,
    logs the gate result, and either exits (FAIL) or triggers supplementary
    analysis (PASS / BORDER_PASS).

    Args:
        window_results: List of WindowResult dicts from run_full_walk_forward.
        exp_dir: Experiment directory created by make_exp_dir (must exist).
        n_drop: n_drop parameter used for the current track (from config
            strategy.main.n_drop in the default pipeline).

    Returns:
        The verdict dict (all 11 D-25 fields). Only reached on PASS or
        BORDER_PASS -- FAIL calls sys.exit(1) before returning.
    """
    cfg = load_config()
    gate_config: dict = cfg["gate"]

    universe: str = cfg["universe"]["primary"]
    topk: int = cfg["strategy"]["topk"]
    track: str = f"topk{topk}_ndrop{n_drop}"

    # Build the 11-field verdict dict.
    verdict_dict = build_verdict(
        window_results=window_results,
        track=track,
        universe=universe,
        gate_config=gate_config,
    )

    # Write verdict.json atomically.
    write_verdict(verdict_dict, exp_dir)

    # Write ic_windows.csv and ic_windows.png.
    write_ic_csv_png(window_results, exp_dir, gate_config=gate_config)

    # Preserve exact YAML formatting and comments alongside results.
    shutil.copy(CONFIG_PATH, exp_dir / "baseline.yaml")
    log.info("config copied -> %s", exp_dir / "baseline.yaml")

    # Log gate result.
    gate: str = verdict_dict["gate"]
    raw_ic = verdict_dict["mean_rank_ic"]
    ic_str = "None" if (raw_ic is None or pd.isna(raw_ic)) else f"{raw_ic:.4f}"
    log.info(
        "gate=%s mean_rank_ic=%s positive_excess_pct=%.1f%%",
        gate,
        ic_str,
        verdict_dict["positive_excess_pct"] * 100,
    )

    if gate == "FAIL":
        log.error("gate FAIL: exiting with code 1")
        sys.exit(1)

    # PASS or BORDER_PASS: trigger supplementary analysis from plan 02-04.
    # Catch ALL exceptions (ImportError when module not yet created, and any
    # runtime error from supplementary steps) to avoid a bare stack trace.
    try:
        from ashare_lab.research.supplementary import run_all_supplementary  # noqa: PLC0415

        run_all_supplementary(window_results, exp_dir)
    except Exception as e:
        log.warning("supplementary failed: %s", e)

    return verdict_dict


def main() -> None:
    """CLI entry point: orchestrate the full baseline research pipeline.

    Invoked as ``ashare-baseline`` via the console_scripts entry point.

    Pipeline:
      1. Load config (sys.exit(2) on any error).
      2. Resolve n_drop, universe, create experiment directory.
      3. Run walk-forward loop (train/backtest/metrics per window).
      4. Run verdict phase (gate, output files, optional supplementary).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        cfg = load_config()
    except Exception as exc:
        log.error("failed to load config: %s", exc)
        sys.exit(2)

    n_drop: int = cfg["strategy"]["main"]["n_drop"]
    universe: str = cfg["universe"]["primary"]
    exp_dir: Path = make_exp_dir()

    # Deferred import: run_full_walk_forward requires qlib runtime.
    from ashare_lab.research.rolling import run_full_walk_forward  # noqa: PLC0415

    window_results: list[dict] = run_full_walk_forward(exp_dir, n_drop, universe)

    run_verdict_phase(window_results, exp_dir, n_drop)
