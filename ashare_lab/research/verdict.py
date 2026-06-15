"""Gate verdict computation and output writing for the walk-forward pipeline.

Provides compute_gate(), build_verdict(), write_verdict(), and
write_ic_csv_png() -- the four functions that aggregate per-window results
into a single verdict.json, CSV, and PNG for plan 02-03.

All qlib imports are deferred so this module is importable without a
qlib runtime (unit-test isolation matches the pattern in metrics.py).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd

log = logging.getLogger(__name__)

# Use a non-interactive backend so plt works without a display.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------


def compute_gate(
    mean_ic: float | None,
    pos_pct: float,
    gate_config: dict,
) -> str:
    """Classify a mean IC and positive-excess percentage into a gate outcome.

    Truth table (thresholds read exclusively from gate_config; no internal
    defaults):

    1. mean_ic is None or NaN                                    -> FAIL
    2. mean_ic <= gate_config["min_mean_rank_ic"]                -> FAIL
       OR pos_pct < gate_config["min_positive_excess_pct"]      -> FAIL
    3. non-FAIL AND (mean_ic < gate_config["border_pass_ic_upper"]
                    OR pos_pct < gate_config["border_pass_excess_upper"]) -> BORDER_PASS
    4. mean_ic >= border_pass_ic_upper AND
       pos_pct >= border_pass_excess_upper                       -> PASS

    Args:
        mean_ic: Mean rank IC across windows. None or NaN treated as missing.
        pos_pct: Fraction of windows with positive after-cost excess return.
            Denominator is the total number of completed windows.
        gate_config: Dict with keys min_mean_rank_ic, min_positive_excess_pct,
            border_pass_ic_upper, border_pass_excess_upper (all float).

    Returns:
        "PASS", "FAIL", or "BORDER_PASS".
    """
    min_ic: float = gate_config["min_mean_rank_ic"]
    min_pct: float = gate_config["min_positive_excess_pct"]
    border_ic: float = gate_config["border_pass_ic_upper"]
    border_pct: float = gate_config["border_pass_excess_upper"]

    # Rule 1: missing IC -> FAIL.
    if mean_ic is None or pd.isna(mean_ic):
        return "FAIL"

    # Rule 2: boundary FAIL (IC must be strictly greater than min_ic; = FAILs).
    if mean_ic <= min_ic or pos_pct < min_pct:
        return "FAIL"

    # Rule 3: non-FAIL but below upper border -> BORDER_PASS.
    if mean_ic < border_ic or pos_pct < border_pct:
        return "BORDER_PASS"

    # Rule 4: all thresholds met -> PASS.
    return "PASS"


# ---------------------------------------------------------------------------
# Verdict construction
# ---------------------------------------------------------------------------


def build_verdict(
    window_results: list[dict],
    track: str,
    universe: str,
    gate_config: dict,
) -> dict:
    """Aggregate window results into the 11-field verdict dict (D-25 schema).

    Args:
        window_results: List of WindowResult dicts from run_full_walk_forward.
            Each dict must contain at minimum: window_id, mean_rank_ic,
            cumulative_excess_return, is_positive_excess, lot_skip_count.
            Errored windows must already be excluded by rolling.py -- no None
            entries expected.
        track: Caller-built track string, e.g. "topk15_ndrop1". Stored as-is.
        universe: Qlib universe name, e.g. "csi500". Taken from caller param
            (not extracted from window_results) so empty-list is safe.
        gate_config: Dict with gate threshold keys (see compute_gate).

    Returns:
        Dict with 11 fields:
            gate, universe, track, mean_rank_ic, n_windows,
            n_positive_excess_windows, positive_excess_pct, window_details,
            rejected_orders_lot_skip, slippage_sensitivity, note.
        slippage_sensitivity is None (backfilled by plan 02-04 supplementary).
    """
    n_windows: int = len(window_results)

    # pos_pct denominator = n_windows (all completed windows, including those
    # with None IC; None IC only affects the mean, not the positive flag).
    if n_windows == 0:
        pos_pct = 0.0
        mean_ic = None
    else:
        n_pos = sum(
            1 for w in window_results if w.get("is_positive_excess", False)
        )
        pos_pct = n_pos / n_windows

        # Compute mean IC over valid (non-None, non-NaN) IC values only.
        valid_ics = [
            w["mean_rank_ic"]
            for w in window_results
            if pd.notna(w.get("mean_rank_ic"))
        ]
        mean_ic = float(sum(valid_ics) / len(valid_ics)) if valid_ics else None

    gate = compute_gate(mean_ic, pos_pct, gate_config)

    # Reuse n_pos computed above (avoid identical second sum).
    n_positive = n_pos if n_windows > 0 else 0

    # rejected_orders_lot_skip: sum of non-None lot_skip_count values.
    # None if ALL windows have lot_skip_count=None.
    lot_skips = [
        w["lot_skip_count"]
        for w in window_results
        if w.get("lot_skip_count") is not None
    ]
    rejected_orders_lot_skip = sum(lot_skips) if lot_skips else None

    # window_details: per-window summary; None-safe mean_rank_ic.
    window_details = [
        {
            "window_id": w["window_id"],
            "mean_rank_ic": w.get("mean_rank_ic"),  # may be None
            "cumulative_excess_return": w.get("cumulative_excess_return"),
            "is_positive_excess": w.get("is_positive_excess"),
        }
        for w in window_results
    ]

    # Human-readable note.
    ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "None"
    note = (
        f"Gate {gate}: mean_rank_ic={ic_str}, "
        f"positive_excess_pct={pos_pct:.0%}"
    )

    return {
        "gate": gate,
        "universe": universe,
        "track": track,
        "mean_rank_ic": mean_ic,
        "n_windows": n_windows,
        "n_positive_excess_windows": n_positive,
        "positive_excess_pct": pos_pct,
        "window_details": window_details,
        "rejected_orders_lot_skip": rejected_orders_lot_skip,
        "slippage_sensitivity": None,  # backfilled by 02-04
        "note": note,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalar types to Python native types.

    Step 1 of the serialization pipeline (must run BEFORE NaN detection so
    that numpy.nan becomes Python float nan, which pd.isna() can then catch).

    numpy>=2 requires explicit conversion for both integer and float types;
    silent coercion via standard json.dumps is no longer reliable.
    """
    if isinstance(obj, numpy.integer):
        return int(obj)
    if isinstance(obj, numpy.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    return obj


def _nan_to_none(obj: Any) -> Any:
    """Recursively replace NaN floats with None (JSON null).

    Step 2: runs after numpy conversion so numpy NaN is already Python float.
    Uses pd.isna() to catch float('nan') values only (not None, not strings).
    Traverses nested dicts and lists.
    """
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(item) for item in obj]
    return obj


def _serialize_verdict(verdict_dict: dict) -> dict:
    """Apply the three serialization transforms required for JSON output.

    Order is load-bearing (numpy>=2):
      1. Convert numpy int/float -> Python int/float (captures numpy NaN too).
      2. Convert all Python float NaN -> None (JSON null).
      3. json.dump will use default=str for any remaining non-serializable types.

    Returns a new dict ready for json.dump.
    """
    step1 = _convert_numpy(verdict_dict)
    step2 = _nan_to_none(step1)
    return step2


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_verdict(verdict_dict: dict, exp_dir: Path) -> Path:
    """Serialize and atomically write verdict_dict to exp_dir/verdict.json.

    Applies all three serialization transforms via _serialize_verdict, then
    writes atomically using a temporary file + os.replace to avoid partial
    writes on crash.

    Args:
        verdict_dict: The 11-field dict returned by build_verdict.
        exp_dir: Experiment output directory (must exist).

    Returns:
        Path to the written verdict.json file.
    """
    serializable = _serialize_verdict(verdict_dict)
    out_path = exp_dir / "verdict.json"

    # Atomic write: write to tmp file in same directory, then replace.
    fd, tmp_path = tempfile.mkstemp(dir=exp_dir, prefix=".verdict_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
    except Exception:
        # Clean up tmp file if replace failed.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    log.info(
        "verdict written: gate=%s path=%s",
        verdict_dict.get("gate"),
        out_path,
    )
    return out_path


def write_ic_csv_png(
    window_results: list[dict],
    exp_dir: Path,
    gate_config: dict,
) -> tuple[Path, Path]:
    """Write per-window IC data to CSV and a bar chart PNG.

    CSV columns: window_id, mean_rank_ic, cumulative_excess_return,
    is_positive_excess (subset of WindowResult -- not all 11 fields).

    PNG: bar chart of mean_rank_ic per window, with a threshold line at
    gate_config["min_mean_rank_ic"]. Label derived from threshold value
    (never hardcoded). If window_results is empty or all IC values are None,
    renders only the threshold line with a "no data" annotation (no bars).

    Args:
        window_results: List of WindowResult dicts.
        exp_dir: Experiment output directory (must exist).
        gate_config: Dict with key min_mean_rank_ic (float).

    Returns:
        (csv_path, png_path) tuple.
    """
    csv_path = exp_dir / "ic_windows.csv"
    png_path = exp_dir / "ic_windows.png"

    thr: float = gate_config["min_mean_rank_ic"]

    # Build DataFrame.
    if window_results:
        df = pd.DataFrame(
            [
                {
                    "window_id": w["window_id"],
                    "mean_rank_ic": w.get("mean_rank_ic"),
                    "cumulative_excess_return": w.get("cumulative_excess_return"),
                    "is_positive_excess": w.get("is_positive_excess"),
                }
                for w in window_results
            ]
        )
    else:
        df = pd.DataFrame(
            columns=[
                "window_id",
                "mean_rank_ic",
                "cumulative_excess_return",
                "is_positive_excess",
            ]
        )

    # Write CSV (no unnamed index column).
    df.to_csv(csv_path, index=False)
    log.info("ic_windows.csv written -> %s", csv_path)

    # Determine if any bars to draw.
    has_data = (
        not df.empty
        and "mean_rank_ic" in df.columns
        and df["mean_rank_ic"].notna().any()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        if has_data:
            plot_df = df.dropna(subset=["mean_rank_ic"])
            ax.bar(
                plot_df["window_id"].astype(str),
                plot_df["mean_rank_ic"],
                color="steelblue",
                label="Mean RankIC",
            )
        else:
            ax.annotate(
                "no data",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )

        # Threshold line (label derived from threshold value, never hardcoded).
        ax.axhline(
            y=thr,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"FAIL if <= {thr}",
        )

        ax.set_xlabel("Window ID")
        ax.set_ylabel("Mean Rank IC")
        ax.set_title("Walk-Forward IC by Window")
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=100)
    finally:
        plt.close(fig)

    log.info("ic_windows.png written -> %s", png_path)
    return csv_path, png_path
