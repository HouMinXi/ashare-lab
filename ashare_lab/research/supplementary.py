"""Conditional supplementary analysis triggered after gate PASS/BORDER_PASS.

Provides run_all_supplementary() and five sub-functions:
  - finalize_artifacts()         -- copy highest-numbered main pkl to models/latest.pkl
  - write_feature_importance()   -- per-window gain importance CSV + PNG
  - run_slippage_sensitivity()   -- 3-row CSV + verdict.json backfill
  - run_control_track()          -- CSI500 n_drop=2 control verdict
  - run_csi300_reference()       -- CSI300 reference walk-forward (isolated)

All qlib imports are deferred inside function bodies so this module is
importable without a qlib runtime (unit-test isolation matches other modules).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ashare_lab.config import MODELS_DIR, load_config
from ashare_lab.research.rolling import run_full_walk_forward
from ashare_lab.research.verdict import _serialize_verdict, build_verdict

log = logging.getLogger(__name__)

# Use non-interactive backend for headless operation.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Re-backtest helper
# ---------------------------------------------------------------------------


def _compute_cumulative_excess(
    portfolio_df: pd.DataFrame,
    bench_close: pd.Series,
) -> tuple[float, bool]:
    """Compute cumulative excess return vs benchmark.

    Shared helper called by both the re-backtest paths (slippage sensitivity,
    control track, CSI300 reference) to avoid formula drift vs
    aggregate_window_metrics.

    Args:
        portfolio_df: DataFrame with DatetimeIndex and "return" column.
            May be empty (0 rows = cash-only).
        bench_close: Series with DatetimeIndex of benchmark close prices.
            14-day pre-fetch produces extra leading rows that are trimmed here.

    Returns:
        (cumulative_excess_return, is_positive_excess) tuple.
        cumulative_excess_return = 0.0 and is_positive_excess = False when
        portfolio_df is empty.
    """
    if portfolio_df.empty:
        return 0.0, False

    bench_return = bench_close.pct_change().dropna()
    # Trim to portfolio period; 14-day pre-fetch adds extra leading rows.
    bench_return = bench_return.loc[
        portfolio_df.index[0] : portfolio_df.index[-1]
    ]
    daily_excess = (
        portfolio_df["return"].reindex(bench_return.index).fillna(0) - bench_return
    )
    cumulative_excess_return = float((1 + daily_excess).prod() - 1)
    is_positive_excess = cumulative_excess_return > 0
    return cumulative_excess_return, is_positive_excess


def _rebacktest_window(
    window_result: dict,
    n_drop: int,
    slippage_override: float | None,
) -> dict | None:
    """Re-backtest one window from its persisted predictions.

    Does NOT retrain. Loads predictions from pred_path via PredictionFile
    schema (pd.read_parquet -> "score" column). Calls run_backtest with the
    given n_drop and slippage_override. Builds a new WindowResult by copying
    metadata from the original and filling derived fields.

    Returns None (and logs a warning) if pred_path is missing or unreadable.
    mean_rank_ic is reused from the original (IC is independent of slippage/n_drop).
    """
    from ashare_lab.research.backtest import run_backtest  # noqa: PLC0415

    pred_path = Path(window_result["pred_path"])
    if not pred_path.exists():
        log.warning(
            "W%d: pred_path missing %s; skipping re-backtest",
            window_result["window_id"],
            pred_path,
        )
        return None

    try:
        pred = pd.read_parquet(pred_path)["score"]
    except Exception as exc:
        log.warning(
            "W%d: failed to load pred_path %s: %s; skipping",
            window_result["window_id"],
            pred_path,
            exc,
        )
        return None

    portfolio_df, bench_close, lot_skip_count = run_backtest(
        window=window_result,
        pred=pred,
        n_drop=n_drop,
        slippage_override=slippage_override,
    )

    cumulative_excess_return, is_positive_excess = _compute_cumulative_excess(
        portfolio_df, bench_close
    )

    # n_drop comes from the param passed (not from run_backtest return).
    return {
        "window_id": window_result["window_id"],
        "universe": window_result["universe"],
        "train_end": window_result["train_end"],
        "test_start": window_result["test_start"],
        "test_end": window_result["test_end"],
        "pred_path": window_result["pred_path"],
        "n_drop": n_drop,
        "mean_rank_ic": window_result["mean_rank_ic"],  # reuse; IC is slippage-independent
        "cumulative_excess_return": cumulative_excess_return,
        "is_positive_excess": is_positive_excess,
        "lot_skip_count": lot_skip_count,
    }


# ---------------------------------------------------------------------------
# Atomic JSON write helper
# ---------------------------------------------------------------------------


def _write_verdict_atomic(verdict_dict: dict, out_path: Path) -> None:
    """Serialize verdict_dict and atomically write to out_path.

    Applies _serialize_verdict transforms (numpy -> Python, NaN -> None),
    then writes via tmp file + os.replace to guard against partial writes.
    """
    serializable = _serialize_verdict(verdict_dict)
    fd, tmp_path = tempfile.mkstemp(
        dir=out_path.parent, prefix=".verdict_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Sub-function 1: finalize_artifacts
# ---------------------------------------------------------------------------


def finalize_artifacts(models_dir: Path) -> Path:
    """Copy the highest-numbered main walk-forward pkl to MODELS_DIR/latest.pkl.

    Globs for w*.pkl in models_dir, filters with re.fullmatch(r"w\\d+\\.pkl"),
    sorts by the integer after "w", and copies the highest to MODELS_DIR/latest.pkl
    via shutil.copy2 (preserves mtime; does NOT delete source -- write_feature_importance
    reads these pkls in step 2).

    Args:
        models_dir: Directory containing per-window model files (w1.pkl, w2.pkl, ...).

    Returns:
        Path to MODELS_DIR/latest.pkl.

    Raises:
        FileNotFoundError: if no matching w*.pkl files found in models_dir.
    """
    candidates = [
        p
        for p in models_dir.glob("w*.pkl")
        if re.fullmatch(r"w\d+\.pkl", p.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"finalize_artifacts: no w*.pkl files found in {models_dir}"
        )

    # Sort by the integer suffix (not lexicographic) so w10 > w9.
    candidates.sort(key=lambda p: int(re.search(r"\d+", p.name).group()))
    latest_src = candidates[-1]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "latest.pkl"
    shutil.copy2(latest_src, dest)
    log.info(
        "finalize_artifacts: copied %s -> %s", latest_src.name, dest
    )
    return dest


# ---------------------------------------------------------------------------
# Sub-function 2: write_feature_importance
# ---------------------------------------------------------------------------


def write_feature_importance(models_dir: Path, exp_dir: Path) -> Path:
    """Write per-window feature importance CSV and PNG to exp_dir/feature_importance/.

    Loads each w*.pkl via LGBModel.load() (classmethod). Extracts feature
    importance via model.model.feature_importance(importance_type="gain").
    Skips missing pkls with a warning; partial output is acceptable.

    This function is called only for main-track models. CSI300 feature
    importance is intentionally omitted (CSI300 is a reference track, not a
    production inference model).

    Args:
        models_dir: Directory containing per-window pkl files.
        exp_dir: Experiment directory. Output written to exp_dir/feature_importance/.

    Returns:
        Path to exp_dir/feature_importance/ directory (may be empty on partial failure).
    """
    from qlib.contrib.model.gbdt import LGBModel  # noqa: PLC0415

    fi_dir = exp_dir / "feature_importance"
    fi_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        p
        for p in models_dir.glob("w*.pkl")
        if re.fullmatch(r"w\d+\.pkl", p.name)
    ]

    if not candidates:
        log.warning("write_feature_importance: no models found in %s", models_dir)
        return fi_dir

    candidates.sort(key=lambda p: int(re.search(r"\d+", p.name).group()))
    total = len(candidates)
    loaded = 0

    for pkl_path in candidates:
        try:
            model = LGBModel.load(str(pkl_path))
            feature_names = model.model.feature_name()
            importances = model.model.feature_importance(importance_type="gain")

            window_label = pkl_path.stem  # e.g. "w1"

            # Write CSV.
            df = pd.DataFrame({
                "feature": feature_names,
                "gain": importances,
            }).sort_values("gain", ascending=False)
            csv_path = fi_dir / f"{window_label}_importance.csv"
            df.to_csv(csv_path, index=False)

            # Write PNG.
            png_path = fi_dir / f"{window_label}_importance.png"
            fig, ax = plt.subplots(figsize=(10, max(4, len(df) // 5)))
            try:
                top_df = df.head(30)  # show top-30 features for readability
                ax.barh(top_df["feature"][::-1], top_df["gain"][::-1], color="steelblue")
                ax.set_xlabel("Gain")
                ax.set_title(f"Feature Importance ({window_label})")
                fig.tight_layout()
                fig.savefig(png_path, dpi=100)
            finally:
                plt.close(fig)

            loaded += 1
            log.info("feature importance written: %s", window_label)

        except Exception as exc:
            log.warning(
                "write_feature_importance: failed for %s: %s; continuing",
                pkl_path,
                exc,
            )

    log.info(
        "write_feature_importance: %d of %d models loaded successfully",
        loaded,
        total,
    )
    return fi_dir


# ---------------------------------------------------------------------------
# Sub-function 3: run_slippage_sensitivity
# ---------------------------------------------------------------------------


def run_slippage_sensitivity(
    main_window_results: list[dict],
    exp_dir: Path,
) -> Path:
    """Build a 3-row slippage sensitivity CSV and backfill verdict.json.

    Row 0 (baseline): impact_cost = cost_model.slippage, is_baseline=True.
    Rows 1+ (config levels): re-backtest at each slippage level from
    config.slippage_sensitivity.

    If main_window_results is empty: log warning and return path without
    writing (returns expected path for consistent caller interface).

    If any window is skipped (missing pred_path): recompute baseline using
    only matched windows so denominators align for fragility comparison.

    Backfills slippage_sensitivity dict into verdict.json atomically.

    Args:
        main_window_results: List of WindowResult dicts from main walk-forward.
        exp_dir: Experiment directory.

    Returns:
        Path to exp_dir/slippage_sensitivity.csv.
    """
    cfg = load_config()
    out_path = exp_dir / "slippage_sensitivity.csv"

    if not main_window_results:
        log.warning("run_slippage_sensitivity: no window results; skipping")
        return out_path

    baseline_slippage: float = cfg["cost_model"]["slippage"]
    levels: list[float] = cfg["slippage_sensitivity"]
    fragility_cfg: dict = cfg["slippage_fragility"]
    rel_drop_threshold: float = fragility_cfg["rel_drop_threshold"]
    abs_tol: float = fragility_cfg["abs_tol"]

    # Step A: baseline row (no re-backtest).
    baseline_mean = float(
        sum(w["cumulative_excess_return"] for w in main_window_results)
        / len(main_window_results)
    )

    rows = [
        {
            "impact_cost": baseline_slippage,
            "mean_cumulative_excess": baseline_mean,
            "is_baseline": True,
        }
    ]

    # Collect per-level re-backtest results.
    per_level_results: list[dict] = []

    for level in levels:
        level_results: list[dict] = []
        level_backtested_ids: set[int] = set()

        for w in main_window_results:
            result = _rebacktest_window(
                window_result=w,
                n_drop=w["n_drop"],
                slippage_override=level,
            )
            if result is not None:
                level_results.append(result)
                level_backtested_ids.add(w["window_id"])

        per_level_results.append(
            {"level": level, "results": level_results, "ids": level_backtested_ids}
        )

    # Update CSV Row 0 baseline if any window was skipped across any level.
    all_input_ids = {w["window_id"] for w in main_window_results}
    # Intersection of per-level backtested IDs: only windows present in EVERY level.
    # Using intersection guarantees that CSV Row 0 (baseline) is computed from the
    # same window population visible in all level rows, making visual comparison
    # correct. Union would mask cross-level mismatches and leave Row 0 computed
    # from a different sample than the level rows that compare against it.
    # Include empty sets in the intersection: if any level produced zero results,
    # set.intersection(..., set()) = set(), so all_seen_ids correctly becomes
    # empty, triggering baseline recomputation from no windows (handled by the
    # guard below). Filtering out empty sets would exclude those levels from the
    # intersection and silently produce a non-empty all_seen_ids.
    level_id_sets = [entry["ids"] for entry in per_level_results]
    all_seen_ids: set[int] = (
        set.intersection(*level_id_sets) if level_id_sets else set()
    )

    if all_seen_ids and all_seen_ids != all_input_ids:
        matched_windows = [
            w for w in main_window_results if w["window_id"] in all_seen_ids
        ]
        recomputed_baseline = float(
            sum(w["cumulative_excess_return"] for w in matched_windows)
            / len(matched_windows)
        )
        log.warning(
            "run_slippage_sensitivity: %d/%d windows present in every level; "
            "updating baseline row (%.6f -> %.6f)",
            len(all_seen_ids),
            len(all_input_ids),
            baseline_mean,
            recomputed_baseline,
        )
        baseline_mean = recomputed_baseline
        rows[0]["mean_cumulative_excess"] = baseline_mean

    # Build non-baseline rows.
    for entry in per_level_results:
        level = entry["level"]
        level_results = entry["results"]

        if not level_results:
            log.warning(
                "run_slippage_sensitivity: zero windows matched for level %s",
                level,
            )
            level_mean = float("nan")
        else:
            level_mean = float(
                sum(r["cumulative_excess_return"] for r in level_results)
                / len(level_results)
            )

        rows.append(
            {
                "impact_cost": level,
                "mean_cumulative_excess": level_mean,
                "is_baseline": False,
            }
        )

    # Compute fragility: compare each level against a baseline from the SAME
    # windows that level successfully backtested. Using a different-sized
    # baseline sample for each level's rel_drop gives consistent denominators.
    is_fragile = False
    for entry in per_level_results:
        level_results = entry["results"]
        if not level_results:
            # Zero matched windows: treat as maximally fragile.
            log.warning(
                "run_slippage_sensitivity: level %s has zero matched windows; "
                "flagging as fragile",
                entry["level"],
            )
            is_fragile = True
            break

        level_mean = float(
            sum(r["cumulative_excess_return"] for r in level_results)
            / len(level_results)
        )
        # Compare against baseline_mean, which is exactly what CSV Row 0 shows
        # (intersection-based or original if no windows were skipped). Using
        # baseline_mean keeps the fragility decision visible from the CSV output.
        denominator = max(abs(baseline_mean), abs_tol)
        rel_drop = (baseline_mean - level_mean) / denominator
        if rel_drop > rel_drop_threshold:
            is_fragile = True
            break

    # Write CSV.
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info("slippage_sensitivity.csv written -> %s", out_path)

    # Build slippage_sensitivity dict for verdict.json backfill.
    sensitivity_rows = [
        {
            "impact_cost": r["impact_cost"],
            "mean_cumulative_excess": r["mean_cumulative_excess"],
            "is_baseline": r["is_baseline"],
        }
        for r in rows
    ]
    slippage_sensitivity_dict = {
        "levels": sensitivity_rows,
        "is_fragile": is_fragile,
    }

    # Atomic read-modify-write of verdict.json.
    verdict_path = exp_dir / "verdict.json"
    with verdict_path.open(encoding="utf-8") as f:
        verdict_dict = json.load(f)

    verdict_dict["slippage_sensitivity"] = slippage_sensitivity_dict
    _write_verdict_atomic(verdict_dict, verdict_path)
    log.info(
        "verdict.json backfilled: slippage_sensitivity.is_fragile=%s", is_fragile
    )

    return out_path


# ---------------------------------------------------------------------------
# Sub-function 4: run_control_track
# ---------------------------------------------------------------------------


def run_control_track(
    main_window_results: list[dict],
    exp_dir: Path,
) -> Path:
    """Re-backtest main predictions with n_drop=2 (CSI500 control track).

    Does NOT retrain. Reuses pred_path from each WindowResult. Writes
    verdict_control.json atomically. Gate result is informational; does NOT
    trigger sys.exit.

    Args:
        main_window_results: List of WindowResult dicts from main walk-forward.
        exp_dir: Experiment directory.

    Returns:
        Path to exp_dir/verdict_control.json.
    """
    cfg = load_config()
    gate_config: dict = cfg["gate"]
    topk: int = cfg["strategy"]["topk"]
    n_drop: int = cfg["strategy"]["control"]["n_drop"]
    universe: str = cfg["universe"]["primary"]

    control_results: list[dict] = []
    for w in main_window_results:
        result = _rebacktest_window(
            window_result=w,
            n_drop=n_drop,
            slippage_override=None,
        )
        if result is not None:
            control_results.append(result)

    track = f"topk{topk}_ndrop{n_drop}"
    verdict_dict = build_verdict(
        window_results=control_results,
        track=track,
        universe=universe,
        gate_config=gate_config,
    )

    out_path = exp_dir / "verdict_control.json"
    _write_verdict_atomic(verdict_dict, out_path)
    log.info(
        "verdict_control.json written: gate=%s track=%s",
        verdict_dict["gate"],
        track,
    )
    return out_path


# ---------------------------------------------------------------------------
# Sub-function 5: run_csi300_reference
# ---------------------------------------------------------------------------


def run_csi300_reference(exp_dir: Path) -> tuple[Path, Path]:
    """Run a full CSI300 walk-forward and write two verdict files.

    Fully isolated under exp_dir/csi300/. Never touches main exp_dir/models
    or main predictions.

    Walk-forward uses n_drop = cfg["strategy"]["main"]["n_drop"]. Then
    re-backtests with n_drop = cfg["strategy"]["control"]["n_drop"] for
    the control verdict. Gate results are informational (no sys.exit).

    Args:
        exp_dir: Parent experiment directory. CSI300 artifacts written under
            exp_dir/{ref_universe}/.

    Returns:
        (csi300_verdict_path, csi300_control_verdict_path) in that order.
    """
    cfg = load_config()
    gate_config: dict = cfg["gate"]
    topk: int = cfg["strategy"]["topk"]
    ref_universe: str = cfg["universe"]["reference"]
    main_n_drop: int = cfg["strategy"]["main"]["n_drop"]
    control_n_drop: int = cfg["strategy"]["control"]["n_drop"]

    csi300_dir = exp_dir / ref_universe
    csi300_dir.mkdir(parents=True, exist_ok=True)

    # Run full walk-forward under csi300_dir (isolated; does not touch main models).
    csi300_window_results = run_full_walk_forward(
        exp_dir=csi300_dir,
        n_drop=main_n_drop,
        universe=ref_universe,
        pred_dir=csi300_dir / "predictions",
    )

    # Build verdict for n_drop=main_n_drop.
    main_track = f"topk{topk}_ndrop{main_n_drop}"
    verdict_dict = build_verdict(
        window_results=csi300_window_results,
        track=main_track,
        universe=ref_universe,
        gate_config=gate_config,
    )
    # Prefix note for clarity.
    verdict_dict["note"] = f"REFERENCE TRACK ({ref_universe}): " + verdict_dict["note"]

    csi300_verdict_path = exp_dir / "verdict_csi300.json"
    _write_verdict_atomic(verdict_dict, csi300_verdict_path)
    log.info(
        "verdict_csi300.json written: gate=%s track=%s universe=%s",
        verdict_dict["gate"],
        main_track,
        ref_universe,
    )

    # Control: re-backtest with control n_drop from csi300 predictions.
    control_results: list[dict] = []
    for w in csi300_window_results:
        # Override universe to ref_universe for the control re-backtest.
        w_copy = dict(w)
        w_copy["universe"] = ref_universe
        result = _rebacktest_window(
            window_result=w_copy,
            n_drop=control_n_drop,
            slippage_override=None,
        )
        if result is not None:
            control_results.append(result)

    control_track = f"topk{topk}_ndrop{control_n_drop}"
    control_verdict_dict = build_verdict(
        window_results=control_results,
        track=control_track,
        universe=ref_universe,
        gate_config=gate_config,
    )
    control_verdict_dict["note"] = (
        f"REFERENCE TRACK ({ref_universe}): " + control_verdict_dict["note"]
    )

    csi300_control_verdict_path = exp_dir / "verdict_csi300_control.json"
    _write_verdict_atomic(control_verdict_dict, csi300_control_verdict_path)
    log.info(
        "verdict_csi300_control.json written: gate=%s track=%s universe=%s",
        control_verdict_dict["gate"],
        control_track,
        ref_universe,
    )

    return csi300_verdict_path, csi300_control_verdict_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_supplementary(
    main_window_results: list[dict],
    exp_dir: Path,
) -> None:
    """Orchestrate all supplementary analysis steps after gate PASS/BORDER_PASS.

    Reads verdict.json at entry; returns None early (with a warning) if:
    - verdict.json does not exist (FileNotFoundError)
    - gate is not PASS or BORDER_PASS (defensive re-check)

    Execution order (critical):
      1. finalize_artifacts -- consume main models before any other track runs
      2. write_feature_importance -- reads main models (must run before CSI300)
      3. run_slippage_sensitivity -- re-backtest main predictions at varied slippage
      4. run_control_track -- CSI500 n_drop=2 re-backtest
      5. run_csi300_reference -- fully isolated CSI300 walk-forward

    Steps 1-2 are fail-fast (exceptions propagate out). Steps 3-5 are each
    wrapped in try/except: log error and continue on failure.

    Args:
        main_window_results: List of WindowResult dicts from the main CSI500
            walk-forward (returned by run_full_walk_forward in report.py).
        exp_dir: Experiment directory containing verdict.json.

    Returns:
        None always (callers check artifacts on disk, not return value).
    """
    # Read verdict.json at entry (defensive re-check).
    verdict_path = exp_dir / "verdict.json"
    try:
        with verdict_path.open(encoding="utf-8") as f:
            verdict_data = json.load(f)
    except FileNotFoundError:
        log.warning(
            "run_all_supplementary: verdict.json not found at %s; skipping",
            verdict_path,
        )
        return None

    gate: str = verdict_data.get("gate", "")
    if gate not in ("PASS", "BORDER_PASS"):
        log.info(
            "run_all_supplementary: gate=%s; skipping supplementary analysis",
            gate,
        )
        return None

    models_dir = exp_dir / "models"

    # Steps 1-2: fail-fast (exceptions propagate out).
    finalize_artifacts(models_dir)
    write_feature_importance(models_dir, exp_dir)

    # Steps 3-5: each wrapped; log and continue on failure.
    try:
        run_slippage_sensitivity(main_window_results, exp_dir)
    except Exception as exc:
        log.error("run_slippage_sensitivity failed: %s", exc, exc_info=True)

    try:
        run_control_track(main_window_results, exp_dir)
    except Exception as exc:
        log.error("run_control_track failed: %s", exc, exc_info=True)

    try:
        run_csi300_reference(exp_dir)
    except Exception as exc:
        log.error("run_csi300_reference failed: %s", exc, exc_info=True)

    return None
