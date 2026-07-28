"""Daily prediction producer: TRA inference + 60/40 blend -> parquet.

Provides predict_for_date(), which runs TRA inference for one trade_date,
applies the locked 60/40 TRA/nTRA blend via research/blend.py, and writes
predictions/{trade_date}.parquet (columns: instrument, score).

This is the Option 3 producer: the paper engine reads prediction files and
never loads a model; all model/qlib/torch dependencies live here.

All heavy imports (torch, qlib, model classes) are deferred inside
predict_for_date() so this module is importable without a qlib/torch
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import filecmp
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

_MODEL_STALE_DAYS = 7  # warn if model file is older than this

# -- Alert infrastructure (fail-open, stdlib only) --------------------------
_ALERT_SENT_THIS_RUN = False  # rate-sanity: one alert per predict run

def _send_alert(text: str) -> None:
    """Send a plain-text alert via the shared bridge module. Fail-open.

    One alert per predict run (module-level _ALERT_SENT_THIS_RUN flag,
    set only after a successful send).  Transport and token resolution
    live in ashare_lab.bridge.
    """
    global _ALERT_SENT_THIS_RUN  # noqa: PLW0603
    if _ALERT_SENT_THIS_RUN:
        return

    try:
        from ashare_lab.bridge import send_bridge_alert  # noqa: PLC0415
    except Exception:
        # gpu-win receives a selective file sync; a missing bridge module
        # must never break prediction (alerting is advisory by design).
        log.warning("alert skipped: bridge module unavailable", exc_info=True)
        return

    if send_bridge_alert("ashare-predict", text):
        _ALERT_SENT_THIS_RUN = True


# ---------------------------------------------------------------------------
# Score diversity / collapse detection (Phase 9 -- 09-04)
# ---------------------------------------------------------------------------


@dataclass
class DiversityResult:
    """Outcome of score diversity check (3-tier gate).

    Attributes:
        tier1_failed: True if any Tier 1 metric breached threshold.
        tier1_detail: Human-readable description of which Tier 1 metric failed.
        tier2_metrics: Diagnostic metrics for meta.json (kurtosis, entropy, gini).
        tier3_drift: Drift detection result (std_ratio, kurtosis_ratio, alerted).
    """

    tier1_failed: bool = False
    tier1_detail: str = ""
    tier2_metrics: dict = field(default_factory=dict)
    tier3_drift: dict = field(default_factory=dict)


def check_score_diversity(
    scores: np.ndarray,
    history_path: Path | None = None,
) -> DiversityResult:
    """Three-tier collapse detection gate.

    Tier 1 -- Fail-fast (refuse parquet write):
        score_std < 0.001: zero variance = all identical
        unique_ratio < 0.10: <10% unique bins at 4dp precision
        iqr_range < 0.005: robust narrow-spread detection

    Tier 2 -- Diagnostic logging (written to meta.json):
        kurtosis: tail heaviness (Fisher/excess; -1.2 for uniform)
        entropy_normalized: Shannon entropy / log(n), 0=collapsed, 1=uniform
        gini: Gini coefficient on abs(scores), 0=equal, 1=concentrated

    Tier 3 -- Drift detection (rolling 30-day):
        Compare today's std/kurtosis against rolling median of last 30 days.
        Alert if today's value < 0.5 * baseline_median.
        Requires 2 consecutive failures OR 1 failure + Tier 1 same day.
        Skip if < 5 days of history (cold start).
    """
    import numpy as np  # noqa: PLC0415

    result = DiversityResult()
    n = len(scores)

    if n < 3:
        # Too few scores to assess diversity meaningfully.
        # Skip the gate (pass-through) rather than failing -- a tiny
        # universe is unusual but not a collapse signal.
        log.debug("Score diversity: only %d scores, skipping gate", n)
        return result

    # -- Tier 1 metrics --
    score_std = float(np.std(scores))
    unique_vals = len(set(np.round(scores, 4)))
    unique_ratio = unique_vals / n
    q1 = float(np.percentile(scores, 25))
    q3 = float(np.percentile(scores, 75))
    iqr_range = q3 - q1

    tier1_failures = []
    if score_std < 0.001:
        tier1_failures.append(f"score_std={score_std:.6f} < 0.001")
    if unique_ratio < 0.10:
        tier1_failures.append(f"unique_ratio={unique_ratio:.4f} < 0.10")
    if iqr_range < 0.005:
        tier1_failures.append(f"iqr_range={iqr_range:.6f} < 0.005")

    if tier1_failures:
        result.tier1_failed = True
        result.tier1_detail = "; ".join(tier1_failures)

    # -- Tier 2 metrics --
    # Kurtosis (Fisher/excess): -1.2 for uniform, 0 for normal, >0 for heavy tails
    try:
        from scipy.stats import kurtosis as sp_kurtosis  # noqa: PLC0415
        kurt = float(sp_kurtosis(scores, fisher=True))
        if not np.isfinite(kurt):
            kurt = 0.0
    except ImportError:
        # Fallback: manual excess kurtosis
        mean = np.mean(scores)
        std = score_std if score_std > 0 else 1e-10
        kurt = float(np.mean(((scores - mean) / std) ** 4) - 3.0)
        if not np.isfinite(kurt):
            kurt = 0.0

    # Shannon entropy normalized
    # Bin scores into 50 bins for entropy calculation
    hist, _ = np.histogram(scores, bins=50, density=True)
    hist = hist[hist > 0]
    if len(hist) > 0:
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        entropy_norm = 0.0

    # Gini coefficient on absolute values
    abs_scores = np.abs(scores)
    abs_sorted = np.sort(abs_scores)
    n_float = float(n)
    cumsum = np.cumsum(abs_sorted)
    gini = float(
        (n_float + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n_float
    ) if cumsum[-1] > 0 else 0.0

    result.tier2_metrics = {
        "score_std": round(score_std, 6),
        "kurtosis": round(kurt, 4),
        "entropy_normalized": round(entropy_norm, 4),
        "gini": round(gini, 4),
        "unique_ratio": round(unique_ratio, 4),
        "iqr_range": round(iqr_range, 6),
    }

    # -- Tier 3: drift detection (rolling 30-day) --
    result.tier3_drift = {"alerted": False}
    if history_path is not None and history_path.exists():
        try:
            with history_path.open() as f:
                history_meta = json.load(f)
            diversity_history = history_meta.get("diversity_history", [])
            if len(diversity_history) >= 5:
                # Extract rolling baseline (median of last 30 days)
                recent = diversity_history[-30:]
                stds = [d["score_std"] for d in recent if "score_std" in d]
                kurts = [d["kurtosis"] for d in recent if "kurtosis" in d]

                if stds and kurts:
                    baseline_std = float(np.median(stds))
                    baseline_kurt = float(np.median(kurts))

                    std_ratio = score_std / baseline_std if baseline_std > 0 else 1.0
                    kurt_ratio = kurt / baseline_kurt if abs(baseline_kurt) > 1e-6 else 1.0

                    std_alert = std_ratio < 0.5
                    kurt_alert = kurt_ratio < 0.5

                    # Require 2 consecutive failures OR 1 failure + Tier 1.
                    # Evaluate previous alerts from stored metrics (not
                    # the std_alert field which is always False on write).
                    consecutive = len(diversity_history) >= 2
                    if consecutive:
                        prev = diversity_history[-1]
                        prev_std = prev.get("score_std", 0)
                        prev_kurt = prev.get("kurtosis", 0)
                        prev_std_ratio = (
                            prev_std / baseline_std
                            if baseline_std > 0 else 1.0
                        )
                        prev_kurt_ratio = (
                            prev_kurt / baseline_kurt
                            if abs(baseline_kurt) > 1e-6 else 1.0
                        )
                        prev_std_alert = prev_std_ratio < 0.5
                        prev_kurt_alert = prev_kurt_ratio < 0.5
                        two_consecutive = (
                            (std_alert and prev_std_alert)
                            or (kurt_alert and prev_kurt_alert)
                        )
                    else:
                        two_consecutive = False

                    one_plus_tier1 = (std_alert or kurt_alert) and result.tier1_failed

                    if two_consecutive or one_plus_tier1:
                        result.tier3_drift = {
                            "alerted": True,
                            "std_ratio": round(std_ratio, 4),
                            "kurtosis_ratio": round(kurt_ratio, 4),
                            "baseline_std": round(baseline_std, 6),
                            "baseline_kurtosis": round(baseline_kurt, 4),
                        }
        except Exception:
            log.debug("Tier 3 drift check failed", exc_info=True)

    return result


def _update_diversity_history(
    history_path: Path,
    trade_date: str,
    tier2_metrics: dict,
    drift_alerted: bool,
) -> None:
    """Append today's diversity metrics to the rolling history file.

    Keeps last 60 days. Written as JSON with key "diversity_history".
    Each entry: {date, score_std, kurtosis, std_alert, kurtosis_alert}.
    """
    try:
        if history_path.exists():
            with history_path.open() as f:
                data = json.load(f)
        else:
            data = {}

        entries = data.get("diversity_history", [])

        # Deduplicate by date
        entries = [e for e in entries if e.get("date") != trade_date]

        entries.append({
            "date": trade_date,
            "score_std": tier2_metrics.get("score_std", 0),
            "kurtosis": tier2_metrics.get("kurtosis", 0),
            "std_alert": False,  # populated on next run's Tier 3 check
            "kurtosis_alert": False,
        })

        # Keep last 60 days
        entries = entries[-60:]
        data["diversity_history"] = entries

        tmp = history_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(history_path)
    except Exception:
        log.debug("Failed to update diversity history", exc_info=True)


def _resolve_model_path(
    trade_date: str,
    model_path: Path | None,
    windows: list[dict],
    cfg: dict,
    models_dir: Path,
) -> tuple[Path, dict]:
    """Resolve the model path and window for a trade date.

    Returns (model_path, window_dict).  May fire _send_alert for genuine
    missing-model situations; deliberately mismatches (expected_live_model)
    are logged at INFO only.
    """
    window: dict | None = None

    for w in windows:
        if w["test_start"] <= trade_date <= w["test_end"]:
            window = w
            break

    if window is None:
        if windows and trade_date > windows[-1]["test_end"]:
            window = windows[-1]
            if model_path is None:
                model_path = models_dir / "latest.pt"
        elif not windows or trade_date < windows[0]["test_start"]:
            raise ValueError(
                "no trained model covers %s "
                "(earliest test_start=%s)"
                % (trade_date, windows[0]["test_start"] if windows else "N/A")
            )

    window_id: int = window["window_id"]

    if model_path is None:
        model_path = models_dir / f"w{window_id}.pt"
        if not model_path.exists():
            fallback = models_dir / "latest.pt"
            if fallback.exists():
                expected = cfg.get("research", {}).get("expected_live_model")
                # Check if latest.pt contains the expected model.
                # Works for both symlinks (X500) and regular-file copies (gpu-win).
                expected_file = models_dir / f"{expected}.pt" if expected else None
                is_expected = (
                    expected
                    and expected_file is not None
                    and expected_file.exists()
                    and filecmp.cmp(str(fallback), str(expected_file), shallow=False)
                )
                if (
                    is_expected
                    and f"w{window_id}" != expected
                ):
                    log.info(
                        "window model w%d.pt not deployed "
                        "(expected_live_model=%s), using %s",
                        window_id, expected, fallback.name,
                    )
                else:
                    log.warning("w%d.pt missing, falling back to latest.pt", window_id)
                    _send_alert(
                        f"[predict] model fallback: w{window_id}.pt missing, "
                        f"using {fallback.name} on {trade_date}. "
                        f"Action: train w{window_id}.pt"
                    )
                model_path = fallback

    if not model_path.exists():
        raise FileNotFoundError(
            "model file %s does not exist "
            "(models are gitignored; train on the GPU host or sync them)"
            % model_path
        )

    return model_path, window


def predict_for_date(
    trade_date: str,
    model_path: Path | None = None,
    provider_uri: str | None = None,
) -> Path:
    """Run TRA inference for one trade_date, blend, write prediction file.

    Loads the walk-forward model for *trade_date*, builds Alpha158 features
    over the model's training window, runs TRA prediction, applies the
    locked 60/40 TRA/nTRA blend (via blend_tra_ntra), drops non-finite
    scores, and writes the result as predictions/{trade_date}.parquet.

    Args:
        trade_date: ISO date string (YYYY-MM-DD) for the prediction day.
        model_path: Explicit model file override. When None (the normal
            case), the walk-forward window selects the correct model
            automatically; passing an explicit path defeats look-ahead
            protection on a backfill.
        provider_uri: qlib data directory. When None, falls back to
            DEFAULT_PROVIDER_URI (~/.qlib/qlib_data/cn_data). Pass
            the GPU path (H:/.qlib/qlib_data/cn_data) when running
            on the Windows training host.

    Returns:
        Path to the written parquet file.

    Raises:
        ValueError: trade_date predates all trained windows, or no rows
            survive for trade_date after inference and filtering.
        FileNotFoundError: the resolved model .pt does not exist (models
            are gitignored; train on the GPU host or sync them).
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import qlib  # noqa: PLC0415
    from qlib.config import C as _QlibC  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
    from qlib.contrib.data.dataset import MTSDatasetH  # noqa: PLC0415

    from ashare_lab.config import MODELS_DIR, PREDICTIONS_DIR, load_config  # noqa: PLC0415
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415
    from ashare_lab.research.blend import blend_tra_ntra  # noqa: PLC0415
    from ashare_lab.research.smoke_test import get_all_windows  # noqa: PLC0415
    from ashare_lab.research.train import ALPHA158_WARMUP_START  # noqa: PLC0415

    # -- 1. Window selection (resolves both window dict and model path) ----
    model_path, window = _resolve_model_path(
        trade_date, model_path, get_all_windows(), load_config(), MODELS_DIR,
    )
    window_id: int = window["window_id"]

    # -- Model freshness check --
    model_age_days = (time.time() - model_path.stat().st_mtime) / 86400
    if model_age_days > _MODEL_STALE_DAYS:
        log.warning(
            "MODEL STALE: %s is %.0f days old (threshold: %d days). "
            "Consider retraining.",
            model_path.name, model_age_days, _MODEL_STALE_DAYS,
        )
        _send_alert(
            f"[predict] model stale: {model_path.name} is {model_age_days:.1f} days old "
            f"(threshold: {_MODEL_STALE_DAYS}) on {trade_date}. "
            f"Action: retrain"
        )

    log.info(
        "predict %s: window W%d, model %s",
        trade_date,
        window_id,
        model_path.name,
    )

    # -- 2. Init qlib (idempotent) ----------------------------------------
    uri = provider_uri or str(DEFAULT_PROVIDER_URI)
    if not getattr(_QlibC, "registered", False):
        qlib.init(provider_uri=uri, region=REG_CN)

    cfg = load_config()
    universe: str = cfg["universe"]["primary"]
    cfg_model: dict = cfg.get("model", {})
    step_len: int = int(cfg_model.get("step_len", 20))
    num_states: int = int(
        cfg_model.get("routing", {}).get("num_states", 3)
    )

    # -- 3. Alpha158 handler (window's train fit-range) --------------------
    fit_start: str = window["train_start"]
    fit_end: str = window["train_end"]

    # learn_processors: train.py's TRA list MINUS DropnaLabel.
    # Omitting DropnaLabel keeps the live date whose forward-return label
    # is NaN; the TRA router score is label-independent (spike-confirmed).
    learn_processors = [
        {
            "class": "CSZScoreNorm",
            "kwargs": {"fields_group": "label"},
        },
        {
            "class": "RobustZScoreNorm",
            "kwargs": {"fields_group": "feature", "clip_outlier": True},
        },
        {
            "class": "Fillna",
            "kwargs": {"fields_group": "feature"},
        },
    ]

    handler = Alpha158(
        instruments=universe,
        start_time=ALPHA158_WARMUP_START,
        end_time=trade_date,
        fit_start_time=fit_start,
        fit_end_time=fit_end,
        learn_processors=learn_processors,
    )

    # -- 4. MTSDatasetH (single-day test segment) -------------------------
    dataset = MTSDatasetH(
        handler=handler,
        segments={"test": (trade_date, trade_date)},
        seq_len=step_len,
        num_states=num_states,
        memory_mode="sample",
        batch_size=-1,
        shuffle=True,
    )

    # -- 5. Load model + writer shim --------------------------------------
    model = torch.load(str(model_path), weights_only=False)
    # qlib drops the TensorBoard SummaryWriter when pickling TRAModel;
    # test_epoch references self._writer and the epoch>=0 guard is never
    # reached because predict calls test_epoch(-1). Set unconditionally
    # so inference never crashes on a missing or broken writer.
    model._writer = None

    # -- 6. Predict --------------------------------------------------------
    pred_df = model.predict(dataset, segment="test")
    pred = pred_df["score"]

    # -- 7. Blend (reuse, locked 0.60/0.40) --------------------------------
    # Let any blend exception PROPAGATE: the file contract says score IS the
    # locked blend, so writing raw TRA scores would violate it. No fallback.
    blended = blend_tra_ntra(pred, window)

    # -- 8. Extract trade_date rows, drop non-finite -----------------------
    ts = pd.Timestamp(trade_date)
    if ts not in blended.index.get_level_values(0):
        raise ValueError(
            "no predictions for %s after blending "
            "(non-trading day or missing bars)" % trade_date
        )

    day = blended.loc[ts]
    day = day[np.isfinite(day.to_numpy(dtype=float))]

    if day.empty:
        raise ValueError(
            "all scores for %s are non-finite after filtering; "
            "refusing to write empty prediction file" % trade_date
        )

    # -- 8b. Score diversity gate (Phase 9 -- 09-04) ------------------------
    # Check for model collapse: all-identical or near-zero variance scores.
    # Tier 1 failure refuses parquet write entirely.
    diversity_history_path = PREDICTIONS_DIR / "diversity_history.json"
    diversity = check_score_diversity(
        day.to_numpy(dtype=float), diversity_history_path,
    )
    if diversity.tier1_failed:
        raise ValueError(
            "Prediction collapse detected for %s: %s" % (
                trade_date, diversity.tier1_detail,
            )
        )

    # Log Tier 2 diagnostics
    if diversity.tier2_metrics:
        log.info(
            "Score diversity: std=%.6f kurt=%.4f entropy=%.4f gini=%.4f",
            diversity.tier2_metrics.get("score_std", 0),
            diversity.tier2_metrics.get("kurtosis", 0),
            diversity.tier2_metrics.get("entropy_normalized", 0),
            diversity.tier2_metrics.get("gini", 0),
        )

    # Log Tier 3 drift alert
    if diversity.tier3_drift.get("alerted"):
        log.warning(
            "DIVERSITY DRIFT ALERT: std_ratio=%.4f kurtosis_ratio=%.4f "
            "(baseline: std=%.6f kurt=%.4f)",
            diversity.tier3_drift.get("std_ratio", 0),
            diversity.tier3_drift.get("kurtosis_ratio", 0),
            diversity.tier3_drift.get("baseline_std", 0),
            diversity.tier3_drift.get("baseline_kurtosis", 0),
        )

    # Update diversity history for Tier 3 drift detection
    _update_diversity_history(
        diversity_history_path, trade_date, diversity.tier2_metrics,
        diversity.tier3_drift.get("alerted", False),
    )

    # -- 9. Build output DataFrame -----------------------------------------
    df = pd.DataFrame(
        {
            "instrument": day.index.astype(str),
            "score": day.to_numpy(dtype=float),
        }
    )

    # -- 9b. Capture IC from TRA model internal state ----------------------
    # The TRA model computes RankIC during test_epoch.  When predicting
    # beyond the training window (live / out-of-sample), no labels exist
    # so the internal IC is NaN.  We surface this in meta.json so the
    # pipeline and report can warn the operator.
    ic_value = None
    try:
        # TRAModel stores per-epoch IC in _ic_list or similar attributes
        # depending on qlib version.  Try the common patterns.
        ic_attr = getattr(model, "_ic_list", None)
        if ic_attr and len(ic_attr) > 0:
            last_ic = ic_attr[-1]
            ic_value = float(last_ic) if np.isfinite(last_ic) else None
        else:
            # Fallback: check if model has a rank_ic or ic attribute
            for attr_name in ("rank_ic", "_rank_ic", "ic"):
                val = getattr(model, attr_name, None)
                if val is not None:
                    try:
                        fval = float(val)
                        ic_value = fval if np.isfinite(fval) else None
                    except (TypeError, ValueError):
                        pass
                    break
    except Exception:
        pass  # best-effort; IC capture never blocks prediction

    if ic_value is not None and math.isnan(ic_value):
        ic_value = None

    # -- 10. Write parquet + provenance sidecar ----------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = PREDICTIONS_DIR / f"{trade_date}.parquet"
    df.to_parquet(out, index=False)

    meta = {
        "model": model_path.name,
        "universe": universe,
        "window_id": window_id,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "n_instruments": len(df),
        "model_age_days": round(model_age_days, 1),
        "ic": ic_value,
        "diversity": diversity.tier2_metrics,
    }
    meta_path = PREDICTIONS_DIR / f"{trade_date}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info(
        "predict %s: wrote %d instruments -> %s",
        trade_date,
        len(df),
        out,
    )

    return out


if __name__ == "__main__":
    import argparse
    import gc

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run TRA inference")
    parser.add_argument("--date", required=True, help="trade date YYYY-MM-DD")
    parser.add_argument(
        "--provider-uri",
        default=None,
        help="qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    args = parser.parse_args()
    predict_for_date(args.date, provider_uri=args.provider_uri)

    # Ensure GPU memory is released before process exits.
    # Windows CUDA driver may not reclaim VRAM promptly without this
    # when batch scripts launch hundreds of predict subprocesses.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()
