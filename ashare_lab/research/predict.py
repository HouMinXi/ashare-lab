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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

# Set by _resolve_model_path: True when the loaded model is the one
# expected_live_model pins. A pinned model is a standing decision, so its
# age must not page anyone; the age is still logged.
_LAST_RESOLUTION: dict[str, bool] = {"deliberate": False}

_MODEL_STALE_DAYS = 7  # warn if model file is older than this

# -- Alert infrastructure (fail-open, stdlib only) --------------------------
_ALERT_SENT_THIS_RUN = False  # rate-sanity: one alert per predict run

def _send_alert(text: str) -> None:
    """Send a plain-text alert via the shared bridge module. Fail-open.

    One alert per predict run (module-level _ALERT_SENT_THIS_RUN flag,
    set only after a successful send).  Transport and token resolution
    live in ashare_lab.bridge.
    """
    global _ALERT_SENT_THIS_RUN
    if _ALERT_SENT_THIS_RUN:
        return

    try:
        from ashare_lab.bridge import send_bridge_alert
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
    import numpy as np

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
        from scipy.stats import kurtosis as sp_kurtosis
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


def _model_age_days(
    model_meta: dict | None,
    model_path: Path,
    today: date,
    meta_reason: str = "",
) -> tuple[float, str]:
    """Days since this model was produced, and where the number came from.

    meta.json's train_date is the cutoff of the training data, and in a
    walk-forward window that cutoff lags today by months: the model retrained
    on 2026-11-07 from window 11 carries train_date 2026-06-30 and would be
    reported as 131 days stale the hour it was written. Staleness answers "has
    anyone retrained", so anchor on trained_at and fall back to the cutoff only
    when the timestamp is absent, then to the file's modification time.
    """
    if model_meta:
        trained_at = model_meta.get("trained_at")
        if trained_at:
            try:
                stamp = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
                return float((today - stamp.date()).days), "meta.json(trained_at)"
            except ValueError:
                pass
        train_date = model_meta.get("train_date")
        if train_date:
            try:
                return float((today - date.fromisoformat(str(train_date))).days), "meta.json(train_date)"
            except ValueError:
                pass
    age = (time.time() - model_path.stat().st_mtime) / 86400
    return age, f"mtime-fallback({meta_reason})" if meta_reason else "mtime-fallback"


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
    # Reset per call: the flag is module state, and a stale True would
    # suppress a genuine staleness alert on a later trade date.
    _LAST_RESOLUTION["deliberate"] = False
    window: dict | None = None

    # The newest walk-forward window clamps its test_end to the latest trading
    # day, so "past every window" never holds for a production date: the live
    # artifact would never be selected. Choose it explicitly for the day being
    # predicted, and leave historical dates on their window models.
    from ashare_lab.research.smoke_test import (
        get_live_window,
        latest_trading_day,
    )

    live_model = models_dir / "wlive.pt"
    if (
        model_path is None
        and live_model.is_file()
        and trade_date >= str(latest_trading_day())
    ):
        # The window carries the fit range the handler normalises with, so it
        # has to be the span this model was trained on -- not whichever
        # walk-forward window happens to cover the date, which would feed the
        # model features scaled over a different period.
        window = dict(get_live_window(), window_id="wlive")
        model_path = live_model

    for w in windows:
        if w["test_start"] <= trade_date <= w["test_end"]:
            if window is None:
                window = w
            break

    if window is None:
        if windows and trade_date > windows[-1]["test_end"]:
            window = windows[-1]
            live_model = models_dir / "wlive.pt"
            if model_path is None and live_model.is_file():
                # The live window is trained through the last trading day and
                # has no held-out period; it is the artifact meant for dates
                # after every walk-forward window has closed. Its artifact id
                # is "wlive", matching the file name and models/meta.json.
                window = {**windows[-1], "window_id": "wlive"}
                model_path = live_model
            elif model_path is None:
                model_path = models_dir / "latest.pt"
        elif not windows or trade_date < windows[0]["test_start"]:
            raise ValueError(
                "no trained model covers %s "
                "(earliest test_start=%s)"
                % (trade_date, windows[0]["test_start"] if windows else "N/A")
            )

    window_id = window["window_id"]

    if model_path is None:
        model_path = models_dir / f"w{window_id}.pt"
        if not model_path.exists():
            fallback = models_dir / "latest.pt"
            if fallback.exists():
                expected = cfg.get("research", {}).get("expected_live_model")
                # Check if latest.pt contains the expected model.
                # Works for both symlinks (W1) and regular-file copies (gpu-win).
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

    # Inspect the selected identity after every automatic/explicit route. Both
    # symlink targets and metadata-bearing copies are normal deployment layouts.
    from ashare_lab.research.model_meta import has_research_identity

    if has_research_identity(model_path):
        raise ValueError("research-only checkpoint: scheduled prediction cannot reconstruct its fit contract")

    # A pinned model is a standing decision regardless of which branch
    # resolved the path, so judge it by content here, once.
    expected = cfg.get("research", {}).get("expected_live_model")
    expected_file = models_dir / f"{expected}.pt" if expected else None
    if (
        expected_file is not None
        and expected_file.exists()
        and model_path.exists()
        and filecmp.cmp(str(model_path), str(expected_file), shallow=False)
    ):
        _LAST_RESOLUTION["deliberate"] = True

    if not model_path.exists():
        raise FileNotFoundError(
            "model file %s does not exist "
            "(models are gitignored; train on the GPU host or sync them)"
            % model_path
        )

    return model_path, window


def predict_without_memory_write(model, data_set):
    """Run TRA inference over a prepared view without the loss-memory store.

    Stock ``TRAModel.predict`` calls ``test_epoch``, which stores each batch's
    loss into the dataset before returning the rows. A live view refuses every
    write by contract. The store is only read by later endpoints, and a live
    view covers one session that evaluates as a single batch, so the same
    forward pass returns the same rows with no write at all. The row assembly
    below mirrors ``test_epoch`` including its index restoration.
    """
    import numpy as np
    import pandas as pd
    import torch

    if not model.fitted:
        raise ValueError("model is not fitted yet!")
    model.model.eval()
    model.tra.eval()
    data_set.eval()

    frames = []
    for batch in data_set:
        data, state = batch["data"], batch["state"]
        label, count = batch["label"], batch["daily_count"]
        with torch.no_grad():
            hidden = model.model(data)
            all_preds, choice, prob = model.tra(hidden, state)
        if model.transport_method != "none":
            _loss, pred, _store, _weights = model.transport_fn(
                all_preds,
                label,
                choice,
                prob,
                state.mean(dim=1),
                count,
                model.transport_method,
                model.alpha,
                training=False,
            )
        else:
            pred = all_preds.mean(dim=1)
        values = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
        columns = ["score", "label"] + [f"score_{d}" for d in range(all_preds.shape[1])]
        frames.append(pd.DataFrame(values, index=batch["index"], columns=columns))

    preds = pd.concat(frames, axis=0)
    preds.index = data_set.restore_index(preds.index)
    preds.index = preds.index.swaplevel()
    preds.sort_index(inplace=True)
    return preds


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
    import numpy as np
    import pandas as pd
    import torch
    import qlib
    from qlib.config import C as _QlibC
    from qlib.config import REG_CN
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from ashare_lab.research.sequence_dataset import (
        ContainedMTSDatasetH,
        SegmentAdmission,
    )

    from ashare_lab.config import MODELS_DIR, PREDICTIONS_DIR, load_config
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI
    from ashare_lab.research.blend import blend_tra_ntra
    from ashare_lab.research.smoke_test import get_all_windows
    from ashare_lab.research.train import ALPHA158_WARMUP_START

    # -- 1. Window selection (resolves both window dict and model path) ----
    model_path, window = _resolve_model_path(
        trade_date, model_path, get_all_windows(), load_config(), MODELS_DIR,
    )
    window_id = window["window_id"]

    # -- Model freshness check (meta.json preferred, mtime fallback) --
    from ashare_lab.research.model_meta import read_model_meta

    model_meta, meta_reason = read_model_meta(model_path)
    # fixed +08:00 offset, no tzdata dependency (works on hosts without IANA data)
    today = datetime.now(timezone(timedelta(hours=8))).date()
    model_age_days, age_source = _model_age_days(
        model_meta, model_path, today, meta_reason
    )
    if model_age_days > _MODEL_STALE_DAYS:
        pinned = _LAST_RESOLUTION["deliberate"]
        log.warning(
            "MODEL STALE: %s is %.0f days old (threshold: %d days, source: %s)%s",
            model_path.name, model_age_days, _MODEL_STALE_DAYS, age_source,
            " [pinned by expected_live_model, not paging]"
            if pinned
            else ". Consider retraining.",
        )
        if not pinned:
            _send_alert(
                f"[predict] model stale: {model_path.name} is "
                f"{model_age_days:.1f} days old (threshold: {_MODEL_STALE_DAYS}, "
                f"source: {age_source}) on {trade_date}. Action: retrain"
            )

    log.info(
        "predict %s: window W%s, model %s",
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

    # -- 4. ContainedMTSDatasetH (one live session) -----------------------
    # Alpha158 materialises its tables in its constructor; the adapter needs
    # them to admit endpoints.
    learned = handler._learn
    if learned is None:
        raise RuntimeError("Alpha158 did not materialise its learn table")
    # The admission calendar is the provider's session calendar, not the union
    # of dates that happen to carry rows: a hole that removes a session for
    # every instrument must read as a calendar gap, not as two consecutive
    # sessions.
    observed = pd.DatetimeIndex(learned.index.get_level_values("datetime"))
    calendar = pd.DatetimeIndex(
        D.calendar(start_time=observed.min(), end_time=observed.max(), freq="day")
    )
    stray = observed.unique().difference(calendar)
    if len(stray):
        raise RuntimeError(
            "handler rows fall outside the provider calendar "
            f"(first few: {[str(d.date()) for d in stray[:5]]}); the calendar is stale "
            "relative to the data"
        )
    # The trade date's forward return does not exist yet. A NaN label is
    # fine for a live admission, which never waits for label maturity.
    # The adapter's parameter is called raw_labels and it only reads
    # finiteness off these values, but what the handler carries here is the
    # processed label column.
    label_column = next(c for c in learned.columns if c[0] == "label")
    processed_labels = learned[label_column].swaplevel().sort_index()
    live_day = pd.Timestamp(trade_date)
    dataset = ContainedMTSDatasetH(
        handler=handler,
        segments={
            "test": SegmentAdmission(
                start=live_day,
                end=live_day,
                purpose="live",
                outcome_cutoff=live_day,
            )
        },
        calendar=calendar,
        raw_labels=processed_labels,
        seq_len=step_len,
        num_states=num_states,
        batch_size=-1,
        shuffle=True,
    )

    # -- 5. Load model -----------------------------------------------------
    model = torch.load(str(model_path), weights_only=False)
    # -- 6. Predict --------------------------------------------------------
    # ``model.predict`` stores each batch's loss in dataset memory, and a live
    # view refuses writes by design; this entry returns the same rows without
    # the store.
    try:
        live_view = dataset.prepare("test")
    except ValueError as exc:
        if "admits no endpoint" in str(exc):
            raise ValueError(
                f"{exc}: no endpoint on {trade_date} passes the admission rule "
                f"(seq_len={step_len}); one missing session breaks every window "
                "that spans it, so check the provider data for a hole"
            ) from exc
        raise
    admission = dict(live_view._diagnostics)
    excluded = admission["input_in_range"] - admission["admitted"]
    if excluded:
        log.warning(
            "predict %s: %d of %d instruments not admitted "
            "(%d without enough own history, %d across a calendar gap)",
            trade_date,
            excluded,
            admission["input_in_range"],
            admission["insufficient_own_history"],
            admission["calendar_gap"],
        )
    pred_df = predict_without_memory_write(model, live_view)
    pred = pred_df["score"]
    # Count the live session only: the adapter answers for one day, but a
    # model that returns more would otherwise inflate the funnel.
    scored_rows = int(
        (pred.index.get_level_values(0) == pd.Timestamp(trade_date)).sum()
    )
    if scored_rows != admission["admitted"]:
        # The adapter admits a set and returns one row per member. A different
        # count is an assembly defect; refuse before the blend spends the day
        # fetching factors for a cross-section nobody admitted.
        raise ValueError(
            f"scored {scored_rows} rows for {admission['admitted']} admitted "
            f"instruments on {trade_date}"
        )

    # -- 7. Blend (reuse, locked 0.60/0.40) --------------------------------
    # Let any blend exception PROPAGATE: the file contract says score IS the
    # locked blend, so writing raw TRA scores would violate it. No fallback.
    blended = blend_tra_ntra(pred, window)
    blended_rows = int(len(blended[blended.index.get_level_values(0) == pd.Timestamp(trade_date)]))

    # -- 8. Extract trade_date rows, drop non-finite -----------------------
    ts = pd.Timestamp(trade_date)
    if ts not in blended.index.get_level_values(0):
        raise ValueError(
            "no predictions for %s after blending "
            "(non-trading day or missing bars)" % trade_date
        )

    day = blended.loc[ts]
    finite_rows = int(np.isfinite(day.to_numpy(dtype=float)).sum())
    day = day[np.isfinite(day.to_numpy(dtype=float))]
    funnel = {
        "admitted": admission["admitted"],
        "scored": scored_rows,
        "after_blend": blended_rows,
        "after_finite_filter": finite_rows,
    }
    if blended_rows != scored_rows or finite_rows != blended_rows:
        # Blend and the finite filter legitimately drop instruments, but the
        # written file is then narrower than the admitted universe.
        log.warning(
            "predict %s: rows lost inside the funnel %s; the written file is "
            "narrower than the admitted universe",
            trade_date,
            funnel,
        )

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

    sidecar = {
        "model": model_path.name,
        "universe": universe,
        "window_id": window_id,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "n_instruments": len(df),
        "admission": admission,
        "funnel": funnel,
        "admission_rule": {
            "purpose": "live",
            "seq_len": step_len,
            "calendar_start": str(calendar[0].date()),
            "calendar_end": str(calendar[-1].date()),
            "label_rule": "not consulted for a live admission",
        },
        "model_age_days": round(model_age_days, 1),
        "model_age_source": age_source,
        "ic": ic_value,
        "diversity": diversity.tier2_metrics,
    }
    if model_meta is not None and "train_date" in model_meta:
        sidecar["model_train_date"] = model_meta["train_date"]
    meta_path = PREDICTIONS_DIR / f"{trade_date}.meta.json"
    meta_path.write_text(json.dumps(sidecar, indent=2))

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
