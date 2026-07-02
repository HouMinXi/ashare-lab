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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


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
    windows = get_all_windows()
    window: dict | None = None

    for w in windows:
        if w["test_start"] <= trade_date <= w["test_end"]:
            window = w
            break

    if window is None:
        if windows and trade_date > windows[-1]["test_end"]:
            # Live / out-of-sample: use the highest trained main window.
            window = windows[-1]
            if model_path is None:
                model_path = MODELS_DIR / "latest.pt"
        elif not windows or trade_date < windows[0]["test_start"]:
            raise ValueError(
                "no trained model covers %s "
                "(earliest test_start=%s)"
                % (trade_date, windows[0]["test_start"] if windows else "N/A")
            )

    window_id: int = window["window_id"]

    if model_path is None:
        model_path = MODELS_DIR / f"w{window_id}.pt"
        if not model_path.exists():
            # Window's model not yet trained; fall back to latest.pt
            fallback = MODELS_DIR / "latest.pt"
            if fallback.exists():
                log.warning("w%d.pt missing, falling back to latest.pt", window_id)
                model_path = fallback
            # else: let the FileNotFoundError below fire with the original path

    if not model_path.exists():
        raise FileNotFoundError(
            "model file %s does not exist "
            "(models are gitignored; train on the GPU host or sync them)"
            % model_path
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

    # -- 9. Build output DataFrame -----------------------------------------
    df = pd.DataFrame(
        {
            "instrument": day.index.astype(str),
            "score": day.to_numpy(dtype=float),
        }
    )

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
