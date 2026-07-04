"""Model-agnostic shadow predictor for matrix experiment candidates.

Unlike predict.py (TRA-only with 60/40 blend), this module runs inference
for any supported model type (lgbm, alstm, tra, densemble) and writes
output to shadow_predictions/{model_tag}/{trade_date}.parquet -- strictly
isolated from the production predictions/ directory.

All heavy imports (torch, qlib, model classes) are deferred inside
shadow_predict_for_date() so this module is importable without a
qlib/torch runtime.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def shadow_predict_for_date(
    trade_date: str,
    config_path: str | Path,
    model_dir: Path | None = None,
    provider_uri: str | None = None,
) -> Path:
    """Run inference for one trade_date using a matrix candidate config.

    Merges the candidate config's model section into the baseline, resolves
    the walk-forward window covering trade_date, loads the trained model,
    and writes predictions to shadow_predictions/{tag}/{trade_date}.parquet.

    Args:
        trade_date: ISO date string (YYYY-MM-DD).
        config_path: Path to matrix candidate YAML (e.g. matrix_a_lgb.yaml).
        model_dir: Directory containing trained models. When None, defaults
            to PROJECT_ROOT / "matrix_results".
        provider_uri: qlib data directory override.

    Returns:
        Path to the written parquet file.

    Raises:
        ValueError: No walk-forward window covers trade_date, or no rows
            survive after inference.
        FileNotFoundError: The resolved model file does not exist.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import qlib  # noqa: PLC0415
    from qlib.config import C as _QlibC  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.contrib.data.handler import Alpha158, Alpha360  # noqa: PLC0415

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415
    from ashare_lab.research.smoke_test import get_all_windows  # noqa: PLC0415
    from ashare_lab.research.train import ALPHA158_WARMUP_START  # noqa: PLC0415

    import ashare_lab.config as cfg  # noqa: PLC0415

    # -- 1. Config merge: baseline + candidate model section ------------------
    # Not reentrant: mutates global cfg.CONFIG_PATH (safe under subprocess isolation).
    tmp_path: Path | None = None
    original_config_path = cfg.CONFIG_PATH
    try:
        with open(cfg.CONFIG_PATH) as f:
            base = yaml.safe_load(f)
        with open(config_path) as f:
            candidate = yaml.safe_load(f)

        base["model"] = candidate["model"]
        base["matrix"] = candidate.get("matrix", {})

        fd, tmp_str = tempfile.mkstemp(suffix=".yaml", prefix="shadow_")
        tmp_path = Path(tmp_str)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(base, f)

        cfg.CONFIG_PATH = tmp_path
        cfg.load_config.cache_clear()
        merged = cfg.load_config()

        # -- 2. Extract model metadata ---------------------------------------
        model_cfg = merged["model"]
        tag: str = model_cfg.get("tag", base.get("matrix", {}).get("tag", "unknown"))
        model_type: str = model_cfg.get("type", "lgbm").lower()
        handler_name: str = model_cfg.get("handler", "alpha158").lower()
        universe: str = merged.get("universe", {}).get("primary", "csi1000")

        if model_dir is None:
            model_dir = cfg.PROJECT_ROOT / "matrix_results"

        # -- 3. Find walk-forward window covering trade_date ------------------
        windows = get_all_windows()
        window: dict | None = None
        for w in windows:
            if w["test_start"] <= trade_date <= w["test_end"]:
                window = w
                break

        if window is None:
            if windows and trade_date > windows[-1]["test_end"]:
                window = windows[-1]
            else:
                raise ValueError(
                    "no walk-forward window covers %s "
                    "(earliest test_start=%s)"
                    % (trade_date, windows[0]["test_start"] if windows else "N/A")
                )

        window_id: int = window["window_id"]

        # -- 4. Resolve model path --------------------------------------------
        ext = ".pkl" if model_type in ("lgbm", "densemble") else ".pt"
        model_path = model_dir / tag / "models" / f"w{window_id}{ext}"

        if not model_path.exists():
            raise FileNotFoundError(
                "model file %s does not exist "
                "(train the candidate first via matrix_runner)"
                % model_path
            )

        log.info(
            "shadow_predict %s: tag=%s type=%s window=W%d model=%s",
            trade_date, tag, model_type, window_id, model_path.name,
        )

        # -- 5. Init qlib (idempotent) ----------------------------------------
        uri = provider_uri or str(DEFAULT_PROVIDER_URI)
        if not getattr(_QlibC, "registered", False):
            qlib.init(provider_uri=uri, region=REG_CN)

        # -- 6. Build handler -------------------------------------------------
        HandlerClass = Alpha360 if handler_name == "alpha360" else Alpha158  # noqa: N806

        if model_type in ("alstm", "tra"):
            learn_procs = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        else:
            learn_procs = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ]

        handler = HandlerClass(
            instruments=universe,
            start_time=ALPHA158_WARMUP_START,
            end_time=trade_date,
            fit_start_time=window["train_start"],
            fit_end_time=window["train_end"],
            learn_processors=learn_procs,
        )

        # -- 7. Build dataset + load model + predict --------------------------
        segs = {"test": (trade_date, trade_date)}

        if model_type == "tra":
            from qlib.contrib.data.dataset import MTSDatasetH  # noqa: PLC0415

            step_len = int(model_cfg.get("step_len", 20))
            num_states = int(model_cfg.get("routing", {}).get("num_states", 3))
            dataset = MTSDatasetH(
                handler=handler, segments=segs, seq_len=step_len,
                num_states=num_states, memory_mode="sample",
                batch_size=-1, shuffle=True,
            )
        elif model_type == "alstm":
            from qlib.data.dataset import TSDatasetH  # noqa: PLC0415

            step_len = int(model_cfg.get("step_len", 20))
            dataset = TSDatasetH(handler=handler, segments=segs, step_len=step_len)
        else:
            from qlib.data.dataset import DatasetH  # noqa: PLC0415

            dataset = DatasetH(handler=handler, segments=segs)

        # Load persisted model
        if ext == ".pt":
            import torch  # noqa: PLC0415

            model = torch.load(str(model_path), weights_only=False)
            # TRA writer shim (same as predict.py)
            if hasattr(model, "_writer"):
                model._writer = None
        else:
            import pickle  # noqa: PLC0415

            with open(model_path, "rb") as mf:
                model = pickle.load(mf)  # noqa: S301

        pred_raw = model.predict(dataset, segment="test")

        # TRAModel.predict returns DataFrame; extract score column
        if hasattr(pred_raw, "columns") and "score" in pred_raw.columns:
            pred = pred_raw["score"]
        elif hasattr(pred_raw, "name"):
            pred = pred_raw
        else:
            pred = pred_raw.iloc[:, 0] if hasattr(pred_raw, "iloc") else pred_raw

        # -- 8. Extract trade_date rows, drop non-finite ----------------------
        ts = pd.Timestamp(trade_date)
        if ts in pred.index.get_level_values(0):
            day = pred.loc[ts]
        else:
            log.warning(
                "timestamp %s not in prediction index; using full output "
                "(%d rows, segment was single-day so this is expected)",
                trade_date, len(pred),
            )
            day = pred

        day = day[np.isfinite(day.to_numpy(dtype=float))]

        if day.empty:
            raise ValueError(
                "all scores for %s are non-finite after filtering" % trade_date
            )

        # -- 9. Build output DataFrame ----------------------------------------
        df = pd.DataFrame({
            "instrument": day.index.astype(str),
            "score": day.to_numpy(dtype=float),
        })

        # -- 10. Write parquet + meta sidecar ---------------------------------
        out_dir = cfg.PROJECT_ROOT / "shadow_predictions" / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        out = out_dir / f"{trade_date}.parquet"

        # Isolation guard: shadow output must never land in production dir.
        # Use resolved paths with trailing separator to prevent prefix collision
        # (e.g. "predictions_v2" matching "predictions").
        prod_prefix = str(cfg.PREDICTIONS_DIR.resolve()) + os.sep
        if str(out.resolve()).startswith(prod_prefix):
            raise ValueError(
                "shadow output %s must not be inside production predictions dir %s"
                % (out, cfg.PREDICTIONS_DIR)
            )

        df.to_parquet(out, index=False)

        meta = {
            "model": model_path.name,
            "tag": tag,
            "type": model_type,
            "universe": universe,
            "window_id": window_id,
            "produced_at": datetime.now(timezone.utc).isoformat(),
            "n_instruments": len(df),
        }
        meta_path = out_dir / f"{trade_date}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info(
            "shadow_predict %s: wrote %d instruments -> %s",
            trade_date, len(df), out,
        )

        return out

    finally:
        # Restore original config no matter what
        cfg.CONFIG_PATH = original_config_path
        cfg.load_config.cache_clear()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Shadow predict for matrix candidate")
    parser.add_argument("--date", required=True, help="trade date YYYY-MM-DD")
    parser.add_argument("--config", required=True, help="path to matrix candidate YAML")
    parser.add_argument("--model-dir", default=None, help="model directory override")
    parser.add_argument("--provider-uri", default=None, help="qlib data directory")
    args = parser.parse_args()

    shadow_predict_for_date(
        args.date,
        args.config,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        provider_uri=args.provider_uri,
    )
