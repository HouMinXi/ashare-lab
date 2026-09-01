"""Single-cell subprocess target for model matrix experiments.

Trains one model on one walk-forward window, runs backtest, computes
metrics, saves predictions/labels, prints a JSON record to stdout.

Designed to run inside an SSH subprocess on the GPU host.  All qlib
imports are deferred inside function bodies.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ashare_lab import config as cfg
from ashare_lab.research.metrics import CELL_SCHEMA_KEYS, is_valid_turnover

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def merge_candidate_config(base: dict, candidate: dict) -> dict:
    """Overlay a candidate config onto the baseline config dict.

    The model section is always replaced. An optional walk_forward map is
    merged key-wise so candidates can sweep window lengths without copying
    the whole section. base is not mutated; the merged copy is returned.
    """
    result = copy.deepcopy(base)
    result["model"] = copy.deepcopy(candidate["model"])
    wf_override = candidate.get("walk_forward")
    if wf_override is not None:
        merged_wf = {**(base.get("walk_forward") or {}), **wf_override}
        result["walk_forward"] = copy.deepcopy(merged_wf)
    return result


def extract_mean_turnover(portfolio_df: "pd.DataFrame") -> float | None:
    """Mean of the qlib portfolio-metrics ``turnover`` column.

    Returns None when the frame is empty, the column is absent, or the
    mean is NaN (mirrors the mean_rank_ic NaN guard in metrics.py).
    """
    if portfolio_df.empty or "turnover" not in portfolio_df.columns:
        return None
    values = [
        float(t) for t in portfolio_df["turnover"] if is_valid_turnover(t)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def run_cell(
    config_path: str,
    window_id: int,
    output_dir: str,
    seed: int | None = None,
    n_epochs: int | None = None,
) -> dict:
    """Train one model on one window, backtest, return metrics record.

    Args:
        config_path: Path to candidate YAML (model + matrix sections).
        window_id: 1-based walk-forward window id.
        output_dir: Directory for predictions/labels parquet output.
        seed: Random seed override.  None = use model default.
        n_epochs: Epoch count override.  None = use config value.

    Returns:
        Dict with CELL_SCHEMA_KEYS (model, window, ic, excess, maxdd,
        completed_at) plus an optional ``turnover`` key holding the mean
        of the backtest turnover column (omitted when unavailable).
    """
    import numpy as np  # noqa: PLC0415

    with open(config_path) as f:
        candidate = yaml.safe_load(f)

    tag: str = candidate["matrix"]["tag"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Merge candidate model section into baseline config.
    # Load baseline, overlay model, write to temp file, point cfg at it.
    # Not reentrant: mutates global cfg.CONFIG_PATH (safe under subprocess isolation).
    original_config_path = cfg.CONFIG_PATH
    tmp_cfg = None
    try:
        with original_config_path.open() as f:
            base = yaml.safe_load(f)

        base = merge_candidate_config(base, candidate)
        tmp_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        )
        yaml.safe_dump(base, tmp_cfg)
        tmp_cfg.close()

        cfg.CONFIG_PATH = Path(tmp_cfg.name)
        cfg.load_config.cache_clear()

        merged = cfg.load_config()
        model_type = merged["model"].get("type", "lgbm").lower()

        # Epoch override per model type.
        if n_epochs is not None:
            if model_type in ("alstm", "tra"):
                merged["model"]["n_epochs"] = n_epochs
            elif model_type == "lgbm":
                merged["model"]["num_boost_round"] = n_epochs
            elif model_type == "densemble":
                merged["model"]["epochs"] = n_epochs
            # Re-write temp config so load_config picks up the override.
            cfg.load_config.cache_clear()
            with open(tmp_cfg.name, "w") as f:
                yaml.safe_dump(merged, f)
            cfg.load_config.cache_clear()

        # Seed.
        if seed is not None:
            np.random.seed(seed)
            try:
                import torch  # noqa: PLC0415
                torch.manual_seed(seed)
            except ImportError:
                pass

        from ashare_lab.research.smoke_test import get_window  # noqa: PLC0415
        from ashare_lab.research.train import train_window  # noqa: PLC0415
        from ashare_lab.research.backtest import run_backtest  # noqa: PLC0415
        from ashare_lab.research.metrics import (  # noqa: PLC0415
            daily_rank_ic,
            aggregate_window_metrics,
            compute_max_drawdown,
        )

        window = get_window(window_id - 1)
        universe = merged.get("universe", {}).get("primary", "csi1000")

        model_path, pred, label = train_window(
            window, out, universe, seed=seed,
        )

        # Save labels (identical across seeds -- skip if exists).
        labels_path = out / f"labels_w{window_id}.parquet"
        if not labels_path.exists():
            import pandas as pd  # noqa: PLC0415
            pd.DataFrame({"label": label}).to_parquet(labels_path)

        # Save predictions.
        import pandas as pd  # noqa: PLC0415
        pred_name = (
            f"pred_w{window_id}_seed{seed}.parquet"
            if seed is not None
            else f"pred_w{window_id}.parquet"
        )
        pd.DataFrame({"score": pred}).to_parquet(out / pred_name)

        # Backtest.
        n_drop = merged.get("strategy", {}).get("main", {}).get("n_drop", 1)
        portfolio_df, bench_close, lot_skip_count = run_backtest(
            window, pred, n_drop, slippage_override=None,
        )

        # Metrics.
        rank_ic_series = daily_rank_ic(pred, label)
        metrics = aggregate_window_metrics(
            rank_ic_series, portfolio_df, bench_close,
            window_id, lot_skip_count,
        )
        max_drawdown_value = compute_max_drawdown(portfolio_df)

        # Extra record key, not part of CELL_SCHEMA_KEYS: old results.jsonl
        # rows without it stay valid.
        turnover = extract_mean_turnover(portfolio_df)

        record = {
            "model": tag,
            "window": window_id,
            "ic": metrics["mean_rank_ic"],
            "excess": metrics["cumulative_excess_return"],
            "maxdd": max_drawdown_value,
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        # Optional key: omitted entirely when turnover data is unavailable,
        # matching pre-sweep jsonl rows.
        if turnover is not None:
            record["turnover"] = turnover

        # Validate record.
        missing = [k for k in CELL_SCHEMA_KEYS if k not in record]
        if missing:
            raise ValueError(f"record missing keys: {missing}")

        # Persist result to GPU-local file so batch_experiment can recover
        # it via SCP if the SSH stdout pipe breaks.
        result_name = (
            f"result_w{window_id}_seed{seed}.json"
            if seed is not None
            else f"result_w{window_id}.json"
        )
        result_path = out / result_name
        result_path.write_text(json.dumps(record), encoding="utf-8")
        log.info("result written: %s", result_path)

        return record

    finally:
        # Restore original config path.
        cfg.CONFIG_PATH = original_config_path
        cfg.load_config.cache_clear()
        if tmp_cfg is not None:
            Path(tmp_cfg.name).unlink(missing_ok=True)


def main() -> None:
    """CLI entry point.  Prints JSON record as last stdout line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Train one model on one walk-forward window",
    )
    parser.add_argument("--config", required=True, help="Candidate YAML path")
    parser.add_argument("--window", type=int, required=True, help="1-based window id")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--n-epochs", type=int, default=None, help="Epoch override")
    args = parser.parse_args()

    record = run_cell(
        args.config, args.window, args.output_dir,
        seed=args.seed, n_epochs=args.n_epochs,
    )
    # Last line of stdout = JSON record for batch_experiment.py to parse.
    print(json.dumps(record))


if __name__ == "__main__":
    main()
