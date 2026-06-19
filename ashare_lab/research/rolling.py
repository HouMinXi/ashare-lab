"""Full walk-forward orchestration: train -> predict -> backtest -> metrics.

Provides run_full_walk_forward(), which loops over all windows returned by
get_all_windows(), trains/backtests each, and returns a list of WindowResult
dicts for the gate verdict in plan 02-03.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ashare_lab.config import load_config
from ashare_lab.research.backtest import run_backtest
from ashare_lab.research.metrics import aggregate_window_metrics, daily_rank_ic
from ashare_lab.research.smoke_test import get_all_windows
from ashare_lab.research.train import train_window

log = logging.getLogger(__name__)


def run_full_walk_forward(
    exp_dir: Path,
    n_drop: int,
    universe: str,
    pred_dir: Path | None = None,
) -> tuple[list[dict], list[int]]:
    """Run the full 6-window expanding walk-forward loop.

    For each window returned by get_all_windows(): trains an LGBModel,
    generates predictions, applies price filter, runs backtest, computes
    RankIC and excess return metrics, and persists predictions.

    Per-window errors are caught and logged with full tracebacks (window
    isolation). However, if zero windows complete or fewer than min_windows
    succeed, the function raises RuntimeError to prevent silent empty
    verdicts.

    All windows from get_all_windows() are evaluated regardless of
    is_complete (incomplete windows count in the denominator with no
    special treatment). is_complete=False windows are logged for visibility.

    Args:
        exp_dir: Experiment output directory. Models saved under
            exp_dir/models/w{N}.pkl. Used as default pred_dir parent.
        n_drop: Number of positions dropped per rebalance (strategy param).
        universe: Qlib universe string, e.g. "csi500" or "csi300".
        pred_dir: Directory for persisted prediction parquet files. Defaults
            to exp_dir/predictions if None.

    Returns:
        Tuple of (window_results, failed_windows).
        window_results: list of WindowResult dicts, each with keys:
            window_id, mean_rank_ic, cumulative_excess_return,
            is_positive_excess, lot_skip_count, universe, n_drop,
            train_end, test_start, test_end, pred_path.
        failed_windows: list of int window IDs that errored during
            training/backtest. Empty list when all windows succeed.

    Raises:
        RuntimeError: When zero windows complete or fewer than min_windows
            succeed. Prevents silent empty/partial verdicts.
    """
    cfg = load_config()
    min_windows: int = cfg["walk_forward"]["min_windows"]

    # Resolve pred_dir.
    if pred_dir is None:
        pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    windows = get_all_windows()
    total = len(windows)
    log.info(
        "walk-forward: %d windows total, min_windows=%d, universe=%s, n_drop=%d",
        total,
        min_windows,
        universe,
        n_drop,
    )

    if total < min_windows:
        log.warning(
            "got %d windows, expected >= %d",
            total,
            min_windows,
        )

    window_results: list[dict] = []
    failed_windows: list[int] = []

    for window in windows:
        window_id: int = window["window_id"]

        if not window["is_complete"]:
            log.warning(
                "W%d: is_complete=False (test span < 170 days); included in denominator",
                window_id,
            )

        try:
            # Step 1: train, predict, price-filter.
            _model_path, pred, label = train_window(window, exp_dir, universe)

            # Step 2: per-date Spearman IC.
            rank_ic_series = daily_rank_ic(pred, label)

            # Step 3: backtest with configured cost model.
            portfolio_df, bench_close, lot_skip_count = run_backtest(
                window=window,
                pred=pred,
                n_drop=n_drop,
                slippage_override=None,
            )

            # Step 4: aggregate metrics.
            metrics = aggregate_window_metrics(
                rank_ic_series=rank_ic_series,
                portfolio_df=portfolio_df,
                bench_close=bench_close,
                window_id=window_id,
                lot_skip_count=lot_skip_count,
            )

            # Step 5: persist predictions as parquet (PredictionFile schema).
            pred_path = pred_dir / f"pred_w{window_id}.parquet"
            pred.to_frame("score").to_parquet(pred_path)
            log.info("W%d: predictions persisted -> %s", window_id, pred_path)

            # Step 6: complete WindowResult by merging window metadata.
            result: dict = {
                **metrics,
                "universe": universe,
                "n_drop": n_drop,
                "is_complete": window["is_complete"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "pred_path": str(pred_path),
            }

            window_results.append(result)
            log.info(
                "W%d: mean_rank_ic=%.6f cumulative_excess=%.4f positive=%s",
                window_id,
                result["mean_rank_ic"] if result["mean_rank_ic"] is not None else float("nan"),
                result["cumulative_excess_return"],
                result["is_positive_excess"],
            )

        except Exception:
            log.exception("W%d: FAILED during training/backtest", window_id)
            failed_windows.append(window_id)

    completed = len(window_results)

    if failed_windows:
        log.error(
            "walk-forward: %d/%d windows FAILED: %s",
            len(failed_windows),
            total,
            failed_windows,
        )

    if completed == 0 and total > 0:
        raise RuntimeError(
            f"walk-forward: 0/{total} windows completed, all failed. "
            f"Failed windows: {failed_windows}. Cannot produce a verdict."
        )

    if completed < min_windows:
        raise RuntimeError(
            f"walk-forward: only {completed}/{total} windows completed "
            f"(min_windows={min_windows}). "
            f"Failed windows: {failed_windows}. Refusing to produce a verdict "
            f"on insufficient data."
        )

    log.info(
        "walk-forward complete: %d/%d windows succeeded",
        completed,
        total,
    )

    return window_results, failed_windows
