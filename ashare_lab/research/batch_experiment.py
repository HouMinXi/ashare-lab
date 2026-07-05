"""Orchestrator for model matrix experiments.

Runs on X500 (Linux), SSHes to GPU (admin@192.168.100.11) per-cell.
Each cell = one candidate config x one walk-forward window, executed
via matrix_runner.py in a subprocess on the GPU host.

Resumes from JSONL checkpoint.  GPU access serialized via flock.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

from ashare_lab.research.metrics import CELL_SCHEMA_KEYS

log = logging.getLogger(__name__)

_DEFAULT_GPU_HOST = "admin@192.168.100.11"
_DEFAULT_GPU_REPO = r"H:\ashare-lab"
_GPU_LOCK_PATH = "/tmp/ashare-gpu.lock"


def discover_configs(configs_dir: str) -> list[Path]:
    """Glob matrix_*.yaml, return sorted list of paths."""
    return sorted(Path(configs_dir).glob("matrix_*.yaml"))


def load_completed(jsonl_path: str) -> set[tuple[str, int]]:
    """Parse JSONL, return set of (model, window) tuples already done."""
    p = Path(jsonl_path)
    if not p.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for line in p.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping corrupt JSONL line: %s", line[:80])
            continue
        completed.add((rec["model"], rec["window"]))
    return completed


def append_result(jsonl_path: str, record: dict) -> None:
    """Append one JSON record to JSONL file.  Validates schema keys."""
    missing = [k for k in CELL_SCHEMA_KEYS if k not in record]
    if missing:
        raise ValueError(f"record missing keys: {missing}")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def commit_summary_csv(output_dir: str, project_root: str) -> None:
    """Read JSONL, write CSV to project_root/matrix_summary.csv, git commit."""
    jsonl_path = Path(output_dir) / "results.jsonl"
    if not jsonl_path.exists():
        log.warning("no results.jsonl found; skipping CSV commit")
        return

    records: list[dict] = []
    for line in jsonl_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping corrupt JSONL line in CSV export: %s", line[:80])

    if not records:
        return

    csv_path = Path(project_root) / "matrix_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_SCHEMA_KEYS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in CELL_SCHEMA_KEYS})

    subprocess.run(
        ["git", "add", "matrix_summary.csv"],
        cwd=project_root, check=True, timeout=60,
    )
    result = subprocess.run(
        ["git", "commit", "-m", "research: update matrix summary CSV"],
        cwd=project_root, check=False, timeout=60,
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        log.error("git commit failed: %s", result.stderr[:200])


def scp_config_to_gpu(
    config_path: str, gpu_host: str, gpu_repo: str,
) -> None:
    """SCP a candidate YAML to GPU configs directory."""
    name = Path(config_path).name
    # Use relative path inside gpu_repo.
    remote = f"{gpu_host}:{gpu_repo}/configs/{name}"
    subprocess.run(["scp", config_path, remote], check=True, timeout=30)


def scp_pred_from_gpu(
    gpu_host: str,
    gpu_repo: str,
    remote_subdir: str,
    local_dir: str,
    filename: str,
) -> None:
    """SCP one prediction/label parquet file from GPU to local."""
    remote = f"{gpu_host}:{gpu_repo}/{remote_subdir}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", remote, local_dir], check=True, timeout=60)


def scp_results(
    output_dir: str, model_tag: str, gpu_host: str, gpu_repo: str,
) -> bool:
    """SCP -r results directory from GPU.  Returns True on success."""
    remote = f"{gpu_host}:{gpu_repo}/matrix_results/{model_tag}"
    local = Path(output_dir) / model_tag
    local.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["scp", "-r", remote, str(local.parent)],
            check=False, timeout=300,
        )
        if result.returncode != 0:
            log.warning(
                "scp_results failed for %s (rc=%d). Retry: scp -r %s %s",
                model_tag, result.returncode, remote, local.parent,
            )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning(
            "scp_results failed for %s: %s. Retry: scp -r %s %s",
            model_tag, exc, remote, local.parent,
        )
        return False


def _kill_remote_python(gpu_host: str) -> None:
    """Kill all python.exe on GPU to clean orphaned processes."""
    try:
        subprocess.run(
            f'ssh -o ConnectTimeout=5 {gpu_host} "taskkill /f /im python.exe"',
            shell=True, capture_output=True, timeout=15,
        )
        log.info("killed remote python processes on %s", gpu_host)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("remote cleanup failed: %s", exc)


def _recover_result_from_gpu(
    gpu_host: str,
    gpu_repo: str,
    tag: str,
    wid: int,
    local_dir: str,
    seed: int | None = None,
) -> dict | None:
    """Try to SCP result JSON from GPU when stdout was lost.

    Returns parsed record dict, or None if file does not exist on GPU.
    """
    result_name = (
        f"result_w{wid}_seed{seed}.json"
        if seed is not None
        else f"result_w{wid}.json"
    )
    remote = f"{gpu_host}:{gpu_repo}/matrix_results/{tag}/{result_name}"
    local_path = Path(local_dir) / tag / result_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cp = subprocess.run(
            ["scp", remote, str(local_path)],
            check=False, capture_output=True, timeout=15,
        )
        if cp.returncode != 0:
            return None
        return json.loads(local_path.read_text(encoding="utf-8"))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.warning("result recovery failed for %s w%d: %s", tag, wid, exc)
        return None


@contextmanager
def _gpu_lock():
    """Serialize GPU access via flock."""
    fd = open(_GPU_LOCK_PATH, "a")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def run_matrix(
    configs_dir: str,
    output_dir: str,
    smoke: bool = False,
    n_epochs: int | None = None,
    windows: list[int] | None = None,
    gpu_host: str = _DEFAULT_GPU_HOST,
    gpu_repo: str = _DEFAULT_GPU_REPO,
    project_root: str | None = None,
) -> None:
    """Main matrix experiment loop.

    Args:
        configs_dir: Directory with matrix_*.yaml files.
        output_dir: Local output directory for results.
        smoke: If True, run first config x first window only.
        n_epochs: Epoch override for quick smoke runs.
        windows: List of 1-based window ids.  None = all from config.
        gpu_host: SSH target for GPU machine.
        gpu_repo: Repo path on GPU.
        project_root: Git repo root for CSV commit.  None = skip commit.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = str(Path(output_dir) / "results.jsonl")

    # SCP baseline.yaml to GPU at start.
    from ashare_lab.config import CONFIG_PATH  # noqa: PLC0415
    scp_config_to_gpu(str(CONFIG_PATH), gpu_host, gpu_repo)

    configs = discover_configs(configs_dir)
    if not configs:
        log.warning("no matrix_*.yaml found in %s", configs_dir)
        return

    if windows is None:
        from ashare_lab.research.smoke_test import get_all_windows  # noqa: PLC0415
        all_wins = get_all_windows()
        windows = [w["window_id"] for w in all_wins]

    if smoke:
        configs = configs[:1]
        windows = windows[:1]

    completed = load_completed(jsonl_path)

    for config_path in configs:
        with open(config_path) as f:
            candidate = yaml.safe_load(f)

        tag: str = candidate["matrix"]["tag"]
        model_type = candidate["model"].get("type", "lgbm").lower()
        seeds: list[int] | None = candidate.get("seeds")
        config_name = config_path.name

        # SCP candidate config to GPU.
        scp_config_to_gpu(str(config_path), gpu_host, gpu_repo)

        for wid in windows:
            if seeds:
                # Multi-seed: run each seed, then average.
                seed_preds_collected = []
                for seed in seeds:
                    seed_tag = f"{tag}_seed{seed}"
                    if (seed_tag, wid) in completed:
                        log.info("skip completed: %s w%d", seed_tag, wid)
                        seed_preds_collected.append(seed)
                        continue

                    smoke_epochs = n_epochs
                    if smoke and smoke_epochs is None:
                        smoke_epochs = 3 if model_type == "densemble" else 5

                    cmd = (
                        f"ssh -o ConnectTimeout=10 {gpu_host} "
                        f'"cd /d {gpu_repo} && python -m ashare_lab.research.matrix_runner '
                        f"--config configs/{config_name} "
                        f"--window {wid} "
                        f"--output-dir matrix_results/{tag} "
                        f"--seed {seed}"
                    )
                    if smoke_epochs is not None:
                        cmd += f" --n-epochs {smoke_epochs}"
                    cmd += '"'

                    with _gpu_lock():
                        try:
                            result = subprocess.run(
                                cmd, shell=True, capture_output=True,
                                encoding="utf-8", errors="replace",
                                timeout=14400,
                            )
                        except subprocess.TimeoutExpired:
                            log.error(
                                "TIMEOUT %s w%d seed%d after 4h", tag, wid, seed,
                            )
                            _kill_remote_python(gpu_host)
                            continue

                    record = None
                    if result.returncode == 0:
                        stdout_lines = result.stdout.strip().split("\n")
                        try:
                            record = json.loads(stdout_lines[-1])
                        except json.JSONDecodeError:
                            pass

                    if record is None:
                        record = _recover_result_from_gpu(
                            gpu_host, gpu_repo, tag, wid, output_dir,
                            seed=seed,
                        )

                    if record is None:
                        log.error(
                            "FAIL %s w%d seed%d: no result from stdout "
                            "or GPU file. stderr: %s",
                            tag, wid, seed,
                            (result.stderr or "")[-500:],
                        )
                        _kill_remote_python(gpu_host)
                        continue
                    record["model"] = seed_tag

                    # SCP artifacts BEFORE marking complete (W2-F2: a cell
                    # is complete only when its parquets are on X500).
                    pred_name = f"pred_w{wid}_seed{seed}.parquet"
                    scp_pred_from_gpu(
                        gpu_host, gpu_repo,
                        f"matrix_results/{tag}", str(Path(output_dir) / tag),
                        pred_name,
                    )
                    if seed == seeds[0]:
                        scp_pred_from_gpu(
                            gpu_host, gpu_repo,
                            f"matrix_results/{tag}",
                            str(Path(output_dir) / tag),
                            f"labels_w{wid}.parquet",
                        )

                    append_result(jsonl_path, record)
                    completed.add((seed_tag, wid))
                    seed_preds_collected.append(seed)

                # Average predictions across seeds if all completed.
                if len(seed_preds_collected) == len(seeds):
                    _average_seeds(
                        output_dir, tag, wid, seeds, jsonl_path, completed,
                    )

            else:
                # Single-seed path.
                if (tag, wid) in completed:
                    log.info("skip completed: %s w%d", tag, wid)
                    continue

                smoke_epochs = n_epochs
                if smoke and smoke_epochs is None:
                    smoke_epochs = 3 if model_type == "densemble" else 5

                cmd = (
                    f"ssh -o ConnectTimeout=10 {gpu_host} "
                    f'"cd /d {gpu_repo} && python -m ashare_lab.research.matrix_runner '
                    f"--config configs/{config_name} "
                    f"--window {wid} "
                    f"--output-dir matrix_results/{tag}"
                )
                if smoke_epochs is not None:
                    cmd += f" --n-epochs {smoke_epochs}"
                cmd += '"'

                with _gpu_lock():
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True,
                            encoding="utf-8", errors="replace",
                            timeout=14400,
                        )
                    except subprocess.TimeoutExpired:
                        log.error("TIMEOUT %s w%d after 4h", tag, wid)
                        _kill_remote_python(gpu_host)
                        continue

                record = None
                if result.returncode == 0:
                    stdout_lines = result.stdout.strip().split("\n")
                    try:
                        record = json.loads(stdout_lines[-1])
                    except json.JSONDecodeError:
                        pass

                if record is None:
                    # stdout lost or SSH failed; try GPU-side result file.
                    record = _recover_result_from_gpu(
                        gpu_host, gpu_repo, tag, wid, output_dir,
                    )

                if record is None:
                    log.error(
                        "FAIL %s w%d: no result from stdout or GPU file. "
                        "stderr: %s", tag, wid,
                        (result.stderr or "")[-500:],
                    )
                    _kill_remote_python(gpu_host)
                    continue
                # SCP artifacts BEFORE marking complete (W2-F2).
                scp_pred_from_gpu(
                    gpu_host, gpu_repo,
                    f"matrix_results/{tag}", str(Path(output_dir) / tag),
                    f"pred_w{wid}.parquet",
                )
                scp_pred_from_gpu(
                    gpu_host, gpu_repo,
                    f"matrix_results/{tag}", str(Path(output_dir) / tag),
                    f"labels_w{wid}.parquet",
                )

                append_result(jsonl_path, record)
                completed.add((tag, wid))

        # SCP full results dir after all windows.
        scp_ok = scp_results(output_dir, tag, gpu_host, gpu_repo)
        if scp_ok and project_root:
            commit_summary_csv(output_dir, project_root)


def _average_seeds(
    output_dir: str,
    tag: str,
    wid: int,
    seeds: list[int],
    jsonl_path: str,
    completed: set[tuple[str, int]],
) -> None:
    """Average predictions across seeds, recompute metrics, write _avg record."""
    import pandas as pd  # noqa: PLC0415

    avg_tag = f"{tag}_avg"
    if (avg_tag, wid) in completed:
        return

    tag_dir = Path(output_dir) / tag
    frames = []
    for seed in seeds:
        pf = tag_dir / f"pred_w{wid}_seed{seed}.parquet"
        if pf.exists():
            frames.append(pd.read_parquet(pf)["score"])

    if len(frames) != len(seeds):
        log.warning("incomplete seeds for %s w%d; skipping average", tag, wid)
        return

    # Inner-join: only instruments present in ALL seeds are averaged.
    aligned = pd.concat(frames, axis=1, join="inner")
    avg_pred = aligned.mean(axis=1)
    avg_pred.name = "score"

    # Save averaged predictions for analyze_matrix.py consumption.
    avg_pred.to_frame("score").to_parquet(tag_dir / f"pred_w{wid}.parquet")

    # Load labels.
    labels_path = tag_dir / f"labels_w{wid}.parquet"
    if not labels_path.exists():
        log.warning("no labels for %s w%d; skipping average", tag, wid)
        return
    label = pd.read_parquet(labels_path)["label"]

    from ashare_lab.research.metrics import (  # noqa: PLC0415
        daily_rank_ic,
        aggregate_window_metrics,
        compute_max_drawdown,
    )
    from ashare_lab.research.smoke_test import get_window  # noqa: PLC0415
    from ashare_lab.research.backtest import run_backtest  # noqa: PLC0415
    from ashare_lab.config import load_config  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    window = get_window(wid - 1)
    cfg = load_config()
    n_drop = cfg.get("strategy", {}).get("main", {}).get("n_drop", 1)

    portfolio_df, bench_close, lot_skip_count = run_backtest(
        window, avg_pred, n_drop, slippage_override=None,
    )
    rank_ic_series = daily_rank_ic(avg_pred, label)
    metrics = aggregate_window_metrics(
        rank_ic_series, portfolio_df, bench_close, wid, lot_skip_count,
    )
    max_drawdown_value = compute_max_drawdown(portfolio_df)

    record = {
        "model": avg_tag,
        "window": wid,
        "ic": metrics["mean_rank_ic"],
        "excess": metrics["cumulative_excess_return"],
        "maxdd": max_drawdown_value,
        "completed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    append_result(jsonl_path, record)
    completed.add((avg_tag, wid))


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Run model matrix experiments on GPU via SSH",
    )
    parser.add_argument(
        "--configs-dir", default="configs",
        help="Directory with matrix_*.yaml files",
    )
    parser.add_argument(
        "--output-dir", default="matrix_results",
        help="Local output directory",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--n-epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--windows", default=None, help="Comma-separated window ids")
    parser.add_argument(
        "--gpu-host", default=_DEFAULT_GPU_HOST, help="GPU SSH target",
    )
    parser.add_argument(
        "--gpu-repo", default=_DEFAULT_GPU_REPO, help="Repo path on GPU",
    )
    args = parser.parse_args()

    win_list = (
        [int(x) for x in args.windows.split(",")]
        if args.windows else None
    )

    from ashare_lab.config import PROJECT_ROOT  # noqa: PLC0415

    run_matrix(
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
        smoke=args.smoke,
        n_epochs=args.n_epochs,
        windows=win_list,
        gpu_host=args.gpu_host,
        gpu_repo=args.gpu_repo,
        project_root=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
