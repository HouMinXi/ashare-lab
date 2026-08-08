"""Signal quality monitors: PSI distribution drift + Lagged IC decay.

Alert-only monitors appended after _step15_graduation in run_daily.
No mutation of orders, weights, or predictions.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ashare_lab.bridge import send_bridge_alert
from ashare_lab.config import PREDICTIONS_DIR, PROJECT_ROOT
from ashare_lab.data.calendar import previous_trading_day

if TYPE_CHECKING:
    from ashare_lab.paper.pipeline import DailyRunContext

log = logging.getLogger(__name__)

# Frozen thresholds (10-CONTEXT.md)
_PSI_THRESHOLD = 0.25
_IC_THRESHOLD = 0.01
_REFERENCE_DAYS = 60
_N_BUCKETS = 10
_ROLLING_WINDOW = 20
_MIN_REFERENCE_FILES = 20
_MIN_INSTRUMENTS = 50


# ---------------------------------------------------------------------------
# Helper: atomic sidecar writes
# ---------------------------------------------------------------------------
def _write_sidecar(path: Path, data: dict) -> None:
    """Atomic read-modify-write of a .meta.json sidecar file."""
    def _to_native(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(data)
    dir_path = path.parent
    tmp = None
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2, default=_to_native)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except (OSError, TypeError) as exc:
        log.error("_write_sidecar failed for %s: %s", path, exc)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 10-01: PSI Monitor
# ---------------------------------------------------------------------------
def compute_psi(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    n_buckets: int = _N_BUCKETS,
) -> float | None:
    """Population Stability Index for distribution drift detection.

    Returns None on empty/degenerate input.
    """
    ref = np.asarray(reference_scores, dtype=np.float64)
    cur = np.asarray(current_scores, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return None
    ref_min, ref_max = float(ref.min()), float(ref.max())
    if ref_min == ref_max:
        return None
    ref_bins = np.linspace(ref_min, ref_max, n_buckets + 1)
    ref_hist, _ = np.histogram(ref, bins=ref_bins)
    cur_clipped = np.clip(cur, a_min=ref_bins[0], a_max=ref_bins[-1])
    cur_hist, _ = np.histogram(cur_clipped, bins=ref_bins)
    ref_prop = ref_hist.astype(np.float64) / len(ref)
    cur_prop = cur_hist.astype(np.float64) / len(cur)
    ref_prop = np.maximum(ref_prop, 1e-4)
    cur_prop = np.maximum(cur_prop, 1e-4)
    ref_prop /= ref_prop.sum()
    cur_prop /= cur_prop.sum()
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return psi


def load_reference_scores(
    predictions_dir: Path,
    end_date: str,
    n_days: int = _REFERENCE_DAYS,
) -> tuple[np.ndarray | None, int]:
    """Load historical scores from parquet files for PSI reference.

    Returns (None, count) if fewer than _MIN_REFERENCE_FILES available.
    """
    files = []
    for f in predictions_dir.glob("*.parquet"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if m and m.group(1) < end_date:
            files.append((m.group(1), f))
    files.sort(key=lambda x: x[0], reverse=True)
    files = files[:n_days]
    scores_list = []
    for _, fpath in files:
        try:
            s = pd.read_parquet(fpath, columns=["score"])["score"]
            scores_list.append(s.to_numpy(dtype=np.float64))
        except Exception:
            log.warning("corrupt parquet skipped: %s", fpath)
            continue
    if len(scores_list) < _MIN_REFERENCE_FILES:
        return None, len(scores_list)
    return np.concatenate(scores_list), len(scores_list)


def signal_quality_psi_step(ctx: DailyRunContext) -> None:
    """PSI distribution-drift monitor. Fail-open, alert-only."""
    if ctx.steps is not None:
        return
    if os.environ.get("ASHARE_USE_STALE") == "1":
        return
    dry_run = os.environ.get("ASHARE_SQ_DRYRUN") == "1"
    pred = ctx.pred_path
    predictions_dir = PREDICTIONS_DIR
    if pred is None:
        sidecar_path = predictions_dir / f"{ctx.trade_date}.meta.json"
        _write_sidecar(sidecar_path, {"psi": None, "psi_reason": "no_prediction"})
        log.warning("no prediction file; recorded null PSI")
        return
    pred_date = ctx.predictions_date_str
    end_date = pred_date
    sidecar = pred.with_suffix(".meta.json")
    try:
        ref, ref_n = load_reference_scores(predictions_dir, end_date, n_days=_REFERENCE_DAYS)
        if ref is None:
            log.warning("insufficient reference history (n=%d)", ref_n)
            _write_sidecar(sidecar, {"psi": None, "psi_reference_days": ref_n, "psi_reason": "insufficient_history"})
            return
        scores = pd.read_parquet(pred, columns=["score"])["score"].to_numpy(dtype=np.float64)
        psi = compute_psi(ref, scores)
        if psi is None:
            log.warning("PSI computation returned None (empty/degenerate)")
        _write_sidecar(sidecar, {"psi": psi, "psi_reference_days": ref_n})
        if psi is not None and psi > _PSI_THRESHOLD:
            if dry_run:
                log.warning("dry-run: alert suppressed")
            else:
                if not send_bridge_alert(
                    title="ashare SIG: PSI",
                    body=f"PSI {psi:.3f} > {_PSI_THRESHOLD} on {pred_date} (ref {_REFERENCE_DAYS}d, {_N_BUCKETS} buckets). Distribution drift tripwire.",
                    timeout=10,
                ):
                    log.error("PSI alert send failed (returned False)")
    except Exception as exc:
        log.error("signal_quality_psi_step failed: %s", exc, exc_info=True)
        _write_sidecar(sidecar, {"psi_error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# 10-02: Lagged IC Monitor
# ---------------------------------------------------------------------------
class SignalQualityError(Exception):
    """Raised by compute_forward_returns on subprocess failure."""


_fetch_script = """
import qlib, json, sys
from pathlib import Path
from ashare_lab.data.update import DEFAULT_PROVIDER_URI
qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI))
from qlib.data import D
instruments = json.loads(sys.argv[1])
date_from = sys.argv[2]
date_to = sys.argv[3]
raw = D.features(instruments=instruments, fields=["$close"],
                 start_time=date_from, end_time=date_to)
if raw is not None and not raw.empty:
    out = {}
    for idx, row in raw.iterrows():
        if isinstance(idx, tuple):
            inst = idx[0]
            date = str(idx[1])[:10]
        else:
            print("compute_forward_returns: flat index encountered, skipping row", file=sys.stderr)
            continue
        close = float(row.get("$close", 0))
        if close > 0:
            out.setdefault(inst, {})[date] = close
    result = {}
    for inst, dates in out.items():
        sorted_dates = sorted(dates.keys())
        if len(sorted_dates) >= 2 and sorted_dates[0] == date_from and sorted_dates[-1] == date_to:
            first_close = dates[date_from]
            last_close = dates[date_to]
            if first_close > 0 and last_close > 0:
                result[inst] = float(last_close / first_close - 1.0)
    print(json.dumps(result))
else:
    print("{}")
"""


def compute_forward_returns(
    instruments: list[str],
    date_from: str,
    date_to: str,
) -> pd.Series:
    """Compute T+5 forward returns via qlib subprocess."""
    if not instruments:
        return pd.Series(dtype=float)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _fetch_script, json.dumps(instruments), date_from, date_to],
            timeout=120,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        raise SignalQualityError("compute_forward_returns subprocess timeout")
    except (subprocess.SubprocessError, OSError) as e:
        raise SignalQualityError(f"compute_forward_returns subprocess error: {e}")
    if result.returncode != 0:
        raise SignalQualityError(f"compute_forward_returns exit {result.returncode}: {result.stderr}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise SignalQualityError(f"compute_forward_returns JSON parse error: {e}\nstdout={result.stdout!r}")
    series = pd.Series(data)
    if series.empty:
        log.warning("compute_forward_returns returned empty Series")
    return series


def lagged_ic_for_date(
    predictions_dir: Path,
    date_t5: str,
    forward_returns: pd.Series,
    min_instruments: int = _MIN_INSTRUMENTS,
) -> tuple[float | None, int]:
    """Compute Spearman rank IC between scores and forward returns."""
    pred_path = predictions_dir / f"{date_t5}.parquet"
    if not pred_path.exists():
        log.warning("missing t-5 parquet: %s", pred_path)
        return None, 0
    try:
        df = pd.read_parquet(pred_path, columns=["score", "instrument"])
    except Exception:
        log.warning("corrupt t-5 parquet: %s", pred_path)
        return None, 0
    score_series = df.set_index("instrument")["score"]
    aligned = score_series.align(forward_returns, join="inner")
    aligned_df = pd.DataFrame({"score": aligned[0], "return": aligned[1]}).dropna()
    if len(aligned_df) < min_instruments:
        log.warning("insufficient instruments (n=%d)", len(aligned_df))
        return None, len(aligned_df)
    if aligned_df["score"].nunique() <= 1 or aligned_df["return"].nunique() <= 1:
        log.warning("constant scores or returns")
        return None, len(aligned_df)
    ic = float(aligned_df["score"].corr(aligned_df["return"], method="spearman"))
    return ic, len(aligned_df)


def rolling_lagged_ic(
    predictions_dir: Path,
    end_date: str,
    window: int = _ROLLING_WINDOW,
) -> tuple[float | None, int]:
    """Compute rolling mean of lagged IC from sidecar files."""
    values = []
    for f in predictions_dir.glob("*.meta.json"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if not m:
            continue
        date = m.group(1)
        if date > end_date:
            continue
        try:
            meta = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt sidecar skipped: %s", f)
            continue
        val = meta.get("lagged_ic_t5")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            values.append((date, val))
    values.sort(key=lambda x: x[0], reverse=True)
    values = values[:window]
    if len(values) < window:
        return None, len(values)
    mean = sum(v for _, v in values) / len(values)
    return mean, len(values)


def signal_quality_ic_step(ctx: DailyRunContext) -> None:
    """Lagged-IC ranking-power decay monitor. Fail-open, alert-only."""
    if ctx.steps is not None:
        return
    if os.environ.get("ASHARE_USE_STALE") == "1":
        return
    predictions_dir = PREDICTIONS_DIR
    dry_run = os.environ.get("ASHARE_SQ_DRYRUN") == "1"
    trade_date = ctx.trade_date
    if ctx.pred_path is None:
        log.warning("no prediction file; skipping IC monitor")
        return
    pred_date = ctx.predictions_date_str  # Use prediction date for t-5 derivation
    date_t5_str = None
    try:
        pred_date_dt = dt.date.fromisoformat(pred_date)
        cursor = pred_date_dt
        for _ in range(5):
            cursor = previous_trading_day(cursor)
        date_t5_str = cursor.isoformat()
    except Exception as exc:
        log.error("failed to derive t5 date: %s", exc)
        _write_sidecar(
            predictions_dir / f"{trade_date}.meta.json",
            {"lagged_ic_error": f"t5_derivation: {type(exc).__name__}: {exc}"},
        )
        return
    pred_t5_path = predictions_dir / f"{date_t5_str}.parquet"
    t5_sidecar_path = predictions_dir / f"{date_t5_str}.meta.json"
    if not pred_t5_path.exists():
        log.warning("missing t-5 parquet for %s", date_t5_str)
        _write_sidecar(t5_sidecar_path, {"lagged_ic_t5": None, "lagged_ic_t5_n": 0, "lagged_ic_t5_reason": "missing_t5_parquet"})
        ic, n_instruments = None, 0
    else:
        try:
            instruments = pd.read_parquet(pred_t5_path, columns=["instrument"])["instrument"].tolist()
            forward_returns = compute_forward_returns(instruments, date_from=date_t5_str, date_to=trade_date)
            ic, n_instruments = lagged_ic_for_date(predictions_dir, date_t5_str, forward_returns)
            coverage = n_instruments / len(instruments) if instruments else 0
            if coverage < 0.95:
                log.warning("low instrument coverage (%.1f%%)", coverage * 100)
            _write_sidecar(t5_sidecar_path, {"lagged_ic_t5": ic, "lagged_ic_t5_n": n_instruments})
        except Exception as exc:
            log.error("IC computation failed: %s", exc, exc_info=True)
            _write_sidecar(t5_sidecar_path, {"lagged_ic_t5": None, "lagged_ic_error": f"{type(exc).__name__}: {exc}"})
    try:
        mean, n = rolling_lagged_ic(predictions_dir, trade_date)
    except Exception as exc:
        log.error("rolling_lagged_ic failed: %s", exc, exc_info=True)
        return
    if n >= _ROLLING_WINDOW and mean is not None and mean < _IC_THRESHOLD:
        if dry_run:
            log.warning("dry-run: alert suppressed")
        else:
            try:
                if not send_bridge_alert(
                    title="ashare SIG: lagged IC",
                    body=f"20d rolling lagged T+5 IC {mean:.4f} < {_IC_THRESHOLD} as of {trade_date} (n={_ROLLING_WINDOW}). Ranking-power decay tripwire.",
                    timeout=10,
                ):
                    log.error("lagged IC alert send failed (returned False)")
            except Exception as exc:
                log.error("lagged IC alert send exception: %s", exc)
