"""Signal-quality monitoring for the daily IC series.

The retrain no longer keys on IC. Over 30 sessions this book's daily
cross-sectional IC (a Spearman correlation across ~996 names) has a standard
deviation of 0.23, so the old "three consecutive days below 0.02" rule was
reading the sign of a volatile series, not decay: 17 of 30 days fell below the
threshold, and consecutive days correlate at 0.74, so three of them carried
about one observation's worth of evidence.

What remains here is a monitor. It reports the mean of a long window, the ICIR
that Ding and Sun (2022) show converges to the ex ante information ratio, and a
Newey-West t-statistic so that autocorrelated days are not counted as
independent evidence. It raises a flag for a human; it never starts a retrain.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

# Window and bandwidth in sessions. 60 sessions (~3 months) is the shortest
# window in which a 0.05 shift is visible at all; the bandwidth follows the
# spec's four weeks, which also matches the series' autocorrelation length.
WINDOW = 60
BANDWIDTH = 20
ALERT_T = -2.0


def load_series(path: Path) -> list[tuple[str, float]]:
    """Read (date, ic) pairs from the append-only history file.

    A malformed line is skipped rather than aborting the monitor: the file is
    appended by a daily job and one bad row should not blind the report.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        date, raw = parts[0].strip(), parts[1].strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            out.append((date, value))
    return out


def _lag(series: list[float], h: int) -> float:
    """Sample autocovariance at lag h (divisor n, as in the NW estimator)."""
    n = len(series)
    mean = sum(series) / n
    return sum((series[i] - mean) * (series[i + h] - mean) for i in range(n - h)) / n


def newey_west_t(series: list[float], bandwidth: int = BANDWIDTH) -> float | None:
    """t-statistic for "mean is zero" with HAC standard errors.

    Bartlett weights (1 - h/(L+1)) keep the estimate positive and downweight
    distant lags. Returns None when the series is too short or has no variance.
    """
    n = len(series)
    if n < 3:
        return None
    mean = sum(series) / n
    variance = _lag(series, 0)
    if variance <= 0:
        return None
    long_run = variance
    for h in range(1, min(bandwidth, n - 1) + 1):
        weight = 1.0 - h / (bandwidth + 1)
        long_run += 2.0 * weight * _lag(series, h)
    if long_run <= 0:
        # A strongly negative first lag can drive the sum below zero; fall back
        # to the iid variance rather than reporting an imaginary standard error.
        long_run = variance
    return mean / math.sqrt(long_run / n)


def summarize(
    pairs: list[tuple[str, float]],
    window: int = WINDOW,
    bandwidth: int = BANDWIDTH,
) -> dict:
    """Trailing-window statistics for the IC series.

    The whole returned dict is JSON-friendly so the report can print it and a
    test can assert on it.
    """
    tail = pairs[-window:]
    values = [v for _, v in tail]
    stats: dict = {
        "n": len(values),
        "window": window,
        "first_date": tail[0][0] if tail else None,
        "last_date": tail[-1][0] if tail else None,
        "mean": None,
        "sd": None,
        "icir": None,
        "t_iid": None,
        "t_hac": None,
        "alert": False,
    }
    if len(values) < 2:
        return stats

    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    stats["mean"] = mean
    stats["sd"] = sd
    if sd > 0:
        stats["icir"] = mean / sd
        stats["t_iid"] = mean / (sd / math.sqrt(len(values)))
    stats["t_hac"] = newey_west_t(values, bandwidth)
    stats["alert"] = should_alert(stats)
    return stats


def should_alert(stats: dict, threshold: float = ALERT_T) -> bool:
    """Flag a sustained negative edge, not a bad day.

    Both conditions are required: a positive mean with a large |t| is a working
    signal, and a negative mean with a t inside the noise band is a drawdown.
    """
    mean = stats.get("mean")
    t_hac = stats.get("t_hac")
    if mean is None or t_hac is None:
        return False
    return mean < 0 and t_hac < threshold


def format_line(stats: dict) -> str:
    """One report line; empty when there is not enough history to say anything."""
    if not stats.get("n") or stats["mean"] is None:
        return ""
    mean = stats["mean"]
    icir = stats.get("icir")
    t_hac = stats.get("t_hac")
    parts = [f"IC {stats['n']}日: 均值 {mean:+.4f}"]
    if icir is not None:
        parts.append(f"ICIR {icir:+.3f}")
    if t_hac is not None:
        parts.append(f"t(NW) {t_hac:+.2f}")
    if stats.get("first_date") and stats.get("last_date"):
        parts.append(f"{stats['first_date']}~{stats['last_date']}")
    line = " | ".join(parts)
    if stats.get("alert"):
        line = "⚠️ 信号持续为负 (需人工判断，不自动重训) | " + line
    return line
