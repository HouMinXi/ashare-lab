"""ILLIQ calibration script for Track S (R6) dynamic-slippage replay.

Computes per-stock Amihud illiquidity (ILLIQ) as 20-day rolling mean of
|r| / (P * V) from daily qlib bars. Outputs a frozen calibration JSON
artifact consumed by the Track S replay engine.

Usage:
    python -m ashare_lab.research.calibrate_illiq \\
        --end-date 2026-08-17 \\
        --output experiments/shadow_slippage/calibration_illiq.json \\
        --window 20

Calibration is frozen at Day 0 (2026-08-17) per R6 rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 20
DEFAULT_END_DATE = "2026-08-17"
DEFAULT_START_DATE_OFFSET = 60  # fetch ~60 trading days for 20d rolling + rank check
MIN_VALID_STOCKS = 50  # sanity threshold for calibration health
DEFAULT_QUINTILE_RATES = [0.0005, 0.0008, 0.0010, 0.0015, 0.0025]


def fetch_qlib_bars_df(
    start_date: str,
    end_date: str,
    symbols: list[str] | None = None,
    provider_uri: str | None = None,
) -> Any:
    """Fetch daily OHLCV bars from qlib as pandas DataFrame."""
    try:
        import qlib
        from qlib.data import D
    except ImportError as e:
        raise ImportError(f"Failed to import qlib for ILLIQ calibration: {e}") from e

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI

    uri = provider_uri or str(DEFAULT_PROVIDER_URI)
    try:
        qlib.init(provider_uri=uri)
    except Exception:
        pass

    if symbols:
        insts = symbols
    else:
        insts = D.instruments(market="all")

    raw = D.features(
        instruments=insts,
        fields=["$close", "$volume", "$change", "$factor"],
        start_time=start_date,
        end_time=end_date,
    )
    return raw


def compute_illiq_from_df(
    raw_df: Any,
    window: int = DEFAULT_WINDOW,
) -> tuple[dict[str, dict[str, Any]], float | None]:
    """Compute 20-day rolling mean ILLIQ per stock and rank correlation.

    ILLIQ_i,t = |r_i,t| / (P_i,t * V_i,t)

    Returns (per_stock_dict, spearman_rank_corr).
    """
    import numpy as np

    if raw_df is None or raw_df.empty:
        return {}, None

    df = raw_df.copy()
    # Denormalize close price: actual_close = $close / $factor
    df["actual_close"] = df["$close"] / df["$factor"].clip(lower=1e-8)
    df["illiq"] = df["$change"].abs() / (df["actual_close"] * df["$volume"])
    # Treat non-positive volume or non-finite price as NaN
    df.loc[df["$volume"] <= 0, "illiq"] = np.nan
    df.loc[df["actual_close"] <= 0, "illiq"] = np.nan

    # Unstack illiq with date as index and symbols as columns
    illiq_table = df["illiq"].unstack(level=0)
    # Filter dates
    dates = illiq_table.index.tolist()

    # Per-stock 20-day mean on the last `window` dates
    if len(dates) < window:
        last_window_table = illiq_table
    else:
        last_window_table = illiq_table.iloc[-window:]

    last_means = last_window_table.mean(axis=0).dropna()

    per_stock: dict[str, dict[str, Any]] = {}
    for sym, val in last_means.items():
        if np.isfinite(val) and val > 0:
            per_stock[str(sym)] = {
                "illiq": float(val),
                "n_days": int(last_window_table[sym].notna().sum()),
            }

    # Gate 4 rank correlation check between two consecutive snapshots
    spearman_corr: float | None = None
    if len(dates) >= 2 * window:
        snap1 = illiq_table.iloc[-2 * window : -window].mean(axis=0).dropna()
        snap2 = illiq_table.iloc[-window:].mean(axis=0).dropna()
        common = snap1.index.intersection(snap2.index)
        if len(common) >= MIN_VALID_STOCKS:
            s1 = snap1.loc[common]
            s2 = snap2.loc[common]
            rank1 = s1.rank()
            rank2 = s2.rank()
            spearman_corr = float(rank1.corr(rank2))

    return per_stock, spearman_corr


def compute_quintile_bands(
    per_stock: dict[str, dict[str, Any]],
    quintile_rates: list[float] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    """Assign quintile bands (0 to 4) by cross-sectional ILLIQ.

    Returns (per_stock_with_quintile, band_boundaries).
    """
    import numpy as np

    rates = quintile_rates or DEFAULT_QUINTILE_RATES
    illiq_values = [v["illiq"] for v in per_stock.values() if v.get("illiq", 0) > 0]
    if not illiq_values:
        return per_stock, []

    # 5 quintiles -> 4 cut points (20%, 40%, 60%, 80%)
    boundaries = [
        float(np.percentile(illiq_values, p)) for p in [20, 40, 60, 80]
    ]

    for data in per_stock.values():
        illiq = data.get("illiq", 0.0)
        q = 0
        for i, b in enumerate(boundaries):
            if illiq > b:
                q = i + 1
        q = max(0, min(q, len(rates) - 1))
        data["quintile"] = q

    return per_stock, boundaries


def gate_4_health_check(
    per_stock: dict[str, dict[str, Any]],
    boundaries: list[float],
    spearman_corr: float | None = None,
) -> dict[str, Any]:
    """Gate 4 calibration health check.

    Verifies sufficient stock coverage, valid boundaries, and rank stability.
    """
    result: dict[str, Any] = {
        "pass": True,
        "n_stocks": len(per_stock),
        "spearman_rank_corr": spearman_corr,
        "warnings": [],
        "failures": [],
    }

    if len(per_stock) < MIN_VALID_STOCKS:
        result["pass"] = False
        result["failures"].append(
            f"Only {len(per_stock)} valid stocks; need >= {MIN_VALID_STOCKS}"
        )

    if len(boundaries) < 4:
        result["pass"] = False
        result["failures"].append(
            "Could not compute 4 quintile boundaries"
        )
    elif not (boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]):
        result["pass"] = False
        result["failures"].append(
            f"Quintile boundaries are not strictly monotonic: {boundaries}"
        )

    if spearman_corr is not None:
        if spearman_corr < 0.70:
            result["warnings"].append(
                f"Spearman rank correlation {spearman_corr:.4f} is below 0.70"
            )
    else:
        result["warnings"].append(
            "Spearman rank correlation could not be computed (insufficient history)"
        )

    return result


def calibrate(
    end_date: str = DEFAULT_END_DATE,
    window: int = DEFAULT_WINDOW,
    output_path: str | Path | None = None,
    symbols: list[str] | None = None,
    provider_uri: str | None = None,
) -> dict[str, Any]:
    """Run full ILLIQ calibration pipeline and write artifact."""
    import qlib
    from qlib.data import D
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI

    uri = provider_uri or str(DEFAULT_PROVIDER_URI)
    try:
        qlib.init(provider_uri=uri)
    except Exception:
        pass

    try:
        cal = D.calendar(start_time="2020-01-01", end_time=end_date)
        cal_dates = sorted([str(d)[:10] for d in cal])
    except Exception as e:
        logger.error("Failed to fetch qlib calendar: %s", e)
        cal_dates = []

    if not cal_dates:
        raise RuntimeError("No trading dates found in qlib calendar")
    try:
        end_idx = cal_dates.index(end_date)
    except ValueError:
        logger.warning("%s not in calendar; using latest available date", end_date)
        end_idx = len(cal_dates) - 1
        end_date = cal_dates[end_idx]

    start_idx = max(0, end_idx - DEFAULT_START_DATE_OFFSET)
    start_date = cal_dates[start_idx]

    logger.info(
        "Calibration window: %s to %s (%d trading days)",
        start_date, end_date, end_idx - start_idx + 1,
    )

    raw_df = fetch_qlib_bars_df(start_date, end_date, symbols=symbols, provider_uri=provider_uri)
    per_stock, spearman_corr = compute_illiq_from_df(raw_df, window=window)
    per_stock, boundaries = compute_quintile_bands(per_stock)
    health = gate_4_health_check(per_stock, boundaries, spearman_corr=spearman_corr)

    artifact = {
        "model": "illiq_20d_rolling",
        "calibration_date": {
            "start": start_date,
            "end": end_date,
        },
        "window_days": window,
        "n_stocks": len(per_stock),
        "per_stock": per_stock,
        "quintile_boundaries": boundaries,
        "quintile_rates": DEFAULT_QUINTILE_RATES,
        "gate_4_health": health,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, ensure_ascii=True))
        logger.info("Wrote calibration artifact to %s", out)

    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ILLIQ calibration for Track S replay (R6)"
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END_DATE,
        help="Calibration end date (default: %(default)s)",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help="Rolling window in trading days (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON artifact path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    calibrate(
        end_date=args.end_date,
        window=args.window,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
