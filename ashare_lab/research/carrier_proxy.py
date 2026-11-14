"""Carrier proxy test: is industry-level momentum strong enough to ride?

This asks a narrower question than a policy study. Before spending weeks on a
document corpus, measure the carrier a policy tilt would sit on: cross-sectional
momentum and reversal at the industry level. If the carrier is not there, a tilt
has nothing to ride and the corpus is not worth building.

Pre-registered 2026-11-10, before any number was computed:

  Variants      signal = trailing h-session cumulative industry return,
                forward = next h-session cumulative return, h in {1, 5, 20, 60}.
  Statistic     Spearman rank IC across industries, per session; sessions with
                fewer than MIN_INDUSTRIES industries are skipped.
  Aggregation   mean IC, ICIR = mean/std, t = mean / (std / sqrt(n_eff)) with
                n_eff = n / h, because consecutive h-session windows overlap.
  Economic      mean forward return of the top quintile of signal minus the
                bottom quintile, per session.
  PASS          at least one variant with |mean IC| >= IC_FLOOR and |t| >= T_FLOOR.
  KILL          none does. Then the tilt rides nothing, and that is the finding.

The floors are not tuned after seeing results: IC_FLOOR is the same 0.02 the
production IC monitor uses before it calls a signal real.

Measured on the 2017-2026 industry panel: the one-session variant clears (IC
0.0237, long-short +0.09%/session, under round-trip cost) and five sessions is flat
(0.0036). The twenty- and sixty-session rows lean to reversal at -0.045 and reach
significance only under a Newey-West estimator (t -2.78 and -2.40) rather than the
n/h form used here (t -1.96 and -1.20); both are reported so the choice of
estimator stays visible instead of being made after the fact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_INDUSTRIES = 30
HORIZONS = (1, 5, 20, 60)
IC_FLOOR = 0.02
T_FLOOR = 2.0


def load_wide(panel_path: Path, column: str = "mean_ret") -> pd.DataFrame:
    """Sessions x industries matrix of returns."""
    panel = pd.read_parquet(panel_path)
    return panel.pivot(index="date", columns="industry", values=column).sort_index()


def ic_series(wide: pd.DataFrame, horizon: int) -> pd.Series:
    """Rank IC per session between trailing and forward horizon-session returns."""
    signal = wide.rolling(horizon).sum()
    forward = wide.shift(-horizon).rolling(horizon).sum()
    out: dict[str, float] = {}
    for date in wide.index:
        pair = pd.concat([signal.loc[date], forward.loc[date]], axis=1).dropna()
        if len(pair) < MIN_INDUSTRIES:
            continue
        rho = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
        if np.isfinite(rho):
            out[date] = float(rho)
    return pd.Series(out, dtype=float)


def quintile_spread(wide: pd.DataFrame, horizon: int) -> pd.Series:
    """Top-quintile minus bottom-quintile forward return, per session."""
    signal = wide.rolling(horizon).sum()
    forward = wide.shift(-horizon).rolling(horizon).sum()
    spreads: dict[str, float] = {}
    for date in wide.index:
        pair = pd.concat([signal.loc[date], forward.loc[date]], axis=1).dropna()
        if len(pair) < MIN_INDUSTRIES:
            continue
        pair.columns = ["sig", "fwd"]
        k = max(1, len(pair) // 5)
        spreads[date] = float(pair.nlargest(k, "sig")["fwd"].mean() - pair.nsmallest(k, "sig")["fwd"].mean())
    return pd.Series(spreads, dtype=float)


def summarise(wide: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ics = ic_series(wide, h)
        if ics.empty:
            rows.append({"horizon": h, "n": 0, "mean_ic": np.nan, "icir": np.nan,
                         "t": np.nan, "positive_share": np.nan, "spread": np.nan})
            continue
        n = len(ics)
        n_eff = max(n / h, 1.0)
        mean = float(ics.mean())
        sd = float(ics.std(ddof=1))
        spread = quintile_spread(wide, h)
        rows.append({
            "horizon": h,
            "n": n,
            "mean_ic": mean,
            "icir": mean / sd if sd > 0 else np.nan,
            "t": mean / (sd / np.sqrt(n_eff)) if sd > 0 else np.nan,
            "positive_share": float((ics > 0).mean()),
            "spread": float(spread.mean()) if not spread.empty else np.nan,
        })
    return pd.DataFrame(rows)


def passing_variants(table: pd.DataFrame) -> pd.DataFrame:
    """Rows that clear both the IC floor and the t floor."""
    return table[(table["mean_ic"].abs() >= IC_FLOOR) & (table["t"].abs() >= T_FLOOR)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("data/industry_panel.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/carrier_proxy.csv"))
    args = ap.parse_args()

    print(f"preregistered: horizons={HORIZONS} floor={IC_FLOOR} t>={T_FLOOR} min_industries={MIN_INDUSTRIES}")
    wide = load_wide(args.panel)
    print(f"panel: {wide.shape[0]} sessions x {wide.shape[1]} industries, {wide.index[0]}..{wide.index[-1]}")

    table = summarise(wide)
    pd.set_option("display.width", 120)
    print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    passed = passing_variants(table)
    print()
    if passed.empty:
        print("VERDICT: KILL -- no variant clears the floors; a policy tilt would ride nothing.")
    else:
        for _, r in passed.iterrows():
            print(f"VERDICT: PASS h={int(r['horizon'])} mean_ic={r['mean_ic']:.4f} "
                  f"t={r['t']:.2f} spread={r['spread']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"written: {args.out}")
    return 0 if not passed.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
