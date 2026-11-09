"""Industry-level return panel built from the price store.

The paper book is 84 sessions long, which is not enough to test a factor with a
20-day horizon. This panel gives the multi-year series that test needs: one
equal-weight return per industry per session, computed from the same store the
model reads, so a later policy tilt is measured against the prices it will trade.

The industry labels come from a single vendor snapshot (2026-07-10). Applying it
backwards is a known bias -- a company that changed its classification carries
today's label through history -- and every consumer must say so.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_lab.data.store_state import calendar, read_field

log = logging.getLogger(__name__)

MIN_MEMBERS = 5

# A-share daily moves are capped at +-20% (30% for a first-day listing). Anything
# larger is a broken row, not a return, and is excluded from the aggregate rather
# than allowed to drag an industry's mean.
SUSPECT_RETURN = 0.30


def load_industry_map(path: Path) -> dict[str, str]:
    """Map a store symbol (``sh600000``) to its industry label.

    baostock writes ``sh.600000``; the store writes ``sh600000``. The dot is
    dropped rather than replaced so the two spellings cannot drift apart.
    """
    df = pd.read_csv(path, dtype=str)
    missing = {"date", "code", "industry"} - set(df.columns)
    if missing:
        raise ValueError(f"industry file missing columns: {sorted(missing)}")
    if df["industry"].isna().any() or df["code"].isna().any():
        raise ValueError("industry file has empty code or industry")
    codes = df["code"].str.strip().str.replace(".", "", regex=False).str.lower()
    return dict(zip(codes, df["industry"].str.strip()))


def _symbol_returns(
    provider_uri: Path,
    symbol: str,
    first: int,
    last: int,
) -> pd.Series | None:
    """Adjusted close-to-close returns for one symbol over slots [first, last]."""
    got = read_field(provider_uri, symbol, "close")
    if got is None:
        return None
    start, values = got
    lo = max(first, start)
    hi = min(last, start + len(values) - 1)
    if hi - lo < 1:
        return None
    window = values[lo - start : hi - start + 1].astype("float64")
    # fill_method=None: a padded NaN would turn a suspended session into a
    # zero return and count that symbol as a member for the day.
    return pd.Series(window, index=range(lo, hi + 1)).pct_change(fill_method=None)


def build_panel(
    provider_uri: Path,
    industry_map: dict[str, str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """One row per (date, industry).

    Columns: date, industry, n_members, n_suspect, mean_ret, median_ret,
    std_ret, up_share. Industries below ``MIN_MEMBERS`` that session are
    dropped: with four names the mean is one stock's move wearing a label.
    """
    cal = calendar(provider_uri)
    # A caller passes calendar dates, not sessions: snap outward so 1 January
    # means "the first session of that year".
    first = 0 if start is None else bisect.bisect_left(cal, start)
    last = len(cal) - 1 if end is None else bisect.bisect_right(cal, end) - 1
    if first >= last:
        raise ValueError(f"empty window: {start}..{end}")

    series: dict[str, pd.Series] = {}
    for symbol in industry_map:
        row = _symbol_returns(provider_uri, symbol, first, last)
        if row is not None:
            series[symbol] = row

    if not series:
        raise ValueError("no symbol had two readable closes in the window")

    frame = pd.DataFrame(series).sort_index()
    suspect = frame.abs() > SUSPECT_RETURN

    groups: dict[str, list[str]] = {}
    for symbol in frame.columns:
        groups.setdefault(industry_map[symbol], []).append(symbol)

    rows = []
    for industry, members in groups.items():
        sub = frame[members].mask(suspect[members])
        n = sub.notna().sum(axis=1).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "slot": sub.index.to_numpy(),
                    "industry": industry,
                    "n_members": n.to_numpy(),
                    "n_suspect": suspect[members].sum(axis=1).astype(int).to_numpy(),
                    "mean_ret": sub.mean(axis=1).to_numpy(),
                    "median_ret": sub.median(axis=1).to_numpy(),
                    "std_ret": sub.std(axis=1).to_numpy(),
                    "up_share": (sub > 0).sum(axis=1).to_numpy() / n.where(n > 0, np.nan).to_numpy(),
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = panel["slot"].map(lambda i: cal[int(i)])
    panel = panel.drop(columns="slot")
    panel = panel[panel["n_members"] >= MIN_MEMBERS].reset_index(drop=True)

    dropped = int((suspect.to_numpy() & frame.notna().to_numpy()).sum())
    if dropped:
        log.warning("industry panel excluded %d cell(s) beyond %.0f%%", dropped, SUSPECT_RETURN * 100)
    return panel[["date", "industry", "n_members", "n_suspect", "mean_ret", "median_ret", "std_ret", "up_share"]]


def to_wide(panel: pd.DataFrame, column: str = "mean_ret") -> pd.DataFrame:
    """date x industry matrix, the shape every statistic below wants."""
    if column not in panel.columns:
        raise ValueError(f"no such column: {column}")
    return panel.pivot(index="date", columns="industry", values=column).sort_index()


def summary(panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20, 60)) -> dict:
    """The handful of numbers that decide whether a tilt is worth building.

    - ``dispersion``: median cross-industry spread per session. A tilt that
      cannot move a name across this spread cannot reorder anything.
    - ``momentum_autocorr_<h>d``: does an industry's past ``h``-session return
      predict the next ``h``? The policy literature's continuation claim lives
      here; a value indistinguishable from zero means the tilt has no carrier.
    """
    wide = to_wide(panel)

    stats: dict = {
        "sessions": int(wide.shape[0]),
        "industries": int(wide.shape[1]),
        # One industry has no cross-section to spread against.
        "dispersion": float(wide.std(axis=1).median()) if wide.shape[1] > 1 else 0.0,
        "first": str(wide.index[0]),
        "last": str(wide.index[-1]),
    }

    for horizon in horizons:
        # ``wide`` holds returns, not prices: a ratio of returns is meaningless.
        # Compounding the next h sessions gives the forward move from t.
        past = wide.rolling(horizon).sum()
        ahead = wide.shift(-horizon).rolling(horizon).sum()
        # Align by (date, industry) through an explicit join: .loc on a sliced
        # MultiIndex silently returns nothing.
        joined = pd.concat(
            [past.stack().rename("past"), ahead.stack().rename("ahead")],
            axis=1,
            join="inner",
        ).dropna()
        stats[f"momentum_autocorr_{horizon}d"] = (
            float(joined["past"].corr(joined["ahead"])) if len(joined) > 2 else float("nan")
        )

    return stats
