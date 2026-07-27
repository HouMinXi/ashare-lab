"""Three timing null models for the L2 timing layer.

Pure functions: close series in, m series out.  No data access, no
side effects, no lookahead (m(t) uses only data through t-1).

Null definitions are FROZEN per .planning/l2_timing_preregistration_20260726.md.
Changes require PM re-freezing with disclosure.

N1 -- CSI1000 20d MA trend filter
N2 -- 20d realized volatility percentile
N3 -- drawdown state map (mirrors L1 risk_state.py lockdown)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Multiplier output values
M_FULL = 1.0
M_HALF = 0.5
M_ZERO = 0.0


# ---------------------------------------------------------------------------
# N1: MA trend filter
# ---------------------------------------------------------------------------


def ma_trend_multiplier(closes: pd.Series) -> pd.Series:
    """N1: CSI1000 20d MA trend filter.

    Definitions (no-lookahead: m(t) uses data through t-1):
      ma20_t  = mean(close, last 20 trading days through t-1)
      slope_t = (ma20_t - ma20_{t-5}) / 5
      m = 1.0 if close_{t-1} >= ma20_t
      m = 0.5 if close_{t-1} <  ma20_t AND slope_t >= 0
      m = 0.0 if close_{t-1} <  ma20_t AND slope_t <  0

    Warmup (fewer than 25 days): m = 1.0 (conservative default).

    Args:
        closes: CSI1000 close prices indexed by trading date.

    Returns:
        Series of m in {0.0, 0.5, 1.0} with same index as input.
    """
    if closes.empty:
        return pd.Series(dtype=float)

    # shift(1) enforces no-lookahead: day t sees only t-1 and earlier
    prev_close = closes.shift(1)
    ma20 = prev_close.rolling(20, min_periods=20).mean()
    slope = (ma20 - ma20.shift(5)) / 5

    m = pd.Series(M_FULL, index=closes.index)

    # Below MA and falling slope -> m=0.0
    below_ma_falling = (prev_close < ma20) & (slope < 0)
    m = m.where(~below_ma_falling, other=M_ZERO)

    # Below MA but rising/flat slope -> m=0.5
    # (only where m is still 1.0, i.e. not already set to 0.0)
    below_ma_rising = (prev_close < ma20) & (slope >= 0) & (m == M_FULL)
    m = m.where(~below_ma_rising, other=M_HALF)

    # Warmup: where ma20 or slope is NaN, keep m=1.0
    warmup = ma20.isna() | slope.isna()
    m = m.where(~warmup, other=M_FULL)

    return m


# ---------------------------------------------------------------------------
# N2: Volatility percentile
# ---------------------------------------------------------------------------


def vol_percentile_multiplier(closes: pd.Series) -> pd.Series:
    """N2: 20d realized volatility percentile.

    Definitions (no-lookahead: m(t) uses data through t-1):
      vol_t  = stdev(log returns, last 20 trading days through t-1)
      pct_t  = fraction of trailing 252 trading days whose vol < vol_t
      m = 1.0 if pct_t <  0.80
      m = 0.5 if 0.80 <= pct_t < 0.95
      m = 0.0 if pct_t >= 0.95

    Warmup: fewer than 21 days (no vol20) -> m=1.0.
    Percentile denominator: only non-NaN vol20 values in trailing window.

    Args:
        closes: CSI1000 close prices indexed by trading date.

    Returns:
        Series of m in {0.0, 0.5, 1.0} with same index as input.
    """
    if closes.empty:
        return pd.Series(dtype=float)

    # Log returns: ln(close_t / close_{t-1})
    log_ret = np.log(closes / closes.shift(1))
    # shift(1) enforces no-lookahead: day t sees only returns through t-1
    vol20 = log_ret.shift(1).rolling(20, min_periods=20).std()

    m = pd.Series(M_FULL, index=closes.index)

    for i in range(len(vol20)):
        v = vol20.iloc[i]
        if pd.isna(v):
            # Warmup: not enough data for vol20
            m.iloc[i] = M_FULL
            continue

        # Trailing 252 days of vol20, excluding current day
        start = max(0, i - 252)
        window = vol20.iloc[start:i].dropna()

        if len(window) == 0:
            # No prior vol20 values: conservative default
            m.iloc[i] = M_FULL
            continue

        pct = (window < v).mean()

        if pct >= 0.95:
            m.iloc[i] = M_ZERO
        elif pct >= 0.80:
            m.iloc[i] = M_HALF
        else:
            m.iloc[i] = M_FULL

    return m


# ---------------------------------------------------------------------------
# N3: Drawdown state map
# ---------------------------------------------------------------------------


def drawdown_state_multiplier(closes: pd.Series) -> pd.Series:
    """N3: Drawdown state map with lockdown (mirrors L1 risk_state.py).

    Definitions (no-lookahead: m(t) uses data through t-1):
      dd_t = (peak252_{t-1} - close_{t-1}) / peak252_{t-1}
      peak252 = max close over trailing 252 trading days through t-1
      m = 1.0 if dd_t <  0.10
      m = 0.5 if 0.10 <= dd_t < 0.12
      m = 0.0 if dd_t >= 0.12, UNTIL dd_t < 0.10 AND >= 5 trading days
        elapsed since m=0.0 entry (L1 pattern: len(between) >= 5)

    Warmup (fewer than 2 days): m = 1.0.

    Args:
        closes: CSI1000 close prices indexed by trading date.

    Returns:
        Series of m in {0.0, 0.5, 1.0} with same index as input.
    """
    if closes.empty:
        return pd.Series(dtype=float)

    n = len(closes)
    m = np.full(n, M_FULL)

    # Track state across the series
    in_lockdown = False
    lockdown_enter_idx: int | None = None
    lockdown_days = 5

    for i in range(n):
        # Need at least 2 days: current uses t-1
        if i < 1:
            m[i] = M_FULL
            continue

        # Peak over trailing 252 days through t-1
        start = max(0, i - 252)
        peak = closes.iloc[start:i].max()

        if peak <= 0:
            m[i] = M_FULL
            continue

        # close_{t-1} (no-lookahead)
        prev_close = closes.iloc[i - 1]
        dd = (peak - prev_close) / peak

        if in_lockdown:
            # Check lockdown exit: dd < 0.10 AND 5 trading days elapsed
            elapsed = i - lockdown_enter_idx  # days since enter (exclusive)
            if dd < 0.10 and elapsed >= lockdown_days:
                in_lockdown = False
                lockdown_enter_idx = None
                m[i] = M_FULL
            else:
                m[i] = M_ZERO
        else:
            if dd >= 0.12:
                in_lockdown = True
                lockdown_enter_idx = i
                m[i] = M_ZERO
            elif dd >= 0.10:
                m[i] = M_HALF
            else:
                m[i] = M_FULL

    return pd.Series(m, index=closes.index)
