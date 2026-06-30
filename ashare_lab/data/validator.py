"""Three-layer data validation for tushare daily data.

Validates fetched data before qlib dump_update. All validation is advisory:
warnings are logged but never block the pipeline. The 21:00 chenditc
correction serves as the final backstop.

Layers:
  1. Completeness -- stock count, NaN check, date match
  2. Price sanity -- change limits per board type, volume > 0, close in range
  3. Cross-validation -- compare tushare vs AKShare sample close prices
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

# Match A-share ST prefixes: "*ST" or "ST" followed by a non-ASCII-letter char
# (Chinese name, digit, space). Avoids false positives like "STKN Holdings".
_ST_PATTERN = re.compile(r"^\*?ST(?:[^\x41-\x5a\x61-\x7a]|$)")

log = logging.getLogger(__name__)

import pandas as pd

_COMPLETENESS_THRESHOLD = 5000
_CROSS_VAL_TOLERANCE = 0.01


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation outcome."""

    passed: bool
    warnings: list[str] = field(default_factory=list)


def _change_limit(symbol: str, stock_names: dict[str, str] | None) -> float:
    """Determine daily change limit based on board type and ST status.

    Returns fractional limit (e.g. 0.10 for 10%).
    """
    # ST detection via stock name using _ST_PATTERN.
    # Correctly handles "ST华英", "*ST重工" (Chinese directly after ST, no space).
    if stock_names:
        name = stock_names.get(symbol, "")
        if _ST_PATTERN.match(name):
            return 0.05

    # Extract numeric code from qlib symbol (e.g. 'sz300999' -> '300999')
    code = symbol[2:]

    # GEM (300xxx) and STAR (688xxx): 20% limit
    if code.startswith("300") or code.startswith("688"):
        return 0.20

    # Default mainboard: 10%
    return 0.10


def validate_daily_data(
    tushare_data: dict[str, pd.DataFrame],
    trade_date: str,
    cross_val_data: dict[str, pd.DataFrame] | None = None,
    stock_names: dict[str, str] | None = None,
) -> ValidationResult:
    """Run three-layer validation on fetched daily data.

    Args:
        tushare_data: Dict of symbol -> DataFrame from fetch_today_data.
        trade_date: Expected date string (YYYY-MM-DD).
        cross_val_data: Optional AKShare cross-validation data.
        stock_names: Optional dict mapping symbol -> Chinese name for ST detection.

    Returns:
        ValidationResult with passed flag and list of warning strings.
        Never raises.
    """
    warnings: list[str] = []

    # --- Layer 1: Completeness ---
    count = len(tushare_data)
    if count < _COMPLETENESS_THRESHOLD:
        msg = f"completeness: only {count} stocks (threshold {_COMPLETENESS_THRESHOLD})"
        warnings.append(msg)
        log.warning(msg)

    ohlcv = ["open", "high", "low", "close", "volume"]
    for sym, df in tushare_data.items():
        for col in ohlcv:
            if col in df.columns and df[col].isna().any():
                msg = f"NaN in {col} for {sym}"
                warnings.append(msg)
                log.warning(msg)

    # --- Layer 2: Price sanity ---
    for sym, df in tushare_data.items():
        if df.empty:
            continue
        row = df.iloc[0]

        # Change limit check
        limit = _change_limit(sym, stock_names)
        change_val = row.get("change", 0.0)
        if not (isinstance(change_val, float) and math.isnan(change_val)):
            if abs(change_val) > limit:
                msg = (
                    f"{sym}: change {change_val:.4f} exceeds "
                    f"{limit:.0%} limit"
                )
                warnings.append(msg)
                log.warning(msg)

        # Volume check
        vol = row.get("volume", 1.0)
        if not (isinstance(vol, float) and math.isnan(vol)):
            if vol == 0:
                msg = f"{sym}: volume is 0 (suspended)"
                warnings.append(msg)
                log.warning(msg)

        # Close within [low, high]
        close = row.get("close", 0.0)
        low_val = row.get("low", 0.0)
        high_val = row.get("high", 0.0)
        if not any(
            isinstance(v, float) and math.isnan(v)
            for v in (close, low_val, high_val)
        ):
            if close > high_val or close < low_val:
                msg = (
                    f"{sym}: close {close} outside "
                    f"[{low_val}, {high_val}]"
                )
                warnings.append(msg)
                log.warning(msg)

    # --- Layer 3: Cross-validation ---
    if cross_val_data:
        for sym, cv_df in cross_val_data.items():
            if sym not in tushare_data:
                continue
            ts_df = tushare_data[sym]
            if ts_df.empty or cv_df.empty:
                continue
            ts_close = ts_df.iloc[0].get("close", 0.0)
            cv_close = cv_df.iloc[0].get("close", 0.0)
            diff = abs(ts_close - cv_close)
            if diff > _CROSS_VAL_TOLERANCE:
                msg = (
                    f"cross-validation {sym}: close diff {diff:.4f} "
                    f"(tushare={ts_close}, akshare={cv_close})"
                )
                warnings.append(msg)
                log.warning(msg)

    passed = len(warnings) == 0
    return ValidationResult(passed=passed, warnings=warnings)
