"""Pure endpoint admission on an explicit provider-session calendar."""

from numbers import Integral

import numpy as np
import pandas as pd


_WINDOW_REASONS = ["eligible", "insufficient-own-history", "calendar-gap"]
_TICKS_PER_DAY = {"s": 86400, "ms": 86400000, "us": 86400000000, "ns": 86400000000000}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _midnight_days(values: np.ndarray, name: str) -> np.ndarray:
    """Divide original ticks; datetime casts can overflow near the lower bound."""
    ticks_per_day = _TICKS_PER_DAY[np.datetime_data(values.dtype)[0]]
    ticks = values.view("i8")
    if (ticks % ticks_per_day != 0).any():
        raise ValueError(f"{name} must contain timezone-naive midnight dates without NaT")
    return ticks // ticks_per_day


def _validate_dates(dates: pd.Index, name: str) -> np.ndarray:
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a DatetimeIndex")
    if dates.hasnans or dates.tz is not None:
        raise ValueError(f"{name} must contain timezone-naive midnight dates without NaT")
    return _midnight_days(dates.to_numpy(), name)


def _session_positions(index: pd.MultiIndex, calendar: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    calendar_days = _validate_dates(calendar, "calendar")
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("calendar must be sorted and unique")
    if (
        not isinstance(index, pd.MultiIndex)
        or not all(isinstance(name, str) for name in index.names)
        or index.names != ["instrument", "datetime"]
    ):
        raise ValueError("index must have instrument, datetime levels in that order")
    instruments = index.get_level_values("instrument")
    if not all(isinstance(value, str) and value for value in instruments):
        raise ValueError("index instruments must be nonempty strings")
    index_days = _validate_dates(index.get_level_values("datetime"), "index dates")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be sorted and unique")
    sessions = pd.Index(calendar_days).get_indexer(index_days)
    if (sessions < 0).any():
        raise ValueError("index date absent from calendar")
    return sessions, calendar_days


def classify_sequence_windows(
    index: pd.MultiIndex,
    calendar: pd.DatetimeIndex,
    seq_len: int,
) -> pd.Series:
    """Classify the last seq_len own rows without removing history rows."""
    seq_len = _positive_integer(seq_len, "seq_len")
    sessions, _ = _session_positions(index, calendar)
    rows = np.arange(len(index))
    instruments = index.codes[0]
    own_break = np.r_[True, instruments[1:] != instruments[:-1]]
    run_break = own_break | np.r_[True, np.diff(sessions) != 1]
    own_start = np.maximum.accumulate(np.where(own_break, rows, 0))
    run_start = np.maximum.accumulate(np.where(run_break, rows, 0))
    codes = np.where(rows - own_start + 1 < seq_len, 1, np.where(rows - run_start + 1 >= seq_len, 0, 2))
    return pd.Series(pd.Categorical.from_codes(codes, categories=_WINDOW_REASONS), index=index, name="window_reason")


def label_maturity_mask(
    index: pd.MultiIndex,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
    cutoff: pd.Timestamp,
) -> np.ndarray:
    """Whether every required future session exists by the inclusive cutoff."""
    horizon = _positive_integer(horizon, "horizon")
    sessions, calendar_days = _session_positions(index, calendar)
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff):
        raise ValueError("cutoff must be a Timestamp without NaT")
    if cutoff.tz is not None:
        raise ValueError("cutoff must be a timezone-naive midnight date")
    cutoff_day = int(_midnight_days(np.asarray(cutoff.asm8).reshape(1), "cutoff")[0])
    last_observed = int(calendar_days.searchsorted(cutoff_day, side="right")) - 1
    # Compare remaining sessions instead of adding a possibly huge horizon.
    return last_observed - sessions >= horizon
