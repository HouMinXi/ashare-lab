"""Read the last stored row out of the qlib binary feature store.

A qlib ``<field>.day.bin`` opens with a float32 holding the calendar slot of its
first value, followed by one float32 per calendar day. A symbol's bin rarely
starts at slot zero (sh601318 starts at 1718), so the header has to be honoured;
everything here is written against that fact.

The two fields can drift: a writer that died between two files leaves one of them
a slot short. Pairs are therefore taken from the slot both files agree on, and a
symbol whose tail is NaN is walked back to its last written row rather than
declared new -- a suspended stock that resumes must not restart at factor 1.0.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from ashare_lab.data.adjust import PrevState


@lru_cache(maxsize=8)
def _calendar_cached(path: str, mtime: float) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def calendar(provider_uri: Path) -> tuple[str, ...]:
    """Trading calendar, cached until the file changes."""
    path = Path(provider_uri) / "calendars" / "day.txt"
    if not path.exists():
        return ()
    return _calendar_cached(str(path), path.stat().st_mtime)


def feature_dir(provider_uri: Path, symbol: str) -> Path:
    return Path(provider_uri) / "features" / symbol.lower()


def read_field(provider_uri: Path, symbol: str, field: str) -> tuple[int, np.ndarray] | None:
    """(first slot, values) for one field, or None when unreadable.

    A file whose body is not a whole number of float32s is the residue of an
    interrupted write. Reading the last four bytes of it yields a denormal, and
    a denormal factor silently destroys the series, so it is rejected here.
    """
    path = feature_dir(provider_uri, symbol) / f"{field}.day.bin"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < 8 or (len(raw) - 4) % 4:
        return None
    header = struct.unpack("<f", raw[:4])[0]
    if not math.isfinite(header) or header < 0:
        # A NaN header is what an interrupted first write leaves behind, and
        # int(nan) raises outside this function's contract.
        return None
    first = int(header)
    values = np.frombuffer(raw[4:], dtype="<f4")
    if len(values) == 0:
        return None
    return first, values


def last_state(provider_uri: Path, symbol: str) -> tuple[PrevState, str] | None:
    """Last stored close and factor for ``symbol``, with its calendar date.

    Walks back over an unwritten (NaN) tail so a suspended stock keeps the basis
    it had when it last traded. Returns None when the symbol has no usable
    history at all -- either no files, or nothing but NaN.
    """
    directory = feature_dir(provider_uri, symbol)
    if not directory.is_dir():
        return None
    closes = read_field(provider_uri, symbol, "close")
    factors = read_field(provider_uri, symbol, "factor")
    if closes is None or factors is None:
        return None

    c_first, close = closes
    f_first, factor = factors
    # Pair on the slot both fields actually reach; a lagging factor must not be
    # matched with a later close.
    last_c = c_first + len(close) - 1
    last_f = f_first + len(factor) - 1
    slot = min(last_c, last_f)

    dates = calendar(provider_uri)
    if not dates or c_first >= len(dates) or f_first >= len(dates):
        # A header past the calendar is a corrupt file; without this the oldest
        # value would be read as the latest and dated with an empty string.
        # Without the calendar a row cannot be dated, and an undated row would
        # silently disable the caller's already-stored check.
        return None
    while True:
        ci = slot - c_first
        fi = slot - f_first
        if ci < 0 or fi < 0:
            return None
        c = float(close[ci])
        f = float(factor[fi])
        if math.isfinite(c) and math.isfinite(f) and c > 0 and f > 0:
            date_str = dates[slot] if 0 <= slot < len(dates) else ""
            return PrevState(close=c, factor=f), date_str
        slot -= 1
