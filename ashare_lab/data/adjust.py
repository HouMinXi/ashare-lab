"""Keep one price series on one adjustment basis.

The qlib store holds forward-adjusted prices: ``close = raw * factor``, where
``factor`` is the cumulative corporate-action multiplier carried by the last
written row. A daily writer that appends the raw close with ``factor`` 1.0 puts
two bases into one series, and every return across that boundary is a corporate
action wearing the costume of a price move. For a high-dividend stock the fake
move is not subtle: a factor of 0.07 turns a flat day into a 1,300% gain.

Every check here is closed, not permissive. A NaN that slips through as a factor
lands in the store and the next run reads it back as "no history", so the symbol
restarts at factor 1.0 -- the defect this module exists to remove, seeded
permanently by a single bad field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A real A-share day rarely moves more than this after adjustment. The limit is
# 10% on the main boards and 20% on ChiNext and STAR, so 30% on a day with no
# corporate action is either a resumption after a long suspension, a new
# listing, or a data fault.
BREAK_RETURN = 0.30
# A broken write is a whole market event, not a handful of names: on 2026-10-09,
# 96% of symbols moved more than 30%. Refusing at 5% leaves room for a genuinely
# eventful day while still catching a basis change.
BREAK_SHARE = 0.05
# Below this the share is noise: ten symbols out of five thousand cannot
# distinguish a broken write from a bad day.
MIN_SYMBOLS_FOR_BREAK = 50

# Beyond this the ratio cannot be a dividend or a split; it is a fault.
MAX_IMPLIED_ACTION = 1.0

# A ratio this close to one is rounding, not a corporate action. Returning the
# previous factor unchanged keeps no-action days bit-identical, which matters
# because the paper pipeline records a corporate action whenever the factor it
# reads differs from yesterday's.
NO_ACTION_TOLERANCE = 1e-5


@dataclass(frozen=True)
class PrevState:
    """What the store already holds for one symbol."""

    close: float
    factor: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.close) and math.isfinite(self.factor)):
            raise ValueError(f"stored state must be finite, got {self}")
        if self.close <= 0 or self.factor <= 0:
            raise ValueError(f"stored state must be positive, got {self}")

    @property
    def raw_close(self) -> float:
        return self.close / self.factor


def next_factor(prev: PrevState, pre_close: float) -> float:
    """Adjustment factor for a new row, continuous with the previous one.

    ``pre_close`` is the previous close restated for any corporate action that
    took effect today (tushare's ``pre_close``, baostock's equivalent). When no
    action happened it equals yesterday's raw close and the factor is returned
    unchanged, bit for bit. When a dividend or split happened, the ratio is
    exactly the corporate action, which is what the factor must absorb.

    Args:
        prev: last stored row for the symbol.
        pre_close: previous close restated to today's terms.

    Returns:
        The factor to store with today's prices.

    Raises:
        ValueError: if either input is not a finite positive number.
    """
    if not math.isfinite(pre_close) or pre_close <= 0:
        raise ValueError(f"pre_close must be finite and positive, got {pre_close!r}")
    ratio = prev.raw_close / pre_close
    if abs(ratio - 1.0) > MAX_IMPLIED_ACTION:
        # A cash dividend moves the restated close by a few percent; a split is
        # already restated away. A ratio beyond that is a corrupt field or a
        # series that skipped sessions, and absorbing it would hide the fault
        # behind a smooth, permanently wrong level.
        raise ValueError(
            f"implied corporate action of {ratio - 1:+.1%} is implausible for one session"
        )
    if abs(ratio - 1.0) <= NO_ACTION_TOLERANCE:
        return prev.factor
    return prev.factor * ratio


def scale_prices(prices: dict[str, float], factor: float) -> dict[str, float]:
    """Apply the factor to every price field of one row.

    Open, high, low and close must share a basis; scaling only the close leaves
    rows where high < close, which upstream feature code can read as a zero-range
    day and downstream filters can drop.

    Raises:
        ValueError: if the factor is not finite and positive.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"factor must be finite and positive, got {factor!r}")
    return {field: value * factor for field, value in prices.items()}


def adjusted_return(prev: PrevState, close_adj: float) -> float:
    """Return between the stored row and a new adjusted close."""
    return close_adj / prev.close - 1.0


def systemic_break(returns: dict[str, float]) -> list[str]:
    """Symbols to blame when the whole batch looks wrong.

    Returns the offending symbols when the share of extreme moves is large
    enough to be a broken write rather than a handful of real events, otherwise
    an empty list. The caller aborts on a non-empty result: a batch that is
    wrong for a twentieth of the market must not be written, because the next
    run has no way to tell those rows from real prices.
    """
    usable = {s: r for s, r in returns.items() if math.isfinite(r)}
    if len(usable) < MIN_SYMBOLS_FOR_BREAK:
        return []
    bad = sorted(s for s, r in usable.items() if abs(r) > BREAK_RETURN)
    if len(bad) / len(usable) > BREAK_SHARE:
        return bad
    return []
