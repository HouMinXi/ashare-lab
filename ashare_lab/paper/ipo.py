"""IPO subscription deterministic simulation for A-share paper trading.

Implements:
- Board type detection from stock code prefix (main/star/chinext/bse).
- Deterministic subscription simulation based on expected value threshold.
- Board-specific sell-date logic: main board sells on first non-limit-up
  day after a limit-up streak; STAR/ChiNext sells on listing day 5.

No cash freeze: post-2016 reform removed the subscription deposit
requirement.  Shares appear in the portfolio on the listing date if won.
"""

from __future__ import annotations


# -- Board-specific limit-up thresholds for IPO sell decisions ----------

_BOARD_LIMIT_THRESHOLDS: dict[str, float] = {
    "main": 0.099,
    "star": 0.199,
    "chinext": 0.199,
}


def detect_board_type(symbol: str) -> str:
    """Classify a stock symbol into its board type.

    Strips any exchange prefix (SH/SZ/sh/sz) and checks the 6-digit
    code prefix.

    Returns one of: "star", "chinext", "bse", "main".
    """
    code = symbol
    # Strip 2-char exchange prefix (SH/SZ) if present
    if len(code) > 6:
        code = code[-6:]

    if code.startswith(("688", "689")):
        return "star"
    if code.startswith("300"):
        return "chinext"
    if code.startswith(("83", "87", "43")):
        return "bse"
    return "main"


def check_ipo_subscription(
    ceiling_lots: int,
    win_rate: float,
    shares_per_lot: int = 500,
) -> tuple[bool, int]:
    """Deterministic IPO subscription simulation.

    expected_value = ceiling_lots * win_rate.  If >= 0.5 the subscription
    is considered won; otherwise it is not.

    At 300K capital, ceiling_lots ~ 15 and typical win_rate ~ 0.02-0.05%,
    so expected_value << 0.5 and the result is always (False, 0).  The
    framework exists for future capital scaling.

    No cash is frozen (post-2016 reform).

    Parameters
    ----------
    ceiling_lots : int
        Maximum lots the account can subscribe for.
    win_rate : float
        Lottery win probability (0.0 to 1.0).
    shares_per_lot : int
        Shares per lot (default 500 for A-share IPO).

    Returns
    -------
    (won, shares) : tuple[bool, int]
        won=True and shares=shares_per_lot if expected_value >= 0.5,
        otherwise (False, 0).
    """
    expected_value = ceiling_lots * win_rate
    if expected_value >= 0.5:
        return True, shares_per_lot
    return False, 0


def determine_ipo_sell_date(
    symbol: str,
    listing_date: str,
    daily_changes: list[tuple[str, float]],
) -> str | None:
    """Determine the IPO sell date based on board-specific rules.

    Parameters
    ----------
    symbol : str
        Stock symbol (may include exchange prefix).
    listing_date : str
        The IPO listing date (YYYY-MM-DD).
    daily_changes : list of (date_str, pct_change)
        Trading bars from listing day onward, 0-indexed:
        [0] = listing day 1, [1] = day 2, etc.

    Returns
    -------
    str or None
        The date to sell.  None when insufficient data to decide
        (active limit-up streak for main board; fewer than 5 bars
        for STAR/ChiNext).

    Board rules:
    - STAR / ChiNext: sell on day 5 (index 4).  Returns None if
      fewer than 5 bars are provided.
    - Main board: sell on the first non-limit-up day.  If the stock
      never hits limit-up on listing day, sell on listing_date itself.
      Returns None only if all provided days are limit-up (streak
      not yet broken, need more data).
    """
    if not daily_changes:
        return None

    board = detect_board_type(symbol)
    threshold = _BOARD_LIMIT_THRESHOLDS.get(board, 0.099)

    # STAR / ChiNext: fixed day-5 sell regardless of price action
    if board in ("star", "chinext"):
        if len(daily_changes) >= 5:
            return daily_changes[4][0]
        return None

    # Main board: sell on first non-limit-up day after streak
    has_streak = False
    for date_str, change in daily_changes:
        if change >= threshold:
            has_streak = True
            continue
        if has_streak:
            # Streak broken -- sell on this day
            return date_str
        # First day is not limit-up and no streak yet -- sell immediately
        return date_str

    # All provided days are limit-up -- need more data
    return None
