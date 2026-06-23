"""Tests for IPO subscription simulation and board-specific sell logic."""

from ashare_lab.paper.ipo import (
    check_ipo_subscription,
    detect_board_type,
    determine_ipo_sell_date,
)


# ---------------------------------------------------------------------------
# detect_board_type
# ---------------------------------------------------------------------------


class TestDetectBoardType:
    """Board classification from stock code prefix."""

    def test_star_688(self):
        assert detect_board_type("688001") == "star"

    def test_star_689(self):
        assert detect_board_type("689001") == "star"

    def test_star_with_prefix(self):
        assert detect_board_type("SH688001") == "star"

    def test_chinext(self):
        assert detect_board_type("300123") == "chinext"

    def test_chinext_with_prefix(self):
        assert detect_board_type("SZ300123") == "chinext"

    def test_main_600(self):
        assert detect_board_type("600000") == "main"

    def test_main_601(self):
        assert detect_board_type("601398") == "main"

    def test_main_with_prefix(self):
        assert detect_board_type("SH600000") == "main"

    def test_bse_83(self):
        assert detect_board_type("830001") == "bse"

    def test_main_000(self):
        assert detect_board_type("000001") == "main"


# ---------------------------------------------------------------------------
# check_ipo_subscription
# ---------------------------------------------------------------------------


class TestCheckIpoSubscription:
    """Deterministic subscription simulation."""

    def test_win_above_threshold(self):
        # ceiling_lots=15, win_rate=0.04 -> ev=0.6 >= 0.5 -> win
        won, shares = check_ipo_subscription(15, 0.04)
        assert won is True
        assert shares == 500

    def test_lose_below_threshold(self):
        # ceiling_lots=10, win_rate=0.03 -> ev=0.3 < 0.5 -> lose
        won, shares = check_ipo_subscription(10, 0.03)
        assert won is False
        assert shares == 0

    def test_realistic_300k_account(self):
        # 300K account: ceiling_lots~15, win_rate~0.0003 -> ev<<0.5
        won, shares = check_ipo_subscription(15, 0.0003)
        assert won is False
        assert shares == 0

    def test_exact_threshold(self):
        # ev = 0.5 exactly -> win (>= 0.5)
        won, shares = check_ipo_subscription(10, 0.05)
        assert won is True
        assert shares == 500

    def test_no_cash_freeze(self):
        # Subscription returns only (won, shares) -- no cash deduction
        # This is a design test: the function signature has no cash param
        import inspect

        sig = inspect.signature(check_ipo_subscription)
        param_names = set(sig.parameters.keys())
        assert "cash" not in param_names


# ---------------------------------------------------------------------------
# determine_ipo_sell_date -- main board
# ---------------------------------------------------------------------------


class TestSellDateMainBoard:
    """Main board: sell on first non-limit-up day after streak."""

    def test_consecutive_then_open(self):
        # 3 limit-up days, then open on d4
        changes = [
            ("2025-01-10", 0.10),   # limit-up (>= 0.099)
            ("2025-01-13", 0.10),
            ("2025-01-14", 0.099),  # exactly at threshold = limit-up
            ("2025-01-15", 0.05),   # open board -> sell here
        ]
        result = determine_ipo_sell_date("SH601000", "2025-01-10", changes)
        assert result == "2025-01-15"

    def test_no_streak_sell_immediately(self):
        # First day not limit-up -> sell on listing date
        changes = [
            ("2025-01-10", 0.05),
            ("2025-01-13", 0.08),
        ]
        result = determine_ipo_sell_date("600001", "2025-01-10", changes)
        assert result == "2025-01-10"

    def test_all_limit_up_returns_none(self):
        # All days limit-up -> need more data
        changes = [
            ("2025-01-10", 0.10),
            ("2025-01-13", 0.10),
        ]
        result = determine_ipo_sell_date("600001", "2025-01-10", changes)
        assert result is None

    def test_empty_changes_returns_none(self):
        result = determine_ipo_sell_date("600001", "2025-01-10", [])
        assert result is None

    def test_uses_main_threshold_0099(self):
        # 0.098 is below 0.099 -> not limit-up on main board
        changes = [("2025-01-10", 0.098)]
        result = determine_ipo_sell_date("600001", "2025-01-10", changes)
        assert result == "2025-01-10"


# ---------------------------------------------------------------------------
# determine_ipo_sell_date -- STAR / ChiNext
# ---------------------------------------------------------------------------


class TestSellDateStarChinext:
    """STAR and ChiNext: sell on day 5 regardless of price action."""

    def test_star_sell_day5(self):
        changes = [
            ("2025-01-10", 0.20),
            ("2025-01-13", 0.15),
            ("2025-01-14", 0.10),
            ("2025-01-15", 0.05),
            ("2025-01-16", 0.03),  # day 5 (index 4) -> sell
        ]
        result = determine_ipo_sell_date("SH688001", "2025-01-10", changes)
        assert result == "2025-01-16"

    def test_chinext_sell_day5(self):
        changes = [
            ("2025-01-10", 0.20),
            ("2025-01-13", 0.15),
            ("2025-01-14", 0.10),
            ("2025-01-15", 0.05),
            ("2025-01-16", 0.08),
        ]
        result = determine_ipo_sell_date("SZ300456", "2025-01-10", changes)
        assert result == "2025-01-16"

    def test_star_insufficient_bars_returns_none(self):
        # Only 4 bars for STAR -> None (need 5)
        changes = [
            ("2025-01-10", 0.20),
            ("2025-01-13", 0.15),
            ("2025-01-14", 0.10),
            ("2025-01-15", 0.05),
        ]
        result = determine_ipo_sell_date("688001", "2025-01-10", changes)
        assert result is None

    def test_star_uses_0199_not_0099(self):
        # STAR with 0.15 change -- below 0.199 threshold.
        # Should still sell on day 5 (STAR ignores limit-up logic).
        changes = [
            ("2025-01-10", 0.15),
            ("2025-01-13", 0.15),
            ("2025-01-14", 0.15),
            ("2025-01-15", 0.15),
            ("2025-01-16", 0.15),
        ]
        result = determine_ipo_sell_date("SH688001", "2025-01-10", changes)
        assert result == "2025-01-16"

    def test_chinext_single_bar_returns_none(self):
        changes = [("2025-01-10", 0.20)]
        result = determine_ipo_sell_date("300001", "2025-01-10", changes)
        assert result is None
