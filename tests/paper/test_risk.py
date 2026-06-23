"""Unit tests for the 7-dimension risk control framework.

All tests are pure: no DB, no qlib, no external data.  Explicit
numeric inputs verify boundary behaviour for each risk dimension.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

from ashare_lab.paper.risk import (
    RiskCheckResult,
    check_concentration,
    check_daily_loss,
    check_drawdown_breaker,
    check_industry_concentration,
    check_market_regime,
    check_soft_drawdown,
    check_trailing_stop,
    manage_trailing_cooldown,
    run_all_risk_checks,
)


# -----------------------------------------------------------------------
# check_drawdown_breaker
# -----------------------------------------------------------------------


class TestDrawdownBreaker:
    def test_drawdown_above_threshold_halts(self):
        # 16% drawdown > 15% threshold
        assert check_drawdown_breaker(100_000, 84_000, 0.15) is True

    def test_drawdown_below_threshold_allows(self):
        # 14% drawdown < 15% threshold
        assert check_drawdown_breaker(100_000, 86_000, 0.15) is False

    def test_drawdown_exactly_at_threshold(self):
        # 15% drawdown == threshold -- not breached (> not >=)
        assert check_drawdown_breaker(100_000, 85_000, 0.15) is False

    def test_zero_peak_nav_returns_false(self):
        # Defensive: peak_nav=0 should not crash
        assert check_drawdown_breaker(0, 50_000, 0.15) is False


# -----------------------------------------------------------------------
# check_daily_loss
# -----------------------------------------------------------------------


class TestDailyLoss:
    def test_loss_above_threshold(self):
        # 3.5% > 3%
        assert check_daily_loss(100_000, 96_500, 0.03) is True

    def test_loss_below_threshold(self):
        # 2.5% < 3%
        assert check_daily_loss(100_000, 97_500, 0.03) is False

    def test_yesterday_nav_zero_returns_false(self):
        # Day 1: pipeline passes 0.0, should return False
        assert check_daily_loss(0.0, 300_000, 0.03) is False

    def test_yesterday_nav_negative_returns_false(self):
        assert check_daily_loss(-1.0, 300_000, 0.03) is False


# -----------------------------------------------------------------------
# check_concentration
# -----------------------------------------------------------------------


class TestConcentration:
    def test_over_cap_returns_excess(self):
        over, excess = check_concentration(16_000, 100_000, 0.15)
        assert over is True
        assert excess == pytest.approx(1_000.0)

    def test_under_cap_returns_zero(self):
        over, excess = check_concentration(14_000, 100_000, 0.15)
        assert over is False
        assert excess == 0.0

    def test_exactly_at_cap(self):
        # 15% == cap -- not over (> not >=)
        over, excess = check_concentration(15_000, 100_000, 0.15)
        assert over is False
        assert excess == 0.0


# -----------------------------------------------------------------------
# check_market_regime
# -----------------------------------------------------------------------


class TestMarketRegime:
    def test_decline_triggers_halt(self):
        # 11 closes: start=1000, end=910 -> cumulative return = -9% > -8%
        closes = [1000.0] + [1000.0 * (1 - 0.01)] * 10
        # Actually construct so return = (end/start) - 1 = -9.6%
        closes = [1000.0]
        price = 1000.0
        for _ in range(10):
            price *= (1 - 0.01)
            closes.append(price)
        # cumulative = closes[-1]/closes[0] - 1 ~ -9.56%
        assert check_market_regime(closes, 0.08, 10) is True

    def test_positive_returns_no_halt(self):
        closes = [1000.0]
        price = 1000.0
        for _ in range(10):
            price *= 1.005
            closes.append(price)
        assert check_market_regime(closes, 0.08, 10) is False

    def test_insufficient_history_returns_false(self):
        # Need 11 for lookback=10, give only 5
        assert check_market_regime([100.0] * 5, 0.08, 10) is False

    def test_slices_to_window(self):
        # Give 20 data points but only last 11 should matter
        # First 10: rising, last 11: sharp decline
        rising = [1000.0 + i * 10 for i in range(10)]
        declining = [rising[-1]]
        price = declining[0]
        for _ in range(10):
            price *= 0.985
            declining.append(price)
        all_closes = rising + declining
        # Window = last 11 = declining portion
        assert check_market_regime(all_closes, 0.08, 10) is True


# -----------------------------------------------------------------------
# check_trailing_stop
# -----------------------------------------------------------------------


class TestTrailingStop:
    def test_decline_triggers_sell(self):
        # 21% > 20%
        assert check_trailing_stop(100.0, 79.0, 0.20) is True

    def test_small_decline_no_trigger(self):
        # 10% < 20%
        assert check_trailing_stop(100.0, 90.0, 0.20) is False

    def test_exactly_at_threshold(self):
        # 20% == threshold -- not triggered (> not >=)
        assert check_trailing_stop(100.0, 80.0, 0.20) is False


# -----------------------------------------------------------------------
# manage_trailing_cooldown
# -----------------------------------------------------------------------


class TestManageTrailingCooldown:
    def test_removes_expired_entries(self):
        cd = {
            "SH600001": {
                "cooldown_until": "2025-01-05",
                "holding_high": 10.0,
            },
        }
        result = manage_trailing_cooldown(cd, "2025-01-06")
        assert "SH600001" not in result

    def test_retains_active_entries(self):
        cd = {
            "SH600001": {
                "cooldown_until": "2025-01-10",
                "holding_high": 10.0,
            },
        }
        result = manage_trailing_cooldown(cd, "2025-01-06")
        assert "SH600001" in result

    def test_boundary_cooldown_until_equals_trade_date_retained(self):
        """R-21: cooldown_until == trade_date is still active (>=)."""
        cd = {
            "SH600001": {
                "cooldown_until": "2025-01-10",
                "holding_high": 10.0,
            },
        }
        result = manage_trailing_cooldown(cd, "2025-01-10")
        assert "SH600001" in result


# -----------------------------------------------------------------------
# check_industry_concentration
# -----------------------------------------------------------------------


class TestIndustryConcentration:
    def test_over_cap_blocked(self):
        positions = {"Manufacturing": 32_000.0, "Finance": 10_000.0}
        blocked = check_industry_concentration(positions, 100_000, 0.30)
        assert "Manufacturing" in blocked
        assert "Finance" not in blocked

    def test_under_cap_not_blocked(self):
        positions = {"Manufacturing": 28_000.0}
        blocked = check_industry_concentration(positions, 100_000, 0.30)
        assert len(blocked) == 0


# -----------------------------------------------------------------------
# check_soft_drawdown
# -----------------------------------------------------------------------


class TestSoftDrawdown:
    def test_drawdown_triggers_reduction(self):
        # 11% > 10%, not currently reduced
        result = check_soft_drawdown(
            100_000, 89_000, False, 0.10, 0.95, 15, 7
        )
        assert result == 7

    def test_small_drawdown_no_reduction(self):
        # 4% < 10%, not currently reduced
        result = check_soft_drawdown(
            100_000, 96_000, False, 0.10, 0.95, 15, 7
        )
        assert result is None

    def test_recovery_restores_topk(self):
        # Currently reduced, at 95% of peak -> restore
        result = check_soft_drawdown(
            100_000, 95_000, True, 0.10, 0.95, 15, 7
        )
        assert result is None

    def test_still_below_recovery_stays_reduced(self):
        # Currently reduced, at 90% of peak -> stay reduced
        result = check_soft_drawdown(
            100_000, 90_000, True, 0.10, 0.95, 15, 7
        )
        assert result == 7


# -----------------------------------------------------------------------
# blocked_rebuys boundary (R-21 regression)
# -----------------------------------------------------------------------


class TestBlockedRebuysBoundary:
    """Verify R-21: cooldown_until == trade_date is STILL blocked."""

    def _make_config(self):
        return {
            "drawdown_hard": 0.15,
            "daily_loss": 0.03,
            "concentration": 0.15,
            "market_regime_decline": 0.08,
            "market_regime_days": 10,
            "trailing_stop": 0.20,
            "trailing_cooldown_days": 10,
            "industry_cap": 0.30,
            "soft_drawdown": 0.10,
            "soft_drawdown_recovery": 0.95,
            "default_topk": 15,
            "reduced_topk": 7,
        }

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_cooldown_boundary_blocked(self, mock_ntd):
        """cooldown_until == trade_date -> symbol in blocked_rebuys."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        nav_history = [{"total_nav": 300_000.0}]
        cooldown_dict = {
            "SH600001": {
                "cooldown_until": "2025-01-15",
                "holding_high": 10.0,
            },
        }
        result = run_all_risk_checks(
            nav_history=nav_history,
            yesterday_nav=300_000.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict=cooldown_dict,
            cash=300_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        assert "SH600001" in result.blocked_rebuys

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_cooldown_expired_not_blocked(self, mock_ntd):
        """cooldown_until < trade_date -> symbol NOT in blocked_rebuys."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        nav_history = [{"total_nav": 300_000.0}]
        cooldown_dict = {
            "SH600001": {
                "cooldown_until": "2025-01-14",
                "holding_high": 10.0,
            },
        }
        result = run_all_risk_checks(
            nav_history=nav_history,
            yesterday_nav=300_000.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict=cooldown_dict,
            cash=300_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        assert "SH600001" not in result.blocked_rebuys


# -----------------------------------------------------------------------
# run_all_risk_checks aggregator
# -----------------------------------------------------------------------


class TestRunAllRiskChecks:
    def _make_config(self):
        return {
            "drawdown_hard": 0.15,
            "daily_loss": 0.03,
            "concentration": 0.15,
            "market_regime_decline": 0.08,
            "market_regime_days": 10,
            "trailing_stop": 0.20,
            "trailing_cooldown_days": 10,
            "industry_cap": 0.30,
            "soft_drawdown": 0.10,
            "soft_drawdown_recovery": 0.95,
            "default_topk": 15,
            "reduced_topk": 7,
        }

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_aggregation_no_issues(self, mock_ntd):
        """Healthy portfolio: no halts, no forced sells."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        nav_history = [{"total_nav": 300_000.0}]
        result = run_all_risk_checks(
            nav_history=nav_history,
            yesterday_nav=300_000.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict={},
            cash=300_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        assert isinstance(result, RiskCheckResult)
        assert result.buying_halted is False
        assert result.forced_sells == {}
        assert result.blocked_industries == set()
        assert result.blocked_rebuys == set()
        assert result.topk_override is None
        assert result.cooldown_entries == {}

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_trailing_stop_supersedes_concentration(self, mock_ntd):
        """F-D: trailing stop full-position exit overwrites concentration
        partial via dict assignment, not accumulation."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        config = self._make_config()
        nav_history = [{"total_nav": 300_000.0}]
        positions = {
            "SH600001": {
                "qty": 1000,
                "avg_cost": 50.0,
                "market_value": 50_000.0,
                "holding_high": 100.0,
            },
        }
        prices = {
            "SH600001": {"close": 50.0, "volume": 1000000, "change": -0.1},
        }
        result = run_all_risk_checks(
            nav_history=nav_history,
            yesterday_nav=300_000.0,
            current_positions=positions,
            current_prices=prices,
            csi1000_closes=[],
            industry_map={},
            cooldown_dict={},
            cash=300_000.0,
            is_soft_reduced=False,
            config=config,
            trade_date="2025-01-15",
        )
        # Trailing stop triggers (50% decline from high 100 to close 50)
        # Should be full position qty (1000), not partial concentration excess
        assert result.forced_sells.get("SH600001") == 1000
        assert "SH600001" in result.cooldown_entries

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_day1_no_daily_loss_halt(self, mock_ntd):
        """Day 1: yesterday_nav=0.0 -> check_daily_loss returns False."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        result = run_all_risk_checks(
            nav_history=[],
            yesterday_nav=0.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict={},
            cash=300_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        assert result.buying_halted is False

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_drawdown_halts_buying(self, mock_ntd):
        """Hard drawdown > 15% halts all buying."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        nav_history = [{"total_nav": 300_000.0}]
        result = run_all_risk_checks(
            nav_history=nav_history,
            yesterday_nav=280_000.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict={},
            cash=240_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        # current_nav = 240000, peak_nav = 300000, drawdown = 20% > 15%
        assert result.buying_halted is True

    @patch("ashare_lab.paper.risk.next_trading_day")
    def test_result_is_frozen(self, mock_ntd):
        """RiskCheckResult is frozen -- immutable."""
        mock_ntd.side_effect = lambda d: d + dt.timedelta(days=1)

        result = run_all_risk_checks(
            nav_history=[{"total_nav": 300_000.0}],
            yesterday_nav=300_000.0,
            current_positions={},
            current_prices={},
            csi1000_closes=[],
            industry_map={},
            cooldown_dict={},
            cash=300_000.0,
            is_soft_reduced=False,
            config=self._make_config(),
            trade_date="2025-01-15",
        )
        with pytest.raises(AttributeError):
            result.buying_halted = True
