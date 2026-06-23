"""Unit tests for ashare_lab.paper.fees -- A-share fee calculation."""

from __future__ import annotations

import pytest

from ashare_lab.paper.fees import FeeResult, calculate_fees


class TestCalculateFees:
    def test_buy_small_notional_min_commission(self) -> None:
        """300 shares at 10 CNY = 3000 notional; commission floors at 5 CNY."""
        r = calculate_fees(3000.0, "buy")
        assert r.commission == pytest.approx(5.0)
        assert r.stamp == pytest.approx(0.0)
        assert r.transfer == pytest.approx(3000.0 * 0.00001)
        assert r.total == pytest.approx(5.0 + 0.03)

    def test_sell_small_notional(self) -> None:
        """Sell side adds stamp duty."""
        r = calculate_fees(3000.0, "sell")
        assert r.commission == pytest.approx(5.0)
        assert r.stamp == pytest.approx(3000.0 * 0.0005)
        assert r.transfer == pytest.approx(0.03)
        assert r.total == pytest.approx(5.0 + 1.5 + 0.03)

    def test_buy_large_notional_above_min(self) -> None:
        """10000 shares at 20 CNY = 200000; commission rate exceeds floor."""
        r = calculate_fees(200_000.0, "buy")
        assert r.commission == pytest.approx(50.0)
        assert r.stamp == pytest.approx(0.0)
        assert r.transfer == pytest.approx(2.0)
        assert r.total == pytest.approx(52.0)

    def test_sell_large_notional(self) -> None:
        """Sell: commission 50 + stamp 100 + transfer 2."""
        r = calculate_fees(200_000.0, "sell")
        assert r.commission == pytest.approx(50.0)
        assert r.stamp == pytest.approx(100.0)
        assert r.transfer == pytest.approx(2.0)
        assert r.total == pytest.approx(152.0)

    def test_zero_notional_min_commission(self) -> None:
        """Zero notional still gets the 5 CNY commission floor."""
        r = calculate_fees(0.0, "buy")
        assert r.commission == pytest.approx(5.0)
        assert r.stamp == pytest.approx(0.0)
        assert r.transfer == pytest.approx(0.0)
        assert r.total == pytest.approx(5.0)

    def test_custom_rates_override(self) -> None:
        """Custom rates replace defaults."""
        r = calculate_fees(
            10_000.0, "sell",
            commission_rate=0.001,
            min_commission=1.0,
            stamp_rate=0.001,
            transfer_rate=0.0001,
        )
        assert r.commission == pytest.approx(10.0)
        assert r.stamp == pytest.approx(10.0)
        assert r.transfer == pytest.approx(1.0)
        assert r.total == pytest.approx(21.0)


class TestFeeResultFrozen:
    def test_immutable(self) -> None:
        r = calculate_fees(1000.0, "buy")
        with pytest.raises(AttributeError):
            r.commission = 99.0  # type: ignore[misc]
