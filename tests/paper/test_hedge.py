"""Tests for the hedge sleeve module."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pytest

from ashare_lab.paper.hedge import (
    HedgeConfig,
    HedgeLeg,
    HedgeState,
    _fetch_hedge_prices,
    _load_hedge_config,
    compute_hedge_state,
    generate_hedge_orders,
    load_hedge_state,
    save_hedge_state,
    update_peak_nav,
)
from ashare_lab.paper.ledger import init_schema


def _default_legs() -> list[HedgeLeg]:
    return [
        HedgeLeg(symbol="511260", weight=0.60, leg_type="treasury_etf"),
        HedgeLeg(symbol="518880", weight=0.25, leg_type="gold_etf"),
        HedgeLeg(symbol="511990", weight=0.15, leg_type="money_market"),
    ]


def _default_config() -> HedgeConfig:
    return HedgeConfig(
        equity_ramp=[(0.05, 0.75), (0.10, 0.50), (0.15, 0.20)],
        legs=_default_legs(),
    )


# ---------------------------------------------------------------------------
# compute_hedge_state
# ---------------------------------------------------------------------------


def test_no_hedge_below_threshold():
    cfg = _default_config()
    state = compute_hedge_state(475_000, 500_000, 0, cfg)
    assert not state.active
    assert state.equity_target_pct == pytest.approx(1.0)
    assert state.hedge_target_pct == pytest.approx(0.0)


def test_activation_at_5pct():
    cfg = _default_config()
    state = compute_hedge_state(475_000, 500_000, 1, cfg)
    assert state.active
    assert state.equity_target_pct == pytest.approx(0.75)
    assert state.hedge_target_pct == pytest.approx(0.25)
    assert len(state.leg_allocations) == 3
    assert sum(state.leg_allocations.values()) == pytest.approx(0.25)


def test_linear_ramp_midpoint():
    cfg = _default_config()
    state = compute_hedge_state(462_500, 500_000, 1, cfg)
    assert state.active
    assert state.equity_target_pct == pytest.approx(0.625)
    assert state.hedge_target_pct == pytest.approx(0.375)


def test_max_protection():
    cfg = _default_config()
    state = compute_hedge_state(425_000, 500_000, 1, cfg)
    assert state.active
    assert state.equity_target_pct == pytest.approx(0.20)
    assert state.hedge_target_pct == pytest.approx(0.80)

    state2 = compute_hedge_state(400_000, 500_000, 1, cfg)
    assert state2.equity_target_pct == pytest.approx(0.20)


def test_recovery_exit():
    cfg = _default_config()
    # dd = (500000 - 485500) / 500000 = 0.029 < recovery_dd 0.03
    state = compute_hedge_state(485_500, 500_000, 11, cfg, prev_active=True,
                               prev_days_in_recovery=10)
    assert not state.active


def test_anti_whipsaw():
    cfg = _default_config()
    state = compute_hedge_state(485_000, 500_000, 5, cfg, prev_active=True)
    assert state.active
    assert state.days_in_hedge == 5


def test_peak_ratchet():
    assert update_peak_nav(280_000, 300_000) == 300_000
    assert update_peak_nav(310_000, 300_000) == 310_000


def test_inactive_equity_is_one():
    cfg = _default_config()
    state = compute_hedge_state(500_000, 500_000, 0, cfg)
    assert state.equity_target_pct == pytest.approx(1.0)


def test_boundary_exactly_activate_dd():
    cfg = _default_config()
    assert compute_hedge_state(475_000, 500_000, 1, cfg).active
    assert not compute_hedge_state(475_025, 500_000, 0, cfg).active


def test_prev_active_holds_days():
    cfg = _default_config()
    state = compute_hedge_state(485_000, 500_000, 5, cfg, prev_active=True)
    assert state.days_in_hedge == 5


def test_gap_risk_documented():
    assert "soft floor" in compute_hedge_state.__doc__.lower()


# ---------------------------------------------------------------------------
# generate_hedge_orders
# ---------------------------------------------------------------------------


def test_max_rebalance_cap():
    cfg = _default_config()
    state = HedgeState(
        active=True,
        drawdown_pct=0.20,
        equity_target_pct=0.20,
        hedge_target_pct=0.80,
        days_in_hedge=5, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={
            "511260": 0.48,
            "518880": 0.20,
            "511990": 0.12,
        },
    )
    prices = {
        "511260": {"close": 10.0},
        "518880": {"close": 20.0},
        "511990": {"close": 100.0},
    }
    orders = generate_hedge_orders({}, state, 500_000, prices, cfg)
    turnover = sum(
        o.target_qty * prices[o.symbol]["close"] for o in orders
    )
    assert turnover <= 150_000 + 1e-6


def test_hedge_buy_with_buying_halted():
    cfg = _default_config()
    state = HedgeState(
        active=True,
        drawdown_pct=0.05,
        equity_target_pct=0.75,
        hedge_target_pct=0.25,
        days_in_hedge=1, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={"511260": 0.15, "518880": 0.0625, "511990": 0.0375},
    )
    prices = {
        "511260": {"close": 10.0},
        "518880": {"close": 20.0},
        "511990": {"close": 100.0},
    }
    orders_halted = generate_hedge_orders(
        {}, state, 500_000, prices, cfg, buying_halted=True
    )
    orders_normal = generate_hedge_orders(
        {}, state, 500_000, prices, cfg, buying_halted=False
    )
    assert len(orders_halted) == len(orders_normal)
    assert all(o.side == "buy" for o in orders_halted)


def test_order_source_tagging():
    cfg = _default_config()
    state = HedgeState(
        active=True,
        drawdown_pct=0.05,
        equity_target_pct=0.75,
        hedge_target_pct=0.25,
        days_in_hedge=1, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={"511260": 0.15, "518880": 0.0625, "511990": 0.0375},
    )
    prices = {
        "511260": {"close": 10.0},
        "518880": {"close": 20.0},
        "511990": {"close": 100.0},
    }
    orders = generate_hedge_orders({}, state, 500_000, prices, cfg)
    assert all(o.source == "hedge" for o in orders)


def test_generate_orders_skips_missing_price():
    cfg = _default_config()
    state = HedgeState(
        active=True,
        drawdown_pct=0.05,
        equity_target_pct=0.75,
        hedge_target_pct=0.25,
        days_in_hedge=1, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={"511260": 0.25},
    )
    prices = {}
    orders = generate_hedge_orders({}, state, 500_000, prices, cfg)
    assert orders == []


def test_generate_orders_inactive_returns_empty():
    cfg = _default_config()
    state = HedgeState(
        active=False,
        drawdown_pct=0.0,
        equity_target_pct=1.0,
        hedge_target_pct=0.0,
        days_in_hedge=0, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={},
    )
    assert generate_hedge_orders({}, state, 500_000, {}, cfg) == []


# ---------------------------------------------------------------------------
# _load_hedge_config
# ---------------------------------------------------------------------------


def test_load_hedge_config():
    cfg = _load_hedge_config({
        "enabled": True,
        "activate_dd": 0.05,
        "equity_ramp": [[0.05, 0.75], [0.10, 0.50], [0.15, 0.20]],
        "legs": [
            {"symbol": "511260", "weight": 0.60, "leg_type": "treasury_etf"},
            {"symbol": "518880", "weight": 0.25, "leg_type": "gold_etf"},
            {"symbol": "511990", "weight": 0.15, "leg_type": "money_market"},
        ],
    })
    assert cfg.activate_dd == pytest.approx(0.05)
    assert len(cfg.legs) == 3


def test_empty_legs_raises():
    with pytest.raises(ValueError):
        _load_hedge_config({"legs": [], "equity_ramp": [[0.05, 1.0]]})


def test_leg_weights_sum_to_one():
    with pytest.raises(ValueError):
        _load_hedge_config({
            "equity_ramp": [[0.05, 0.75], [0.10, 0.50], [0.15, 0.20]],
            "legs": [
                {"symbol": "511260", "weight": 0.60, "leg_type": "treasury_etf"},
                {"symbol": "518880", "weight": 0.30, "leg_type": "gold_etf"},
            ],
        })


def test_invalid_leg_type_raises():
    with pytest.raises(ValueError):
        _load_hedge_config({
            "equity_ramp": [[0.05, 0.75], [0.10, 0.50], [0.15, 0.20]],
            "legs": [
                {"symbol": "511260", "weight": 1.0, "leg_type": "crypto"},
            ],
        })


def test_recovery_must_be_below_activation():
    with pytest.raises(ValueError):
        _load_hedge_config({
            "activate_dd": 0.05,
            "recovery_pct": 0.94,
            "equity_ramp": [[0.05, 0.75], [0.10, 0.50], [0.15, 0.20]],
            "legs": [{"symbol": "511260", "weight": 1.0, "leg_type": "treasury_etf"}],
        })


def test_ramp_first_breakpoint_must_match_activate():
    with pytest.raises(ValueError):
        _load_hedge_config({
            "activate_dd": 0.05,
            "equity_ramp": [[0.06, 0.75], [0.10, 0.50], [0.15, 0.20]],
            "legs": [{"symbol": "511260", "weight": 1.0, "leg_type": "treasury_etf"}],
        })


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def test_save_and_load_hedge_state():
    conn = _mem_conn()
    state = HedgeState(
        active=True,
        drawdown_pct=0.05,
        equity_target_pct=0.75,
        hedge_target_pct=0.25,
        days_in_hedge=1, days_in_recovery=0,
        peak_nav=500_000,
        leg_allocations={"511260": 0.15},
    )
    save_hedge_state(conn, "2026-01-02", state)
    loaded = load_hedge_state(conn, "2026-01-02")
    assert loaded is not None
    assert loaded.active is True
    assert loaded.equity_target_pct == pytest.approx(0.75)
    assert loaded.leg_allocations == {"511260": 0.15}


def test_load_missing_hedge_state_returns_none():
    conn = _mem_conn()
    assert load_hedge_state(conn, "2026-01-02") is None


# ---------------------------------------------------------------------------
# _fetch_hedge_prices
# ---------------------------------------------------------------------------


@dataclass
class _MockCtx:
    prices: dict = field(default_factory=dict)


def test_fetch_hedge_prices_writes_close(monkeypatch):
    ctx = _MockCtx()

    def _fake_spot():
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame({
            "代码": ["511260", "518880"],
            "最新价": [10.5, 20.0],
        })

    monkeypatch.setattr("akshare.fund_etf_spot_em", _fake_spot)
    _fetch_hedge_prices(ctx, ["511260"])
    assert "511260" in ctx.prices
    assert ctx.prices["511260"]["close"] == pytest.approx(10.5)
    assert ctx.prices["511260"]["factor"] == pytest.approx(1.0)


def test_fetch_hedge_prices_skips_existing(monkeypatch):
    ctx = _MockCtx(prices={"511260": {"close": 99.0}})

    def _fake_spot():
        raise AssertionError("should not be called")

    monkeypatch.setattr("akshare.fund_etf_spot_em", _fake_spot)
    _fetch_hedge_prices(ctx, ["511260"])
    assert ctx.prices["511260"]["close"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_n_leg_config_driven():
    cfg = HedgeConfig(
        equity_ramp=[(0.05, 0.75), (0.10, 0.50), (0.15, 0.20)],
        legs=[
            HedgeLeg(symbol="511260", weight=0.70, leg_type="treasury_etf"),
            HedgeLeg(symbol="518880", weight=0.30, leg_type="gold_etf"),
        ],
    )
    state = compute_hedge_state(475_000, 500_000, 1, cfg)
    assert len(state.leg_allocations) == 2


# ---------------------------------------------------------------------------
# Fix #4: Hedge volume must not trigger _is_suspended
# ---------------------------------------------------------------------------


def test_hedge_volume_not_suspended():
    """ETF prices injected by _fetch_hedge_prices must have large finite
    volume so _is_suspended() returns False and hedge orders are never
    carried forward as suspended.  Also verifies _cap_fill_by_volume
    does not crash on the value."""
    from ashare_lab.paper.engine_settle import _is_suspended, _cap_fill_by_volume

    ctx = _MockCtx(prices={})

    def _fake_spot():
        import pandas as pd
        return pd.DataFrame({
            "代码": ["511260"],
            "最新价": [10.5],
        })

    import akshare
    original = akshare.fund_etf_spot_em
    akshare.fund_etf_spot_em = _fake_spot
    try:
        _fetch_hedge_prices(ctx, ["511260"])
    finally:
        akshare.fund_etf_spot_em = original

    vol = ctx.prices["511260"]["volume"]
    assert vol > 0 and vol < float("inf"), f"hedge volume must be finite positive, got {vol}"
    assert not _is_suspended(vol), "hedge ETF must not be treated as suspended"
    # Must not crash in cap_fill (inf would OverflowError here)
    assert _cap_fill_by_volume(500, vol, "buy", 0.05) == 500
