"""Bug-inject tests for hedge sleeve.

Each inject: modify code/config → assert test FAILS → revert → assert PASS.
Run manually: pytest tests/paper/test_hedge_inject.py -v
"""
import copy
import pytest
import yaml
from ashare_lab.paper.hedge import (
    HedgeConfig, HedgeLeg, _load_hedge_config, _interpolate_ramp,
    compute_hedge_state,
)


def _default_config():
    with open("configs/baseline.yaml") as f:
        return _load_hedge_config(yaml.safe_load(f)["paper"]["hedge"])


def _make_legs():
    return [
        HedgeLeg("511260", 0.60, "treasury_etf"),
        HedgeLeg("518880", 0.25, "gold_etf"),
        HedgeLeg("511990", 0.15, "money_market"),
    ]


# === Inject 1: ramp non-increasing ===
@pytest.mark.xfail(strict=True, reason="inject sentinel: bad ramp produces wrong equity")
def test_inject_1_bad_ramp_produces_wrong_equity():
    """INJECT: bypass validation, construct bad ramp directly.
    bad ramp [(0.03,0.20),(0.15,0.95)] → DD=8% → equity=0.5125 ≠ 0.6375.
    """
    bad_cfg = HedgeConfig(
        equity_ramp=[(0.03, 0.20), (0.15, 0.95)],
        legs=_make_legs(), activate_dd=0.03, min_equity_pct=0.20,
        anti_whipsaw_days=10, recovery_pct=0.98,
        max_single_day_rebalance=0.30, min_delta_pct=0.005,
    )
    result = _interpolate_ramp(0.08, bad_cfg.equity_ramp,
                               bad_cfg.equity_ramp[0][1], bad_cfg.min_equity_pct)
    assert result == 0.6375, f"INJECT CAUGHT: bad ramp gave {result} instead of 0.6375"


def test_revert_1_valid_ramp():
    cfg = _default_config()
    result = _interpolate_ramp(0.08, cfg.equity_ramp,
                               cfg.equity_ramp[0][1], cfg.min_equity_pct)
    assert abs(result - 0.6375) < 0.001


# === Inject 2: anti-whipsaw bypass ===
@pytest.mark.xfail(strict=True, reason="inject sentinel: no whipsaw allows early exit")
def test_inject_2_no_whipsaw_exits_early():
    """INJECT: anti_whipsaw_days=0 allows immediate exit.
    DD=0.6%, days_in_recovery=1 >= anti_whipsaw_days=0 → exits.
    """
    cfg = _default_config()
    bad_cfg = HedgeConfig(
        equity_ramp=cfg.equity_ramp, legs=cfg.legs, activate_dd=cfg.activate_dd,
        min_equity_pct=cfg.min_equity_pct, anti_whipsaw_days=0,
        recovery_pct=cfg.recovery_pct,
        max_single_day_rebalance=cfg.max_single_day_rebalance,
        min_delta_pct=cfg.min_delta_pct,
    )
    state = compute_hedge_state(497_000, 500_000, 1, bad_cfg,
                                prev_active=True, prev_days_in_recovery=1)
    assert state.active, "INJECT CAUGHT: exited at days_in_recovery=1 with anti_whipsaw_days=0"


def test_revert_2_whipsaw_works():
    cfg = _default_config()
    state = compute_hedge_state(497_000, 500_000, 5, cfg,
                                prev_active=True, prev_days_in_recovery=5)
    assert state.active


# === Inject 3: wrong activation threshold ===
@pytest.mark.xfail(strict=True, reason="inject sentinel: wrong threshold blocks activation")
def test_inject_3_wrong_threshold_blocks_activation():
    """INJECT: activate_dd=0.05, DD=4%, cold start.
    Caller sees dd < activate_dd → days=0 → inactive.
    Correct threshold (0.03) would set days=1 → active.
    """
    NAV = 480_000  # DD = 4%
    cfg = _default_config()
    bad_cfg = HedgeConfig(
        equity_ramp=[(0.05, 0.95), (0.15, 0.20)],
        legs=_make_legs(), activate_dd=0.05, min_equity_pct=0.20,
        anti_whipsaw_days=10, recovery_pct=0.98,
        max_single_day_rebalance=0.30, min_delta_pct=0.005,
    )
    dd = (500_000 - NAV) / 500_000
    days = 1 if dd >= bad_cfg.activate_dd else 0
    state = compute_hedge_state(NAV, 500_000, days, bad_cfg, prev_active=False)
    assert state.active, f"INJECT CAUGHT: DD=4%, days={days} inactive with activate_dd=0.05"


def test_revert_3_correct_threshold():
    cfg = _default_config()
    state = compute_hedge_state(480_000, 500_000, 1, cfg, prev_active=False)
    assert state.active
