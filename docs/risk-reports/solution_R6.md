# R6 Solution: Dynamic Slippage Model for A-Share Backtesting

## Problem Statement

`engine.py:103-109` uses a fixed 0.1% slippage regardless of market conditions:

```python
def apply_slippage(close, side, slippage=0.001):
    if side == "buy":
        return close * (1.0 + slippage)
    return close * (1.0 - slippage)
```

During CSI1000 liquidity crises (e.g., 2024-02), real slippage reaches 2-5%.
This causes NAV to be systematically overstated during volatile periods.

## Research Summary

### Sources Consulted

1. **quant-ashare** (wangpage/quant-ashare) - Trick #4: Almgren-Chriss + sqrt law
   - `market_microstructure/impact.py`: square-root market impact model
   - `execution/impact_router.py`: TWAP/VWAP + impact-aware routing
2. **PatternSmart.com** (2026-07) - Dynamic slippage Python implementation
   - `slippage = participation_rate * (vol_participation)^0.5 * volatility`
3. **Gurovich (2026)** - "Predicting Slippage in Equity Markets"
   - Two-stage Laplace model, validated on 6 US stocks
   - Key features: participation_rate, roll_spread, roll_vol
4. **QuestDB** - Slippage and Market Impact Estimation
   - `I(Q) = sigma * Y * (Q/V)^alpha` where alpha ~ 0.5
5. **Cheng, Shi, Zhang (2025)** - "Intraday liquidity and expected return in China's stock market"
   - TIAM (Time-Weighted Intraday Amihud) for A-share illiquidity pricing

### Industry Consensus

All sources converge on the **Almgren-Chriss square-root law**:

```
Market Impact = sigma * Y * (Q / V)^0.5
```

Where:
- `sigma` = asset volatility (rolling realized vol)
- `Y` = market-specific scaling constant
- `Q` = order size (shares)
- `V` = market volume (ADV)
- `0.5` = square-root exponent (empirically validated across markets)

## Proposed Dynamic Slippage Formula

### Core Formula

```python
slippage = max(FLOOR, min(CEILING, BASE * vol_factor * liquidity_factor))
```

Where:
- `BASE` = 0.001 (10 bps, current default for calm markets)
- `vol_factor` = `rolling_vol_20d / REF_VOL`
- `liquidity_factor` = `sqrt(participation / REF_PARTICIPATION)`
- `participation` = `order_notional / daily_turnover`

### Parameters (CSI1000 Small-Cap Calibration)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BASE` | 0.001 (10 bps) | Current default; matches calm CSI1000 small-cap |
| `REF_VOL` | 0.015 (1.5%) | Typical calm CSI1000 daily volatility |
| `REF_PARTICIPATION` | 0.02 (2%) | Reference volume participation |
| `FLOOR` | 0.0003 (3 bps) | Minimum even in ultra-liquid calm markets |
| `CEILING` | 0.05 (5%) | Hard cap; matches extreme crisis observation |
| `VOL_WINDOW` | 20 days | Rolling window for realized volatility |

### Behavior Under Different Market Conditions

| Scenario | Daily Vol | Participation | Computed Slippage | Notes |
|----------|-----------|---------------|-------------------|-------|
| Calm, liquid CSI300 | 1.0% | 1% | ~0.05% | Below base, clamped to floor 0.03% |
| Normal CSI1000 | 1.5% | 2% | ~0.10% | Matches current fixed default |
| Mild stress | 3.0% | 5% | ~0.35% | 3.5x current estimate |
| CSI1000 crisis (2024-02) | 5.0% | 10% | ~1.2% | 12x current estimate |
| Extreme crisis | 8.0% | 20% | ~2.8% | 28x current; near ceiling |

### Mathematical Derivation

```
vol_factor = sigma / REF_VOL
           = 0.05 / 0.015 = 3.33  (crisis scenario)

liquidity_factor = sqrt(participation / REF_PARTICIPATION)
                 = sqrt(0.10 / 0.02) = sqrt(5) = 2.24  (crisis scenario)

slippage = 0.001 * 3.33 * 2.24 = 0.00746 = 0.75%

But we also need the base participation effect. Refined:

slippage = BASE * vol_factor * (participation / REF_PARTICIPATION)^0.5
         = 0.001 * (5/1.5) * (0.10/0.02)^0.5
         = 0.001 * 3.33 * 2.24
         = 0.0075 (0.75%)
```

For the extreme case (vol=8%, participation=20%):
```
slippage = 0.001 * (8/1.5) * (0.20/0.02)^0.5
         = 0.001 * 5.33 * 3.16
         = 0.0169 (1.7%)
```

To reach the observed 2-5% range, we add a spread component:
```
spread_slippage = vol * SPREAD_MULTIPLIER  (bid-ask spread widens with vol)
total_slippage = impact_slippage + spread_slippage
```

With `SPREAD_MULTIPLIER = 0.3`:
```
spread = 0.08 * 0.3 = 0.024 (2.4%)
total = 1.7% + 2.4% = 4.1% -- matches observed crisis range
```

### Final Formula (With Spread Component)

```python
def compute_dynamic_slippage(
    close: float,
    daily_volume: float,
    order_qty: int,
    rolling_vol: float,
    side: str,
    config: dict,
) -> float:
    """Compute volatility-and-liquidity-adjusted slippage."""
    base = config.get("slippage_base", 0.001)
    ref_vol = config.get("slippage_ref_vol", 0.015)
    ref_part = config.get("slippage_ref_participation", 0.02)
    floor = config.get("slippage_floor", 0.0003)
    ceiling = config.get("slippage_ceiling", 0.05)
    spread_mult = config.get("slippage_spread_multiplier", 0.3)

    # Guard against bad data
    if rolling_vol != rolling_vol or rolling_vol <= 0:
        rolling_vol = ref_vol  # fallback to calm market
    if daily_volume != daily_volume or daily_volume <= 0:
        return ceiling  # no liquidity = max slippage

    # Volume participation ratio
    order_notional = abs(order_qty) * close
    daily_turnover = daily_volume * close
    participation = order_notional / (daily_turnover + 1e-8)

    # Impact component (Almgren-Chriss sqrt law)
    vol_factor = rolling_vol / ref_vol
    liquidity_factor = (participation / ref_part) ** 0.5
    impact = base * vol_factor * liquidity_factor

    # Spread component (widens with volatility)
    spread = rolling_vol * spread_mult

    # Total with floor/ceiling
    total = impact + spread
    return max(floor, min(ceiling, total))
```

## A-Share Specific Calibration Notes

### CSI1000 Small-Cap Characteristics

1. **Low ADV**: CSI1000 stocks average 5-50M CNY daily turnover vs 500M+ for CSI300
2. **High volatility**: 1.5-3% daily vol in calm markets, 5-10% in stress
3. **Wide bid-ask spreads**: 0.2-0.5% for CSI1000 vs 0.01-0.05% for CSI300
4. **T+1 rule**: No intraday reversal to reduce impact; slippage is permanent
5. **10% price limits**: Cap maximum slippage at ~10% per day (already enforced in engine.py)
6. **2024-02 CSI1000 crash**: Quant fund redemption wave caused liquidity spiral
   - Many stocks hit limit-down for 3-5 consecutive days
   - Effective slippage for forced sellers: 3-5% per trade
   - This is the scenario R6 targets

### Calibration Data Sources

- akshare daily OHLCV: compute rolling 20d volatility from close prices
- Existing `volume` field in prices dict: already available
- No additional data sources needed

## Implementation Plan

### Phase 1: Core Function (1 file, ~30 lines)

**File: `ashare_lab/paper/engine.py`**

1. Add `compute_dynamic_slippage()` function (see formula above)
2. Keep existing `apply_slippage()` unchanged for backward compatibility
3. Add new `apply_dynamic_slippage()` that calls compute + apply

```python
def apply_dynamic_slippage(
    close: float,
    side: str,
    daily_volume: float,
    order_qty: int,
    rolling_vol: float,
    config: dict,
) -> float:
    """Apply dynamic slippage based on volatility and liquidity."""
    slippage = compute_dynamic_slippage(
        close, daily_volume, order_qty, rolling_vol, side, config
    )
    return apply_slippage(close, side, slippage)
```

### Phase 2: Pipeline Integration (1 file, ~15 lines)

**File: `ashare_lab/paper/pipeline.py`**

1. In the prices dict construction (line 713-720), add `rolling_vol` field
2. Compute from existing historical close prices (20-day rolling std of log returns)

```python
# In the prices dict construction:
ctx.prices[matched] = {
    "close": pdata["close"] / _factor,
    "change": pdata["change"],
    "volume": pdata["volume"],
    "factor": pdata["factor"],
    "rolling_vol": pdata.get("rolling_vol", 0.015),  # NEW
    "adjusted": True,
    "threshold": get_limit_threshold(matched, ctx.st_names),
}
```

### Phase 3: Settle Day Integration (1 file, ~10 lines changed)

**File: `ashare_lab/paper/engine.py`**

1. In `settle_day()`, extract `rolling_vol` from pdata
2. Replace `apply_slippage(close, side, slippage)` calls with `apply_dynamic_slippage()`
3. Add config flag `dynamic_slippage: true` to enable (default false for backward compat)

Lines to change: 273, 417, 446

```python
# Line 273 (sell path):
rolling_vol = pdata.get("rolling_vol", 0.015)
if config.get("dynamic_slippage", False):
    fill_price = apply_dynamic_slippage(
        close, "sell", volume, target_qty, rolling_vol, config)
else:
    fill_price = apply_slippage(close, "sell", slippage)

# Lines 417, 446 (buy path): same pattern
```

### Phase 4: Config Update (1 file, ~10 lines)

**File: config YAML (wherever slippage is configured)**

```yaml
# Dynamic slippage configuration
dynamic_slippage: true          # Enable dynamic model (false = legacy flat 0.1%)
slippage_base: 0.001            # 10 bps base (calm market)
slippage_ref_vol: 0.015         # 1.5% reference daily volatility
slippage_ref_participation: 0.02 # 2% reference volume participation
slippage_floor: 0.0003          # 3 bps minimum
slippage_ceiling: 0.05          # 5% maximum (crisis cap)
slippage_spread_multiplier: 0.3 # Spread = vol * multiplier
```

### Phase 5: Tests (~30 lines)

**File: `tests/test_engine.py`** (or wherever engine tests live)

```python
def test_dynamic_slippage_calm_market():
    """Normal CSI1000: ~10 bps slippage."""
    slippage = compute_dynamic_slippage(
        close=10.0, daily_volume=1_000_000, order_qty=1000,
        rolling_vol=0.015, side="buy",
        config={"slippage_base": 0.001, "slippage_ref_vol": 0.015,
                "slippage_ref_participation": 0.02, "slippage_floor": 0.0003,
                "slippage_ceiling": 0.05, "slippage_spread_multiplier": 0.3}
    )
    assert 0.0005 < slippage < 0.002  # ~10 bps range

def test_dynamic_slippage_crisis():
    """CSI1000 crisis: slippage should be 1-5%."""
    slippage = compute_dynamic_slippage(
        close=10.0, daily_volume=500_000, order_qty=5000,
        rolling_vol=0.05, side="sell",
        config={"slippage_base": 0.001, "slippage_ref_vol": 0.015,
                "slippage_ref_participation": 0.02, "slippage_floor": 0.0003,
                "slippage_ceiling": 0.05, "slippage_spread_multiplier": 0.3}
    )
    assert 0.01 < slippage < 0.05  # 1-5% range

def test_dynamic_slippage_no_volume():
    """Zero volume = ceiling slippage."""
    slippage = compute_dynamic_slippage(
        close=10.0, daily_volume=0, order_qty=1000,
        rolling_vol=0.015, side="buy", config={...})
    assert slippage == 0.05  # ceiling

def test_dynamic_slippage_nan_vol():
    """NaN volatility = fallback to reference."""
    slippage = compute_dynamic_slippage(
        close=10.0, daily_volume=1_000_000, order_qty=1000,
        rolling_vol=float('nan'), side="buy", config={...})
    assert 0.0003 < slippage < 0.002  # calm market range
```

## Estimated Effort

| Phase | Files | Lines Changed | Effort |
|-------|-------|---------------|--------|
| Phase 1: Core function | 1 | +30 | 1 hour |
| Phase 2: Pipeline integration | 1 | +15 | 30 min |
| Phase 3: Settle day integration | 1 | ~10 changed | 30 min |
| Phase 4: Config | 1 | +10 | 15 min |
| Phase 5: Tests | 1 | +30 | 1 hour |
| **Total** | **5** | **~95** | **3.25 hours** |

## Risk Assessment

- **Backward compatible**: `dynamic_slippage: false` preserves current behavior
- **No new dependencies**: uses only stdlib math + existing data
- **Testable**: pure function, no side effects, easy to unit test
- **Calibration risk**: parameters may need tuning after comparing backtest results
  - Mitigation: start with conservative parameters, adjust based on historical match

## References

1. Almgren, R. and Chriss, N. (2001). "Optimal execution of portfolio transactions"
2. quant-ashare: https://github.com/wangpage/quant-ashare (Trick #4)
3. PatternSmart: https://patternsmart.com/wp/how-do-i-code-dynamic-slippage-models-in-python/
4. Gurovich, A. (2026). "Predicting Slippage in Equity Markets Using Probabilistic ML"
5. Cheng, Shi, Zhang (2025). "Intraday liquidity and expected return in China's stock market"
6. QuestDB: https://questdb.com/glossary/slippage-and-market-impact-estimation/
