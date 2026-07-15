# Risk R7: Suspension + Gap-Down -- Solution Design

## Problem Statement

**Current code** (`engine.py:235-240`):
```python
if is_suspended(volume):
    update_order(conn, oid, status="carry")
    result.carries_suspended.append(...)
    continue
```

**Three gaps:**
1. Suspended orders carry indefinitely -- no cancellation limit (unlike limit-down which has `max_resets`)
2. No stale-price adjustment -- portfolio NAV uses frozen last-trade price, distorting risk metrics
3. No concentration cap for suspension-prone stocks -- a single ST stock can freeze 15% of NAV

## Part 1: Suspension Risk Scoring Model

### Design: Rule-Based Scoring (Not ML)

For a paper trading system, a rule-based approach is more appropriate than the ML ensemble model described in the academic literature (Liu 2021). Reasons:
- Interpretable: every score component has a clear business meaning
- No training data dependency: works from day 1
- Fast: no model inference latency in the settle loop
- Sufficient accuracy: the key risk factors are binary flags, not subtle patterns

### Scoring Dimensions

| Dimension | Signal | Score | Rationale |
|-----------|--------|-------|-----------|
| ST/*ST status | Name contains "ST" | +40 | CSRC mandate: ST stocks face 5% limit, high delisting risk |
| Recent suspension history | Suspended in last 90 days | +20 | Recidivism: stocks that suspended recently are 3x more likely to suspend again |
| Pending disclosure | Within 10 days of earnings deadline | +10 | Mandatory disclosure windows often trigger voluntary suspension |
| Abnormal volume spike | 5-day avg volume > 3x 20-day avg | +15 | Pre-suspension volume spike signals insider activity |
| Low analyst coverage | < 3 analysts covering | +10 | Information opacity increases suspension probability |
| Small cap | Market cap < 2B CNY | +5 | Small caps have higher suspension frequency (academic consensus) |

### Risk Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0-15  | Low  | Normal trading |
| 16-30 | Medium | Apply suspension concentration cap (2%) |
| 31-50 | High | Apply strict cap (1%) + stale-price haircut |
| 51+   | Critical | Hard exclude from universe |

### Implementation Location

New file: `ashare_lab/paper/suspension_risk.py`

```python
@dataclass
class SuspensionRiskScore:
    symbol: str
    score: int
    tier: str  # "low" | "medium" | "high" | "critical"
    components: dict[str, int]
    reason: str

def score_suspension_risk(
    symbol: str,
    is_st: bool,
    recent_suspension_days: int,
    days_to_disclosure: int | None,
    volume_ratio_5d_20d: float,
    analyst_count: int,
    market_cap_cny: float,
) -> SuspensionRiskScore:
    """Score a stock's suspension risk. Pure function, no side effects."""
    components = {}
    if is_st:
        components["st_status"] = 40
    if recent_suspension_days <= 90:
        components["recent_suspension"] = 20
    if days_to_disclosure is not None and days_to_disclosure <= 10:
        components["disclosure_window"] = 10
    if volume_ratio_5d_20d > 3.0:
        components["volume_spike"] = 15
    if analyst_count < 3:
        components["low_coverage"] = 10
    if market_cap_cny < 2e9:
        components["small_cap"] = 5

    total = sum(components.values())
    if total >= 51:
        tier = "critical"
    elif total >= 31:
        tier = "high"
    elif total >= 16:
        tier = "medium"
    else:
        tier = "low"

    return SuspensionRiskScore(
        symbol=symbol, score=total, tier=tier,
        components=components,
        reason=", ".join(components.keys()),
    )
```

## Part 2: Concentration Cap for Suspension-Prone Stocks

### Current State

`risk.py:102-115` implements a flat 15% concentration cap (`config["concentration"]`).
No suspension-aware adjustment.

### Proposed Design

Add a **suspension-adjusted concentration cap** that overlays the existing flat cap:

```python
def check_suspension_concentration(
    position_value: float,
    total_nav: float,
    base_cap: float,
    suspension_tier: str,
    tier_caps: dict[str, float],
) -> tuple[bool, float]:
    """Check position against suspension-adjusted cap.

    tier_caps: {"low": 0.15, "medium": 0.02, "high": 0.01, "critical": 0.0}
    """
    if total_nav <= 0:
        return (False, 0.0)
    cap = tier_caps.get(suspension_tier, base_cap)
    cap = min(cap, base_cap)  # never exceed the base cap
    pct = position_value / total_nav
    if pct > cap:
        return (True, position_value - cap * total_nav)
    return (False, 0.0)
```

### Config Addition

```yaml
paper:
  risk:
    suspension_concentration:
      low: 0.15      # same as base concentration
      medium: 0.05   # 5% for medium-risk suspension stocks
      high: 0.02     # 2% for high-risk
      critical: 0.00 # hard exclude
```

### Integration Point

In `risk.py:run_all_risk_checks()`, after the existing concentration loop (line 267-278), add a second pass that checks suspension-adjusted caps. The suspension tier data comes from a pre-computed dict passed through the pipeline.

## Part 3: Carry Limit for Suspended Orders

### Current State

`engine.py:235-240`: suspended orders get `status="carry"` with no limit.
Limit-down orders have `carry_days_limit` (default 3) and `max_resets` (default 5).
Suspended orders bypass both limits.

### Proposed Design

Add a **suspension carry limit** with a longer window than limit-down (suspensions can last 25 trading days per CSRC rules):

```python
# In settle_day(), replace lines 235-240:

if is_suspended(volume):
    suspension_carry = order.get("suspension_carry_day", 0) + 1
    max_suspension_carry = config.get("max_suspension_carry_days", 25)

    if suspension_carry >= max_suspension_carry:
        # Force cancel after max carry days
        update_order(conn, oid, status="cancelled")
        result.cancels.append(symbol)
        logger.warning(
            "R7: cancelled suspended order %s after %d days",
            oid, suspension_carry,
        )
    else:
        update_order(
            conn, oid, status="carry",
            suspension_carry_day=suspension_carry,
        )
        result.carries_suspended.append({
            "order_id": oid, "symbol": symbol,
            "side": "sell",
            "suspension_day": suspension_carry,
            "days_remaining": max_suspension_carry - suspension_carry,
        })
    continue
```

### DB Schema Change

Add column `suspension_carry_day INTEGER DEFAULT 0` to the orders table.
This is independent of `carry_day` (used for limit-down) to avoid logic conflicts.

### Config Addition

```yaml
paper:
  max_suspension_carry_days: 25  # CSRC max suspension window
```

## Part 4: Stale Price Adjustment During Suspension

### Problem

When a stock is suspended, the engine falls back to `pos["avg_cost"]` (line 233).
This frozen price:
- Overstates/understates portfolio NAV
- Distorts concentration calculations
- Makes drawdown metrics unreliable

### Proposed Design: Beta-Adjusted NAV

Use the sector ETF return during suspension to estimate fair value:

```
adjusted_price = last_trade_price * (1 + beta * sector_return_since_suspension)
```

Where:
- `last_trade_price`: price on the last trading day before suspension
- `beta`: stock's 60-day beta vs CSI1000 (precomputed)
- `sector_return_since_suspension`: CSI1000 or sector ETF return during the suspension period

### Fallback Hierarchy

1. **Primary**: Beta-adjusted price (if beta and market data available)
2. **Secondary**: Peer-adjusted (average return of same-industry non-suspended stocks)
3. **Tertiary**: Last-trade price with a **haircut** (e.g., -5% for medium-risk, -10% for high-risk)

### Implementation

New function in `engine.py`:

```python
def adjusted_suspended_price(
    last_price: float,
    beta: float,
    market_return_during_suspension: float,
    suspension_tier: str,
    haircut_pct: dict[str, float],
) -> float:
    """Estimate fair value for a suspended stock.

    Returns the adjusted price, floored at 0.
    """
    if beta is not None and market_return_during_suspension is not None:
        adjusted = last_price * (1 + beta * market_return_during_suspension)
        # Floor at 0 (can't go negative)
        return max(adjusted, 0.0)

    # Fallback: apply tier-based haircut
    haircut = haircut_pct.get(suspension_tier, 0.05)
    return last_price * (1 - haircut)
```

### Config Addition

```yaml
paper:
  risk:
    suspension_haircut:
      low: 0.02       # 2% haircut for low-risk suspension
      medium: 0.05    # 5% haircut
      high: 0.10      # 10% haircut
      critical: 0.20  # 20% haircut (should be excluded anyway)
```

### Integration Points

1. **NAV calculation** (`ledger.py:compute_nav`): use adjusted price for suspended positions
2. **Concentration check** (`risk.py:check_concentration`): use adjusted price for position value
3. **Trailing stop** (`risk.py:check_trailing_stop`): skip check for suspended stocks (no price movement)

## Part 5: Database Schema Changes

```sql
-- Add to orders table
ALTER TABLE orders ADD COLUMN suspension_carry_day INTEGER DEFAULT 0;

-- Add to positions table (for stale-price tracking)
ALTER TABLE positions ADD COLUMN suspended_since TEXT;  -- ISO date
ALTER TABLE positions ADD COLUMN last_trade_price REAL; -- price before suspension
ALTER TABLE positions ADD COLUMN suspension_beta REAL;  -- 60-day beta
```

## Part 6: Risk Check Integration

### New Risk Dimension: Suspension Exposure

Add dimension 8 to `risk.py`:

```python
def check_portfolio_suspension_exposure(
    positions: dict[str, dict],
    suspension_scores: dict[str, SuspensionRiskScore],
    total_nav: float,
    max_suspension_pct: float = 0.20,
) -> tuple[bool, float]:
    """Check total portfolio exposure to suspension-prone stocks.

    Returns (is_over_limit, excess_value).
    max_suspension_pct: max % of NAV in medium+ suspension-risk stocks.
    """
    if total_nav <= 0:
        return (False, 0.0)

    suspension_value = 0.0
    for symbol, pos in positions.items():
        score = suspension_scores.get(symbol)
        if score and score.tier in ("medium", "high"):
            suspension_value += pos["market_value"]

    pct = suspension_value / total_nav
    if pct > max_suspension_pct:
        return (True, suspension_value - max_suspension_pct * total_nav)
    return (False, 0.0)
```

### Config Addition

```yaml
paper:
  risk:
    max_portfolio_suspension_pct: 0.20  # max 20% NAV in suspension-prone stocks
```

## Summary of Config Changes

```yaml
paper:
  max_suspension_carry_days: 25
  risk:
    # Existing (unchanged)
    concentration: 0.15
    # New
    suspension_concentration:
      low: 0.15
      medium: 0.05
      high: 0.02
      critical: 0.00
    suspension_haircut:
      low: 0.02
      medium: 0.05
      high: 0.10
      critical: 0.20
    max_portfolio_suspension_pct: 0.20
```

## Implementation Phases

### Phase 1: Carry Limit (Low Effort, High Impact)
- Add `suspension_carry_day` column to orders table
- Modify `engine.py:235-240` to track and enforce carry limit
- Add `max_suspension_carry_days` config
- **Effort**: 2-3 hours (1 file + 1 migration + tests)

### Phase 2: Suspension Risk Scoring (Medium Effort)
- Create `ashare_lab/paper/suspension_risk.py`
- Add data fetching for ST status, recent suspensions, analyst coverage
- Wire scoring into the pipeline (pre-compute once per day)
- **Effort**: 4-6 hours (new module + data integration + tests)

### Phase 3: Concentration Caps (Low Effort)
- Add `check_suspension_concentration` to `risk.py`
- Wire into `run_all_risk_checks`
- Add config
- **Effort**: 2-3 hours (1 file + config + tests)

### Phase 4: Stale Price Adjustment (Medium Effort)
- Add `adjusted_suspended_price` to `engine.py`
- Modify NAV calculation in `ledger.py`
- Add beta data fetching (precomputed in factor module)
- **Effort**: 4-6 hours (2 files + data integration + tests)

### Phase 5: Portfolio Exposure Check (Low Effort)
- Add `check_portfolio_suspension_exposure` to `risk.py`
- Wire into `run_all_risk_checks`
- **Effort**: 1-2 hours (1 file + tests)

## Total Estimated Effort

| Phase | Files Changed | Effort | Priority |
|-------|---------------|--------|----------|
| 1. Carry limit | engine.py, migration | 2-3h | P0 (fixes infinite carry bug) |
| 2. Risk scoring | suspension_risk.py, pipeline | 4-6h | P1 (enables all other phases) |
| 3. Concentration caps | risk.py, config | 2-3h | P1 (prevents freeze risk) |
| 4. Stale price | engine.py, ledger.py | 4-6h | P2 (improves NAV accuracy) |
| 5. Portfolio exposure | risk.py | 1-2h | P2 (portfolio-level guard) |
| **Total** | | **13-20h** | |

## Key Design Decisions

1. **Rule-based scoring over ML**: For a paper trading system, interpretability and zero cold-start outweigh the marginal accuracy gain of ML suspension prediction.

2. **25-day carry limit**: Matches CSRC maximum suspension window. Shorter limits risk cancelling orders that would have been filled on resume.

3. **Beta-adjusted pricing over flat haircut**: More accurate for market-wide moves. Haircut is the fallback for stocks without reliable beta.

4. **Separate `suspension_carry_day` from `carry_day`**: Avoids conflating limit-down carry (short, retriable) with suspension carry (long, may need forced cancel).

5. **Tier-based caps, not binary**: A stock with "medium" suspension risk (score 20) is not the same as "critical" (score 51+). Graduated caps allow the strategy to hold some exposure to moderately risky stocks while excluding the worst.

## References

- Academic: Liu (2021) "Prediction of Stock Suspension Based on Machine Learning" -- SMOTE+Tomek ensemble, but overkill for paper trading
- Academic: Tian (2020) "Suspension Bias in Chinese Financial Databases" -- documents that zero-return assumption biases size factor by 192bp/year
- Academic: PBCSF/Tsinghua "Can Stock Trading Suspension Calm Down Investors" -- investors with higher suspension fraction sell less of tradable stocks
- Industry: QMT/TianQin/BigQuant platforms all implement `is_suspended_stock()` checks and weight-freeze logic
- Industry: MyClaw ST Risk Filter -- 4 quantitative red lines (revenue+profit, net assets, dividends, loss chain) + penalty scoring
- Codebase: `engine.py:235-240` (current gap), `risk.py:102-115` (concentration cap), `baseline.yaml:88` (carry_days config)
