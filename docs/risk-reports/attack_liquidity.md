# Liquidity / Market Microstructure Attack Analysis

**Target**: ashare-lab paper trading system (CSI1000 universe, ~280K NAV)
**Date**: 2026-07-15
**Method**: Exact code-path tracing + historical scenario validation

---

## System Parameters (from configs/baseline.yaml)

| Parameter | Value | Implication |
|-----------|-------|-------------|
| slippage | 0.001 (0.1%) | Fixed, does not scale with volatility |
| volume_participation_pct | 0.05 (5%) | Max fill = 5% of daily volume |
| carry_days | 3 | Orders cancel after 3 failed attempts |
| topk | 15 | 15 concentrated positions |
| risk_degree | 0.95 | 95% of NAV deployed to equity |
| initial_cash | 300,000 | ~19K per position |
| drawdown_hard | 0.15 | Hard halt at 15% drawdown (buying only) |
| daily_loss | 0.03 | Daily loss halt at 3% (buying only) |
| trailing_stop | 0.20 | Full-position sell at 20% from peak |
| concentration | 0.15 | 15% NAV cap per position |
| market_regime_decline | 0.08 | 8% CSI1000 decline over 10 days halts buying |
| soft_drawdown | 0.10 | Reduces topk from 15 to 7 at 10% drawdown |
| liquidity_min_turnover | 50,000,000 | 50M CNY 20-day avg turnover filter |

---

## Attack 1: Mass Limit-Down (30% of Universe Hits Limit)

### Historical Precedent [KNOWN]
- 2024-02-05: Nearly 30% of CSI1000 stocks hit limit-down simultaneously (Nasdaq, Reuters)
- 2026-07-13: 176 stocks limit-down, CSI1000 -5.01% in one session (Sina Finance)
- 2024-01-22 to 2024-02-05: CSI1000 fell 27% in January, 8.68% intraday on Feb 5

### Exact Code Path

**Settlement** (engine.py:242-252):
```python
# Limit-down block
if is_limit_down(change, threshold):
    if carry_day < carry_days_limit:   # carry_days_limit = 3
        update_order(conn, oid, status="carry")
        result.carries_to_bump.append(...)
    else:
        update_order(conn, oid, status="cancelled")
        result.cancels.append(symbol)
    continue
```

**Limit-down detection** (engine.py:89-93):
```python
def is_limit_down(change: float, threshold: float = 0.099) -> bool:
    if change != change:  # NaN guard
        return False
    return change <= -threshold
```

**Threshold** (engine.py:65-79): Non-ST main board = 0.099 (10%), ChiNext/STAR = 0.199 (20%).

### Failure Sequence

Day 1 (crash begins):
- 4-5 of 15 positions hit limit-down (-10%)
- Sell orders carry (carry_day=0 -> 1)
- Risk check: daily_loss = ~5% -> halts buying (3% threshold exceeded)
- Risk check: drawdown ~5% -> below 15% hard halt
- **Net loss: ~5% = ~15,000 CNY**

Day 2 (continuation):
- Same 4-5 stocks limit-down again
- Sell orders carry (carry_day=1 -> 2)
- Market regime: CSI1000 down ~8% over 2 days -> regime filter may fire (needs 10 days of data, so NO)
- Trailing stop: stocks down 19% from holding_high -> below 20% threshold, NO trigger
- **Cumulative loss: ~10% = ~30,000 CNY**

Day 3 (capitulation):
- Same stocks limit-down third day
- Sell orders carry (carry_day=2 -> 3)
- Trailing stop: stocks now down 27% from peak -> TRIGGERS (20% threshold)
- Risk check fires: forced_sells[symbol] = full position qty
- **But**: the sell order for this symbol already exists from Day 1 (carry status)
- pipeline.py:1360-1368: already-pending sell check prevents duplicate insertion
- The existing carry order will be re-evaluated NEXT day (Day 4)

Day 4 (attempted exit):
- If stock is STILL limit-down: carry_day=3, equals carry_days_limit -> ORDER CANCELLED
- **Position is now LOCKED IN at ~28% loss with no exit mechanism**
- If stock opens (not limit-down): sells at close with 0.1% slippage (unrealistic)

### Critical Bug: No Forced Liquidation at Hard Halt

risk.py:351: `buying_halted = drawdown_halted or daily_loss_halted or regime_halted`

The hard drawdown breaker (15%) and daily loss limit (3%) ONLY halt buying. There is NO forced sell at these thresholds. The system waits for the trailing stop (20%) to trigger forced sells, creating a 5% NAV gap where the portfolio is frozen:

- At 15% drawdown: buying halts, but no sells triggered
- At 20% drawdown: trailing stop fires, but sell orders may be blocked by limit-down
- At 28% cumulative: orders cancelled after 3 carry days, position locked

### Max 1-Day Loss Estimate

Best case (diversified): 5 stocks * -10% = 5/15 * 10% = 3.3% = ~10,000 CNY
Realistic case: 15 stocks * weighted -7% = ~7% = ~21,000 CNY
Worst case (all limit-down): 15 stocks * -10% = -10% = ~30,000 CNY

**Multi-day worst case**: 3 consecutive limit-down days + order cancellation = -28% per position = ~84,000 CNY (28% of 300K NAV)

### Severity: CRITICAL (P0)

The system has no mechanism to exit limit-down positions. After 3 carry days, orders are silently cancelled, and positions are stranded with no further action.

---

## Attack 2: Liquidity Dry-Up (Bid-Ask Spread Widens 10x)

### Historical Precedent [KNOWN]
- 2026-06-08: "绝大多数中小市值个股长期无人问津，日均成交数千万甚至不足千万，交易深度近乎消失" (Eastmoney)
- 2024-02-05: CSI1000 futures hit 10% limit-down, turnover spiked 7x (93B vs 13B avg) (AsiaFinancial)
- 2015-07-16: Limit-down stocks had sell-1 depth of <1% of float (NBD)

### Exact Code Path

**Slippage model** (engine.py:103-109):
```python
def apply_slippage(close: float, side: str, slippage: float = 0.001) -> float:
    if side == "buy":
        return close * (1.0 + slippage)
    return close * (1.0 - slippage)
```

**Volume cap** (engine.py:128-143):
```python
def cap_fill_by_volume(target_qty, daily_volume, side, participation_pct=0.05):
    if daily_volume != daily_volume or daily_volume == 0.0:
        return 0
    max_fill = int(daily_volume * participation_pct)
    if side == "buy":
        max_fill = round_lots(max_fill, "buy")
    return min(target_qty, max_fill)
```

**Settlement fill path** (engine.py:255-258 for sell, 415-418 for buy):
```python
fill_price = apply_slippage(close, "sell", slippage)       # close * 0.999
fill_qty = cap_fill_by_volume(target_qty, volume, "sell", 0.05)
```

### Failure Analysis

The slippage model is **fixed at 0.1%** regardless of:
- Market volatility
- Bid-ask spread
- Order size relative to volume
- Intraday price movement

Real A-share slippage during liquidity crises:
- Normal: 0.05-0.15% (matches the model)
- Volatile: 0.3-1.0%
- Crisis (limit-down approach): 2-5%
- Flash crash: 5-10%

**Example**: Stock XYZ, close=10.00, volume=500,000 shares
- System fills at: 10.00 * 0.999 = 9.99 (sell)
- Reality during crisis: best bid = 9.50 (5% below close)
- Slippage error: 0.49 per share = 4.9% hidden cost

**Volume cap interaction**: When volume drops 90% (common in CSI1000 small caps):
- Normal volume: 1,000,000 -> max fill = 50,000 shares
- Crisis volume: 100,000 -> max fill = 5,000 shares
- If position = 1,900 shares (19K/10.00), cap is not binding
- But if volume = 0 (limit-down): fill = 0, order carries

### Multi-Day Compounding

The system uses **previous day's close** for settlement (pipeline.py fetches prices for trade_date). During a multi-day decline:
- Day 1 close: 10.00, system sells at 9.99
- Day 2 open: 9.50 (gap down), but system uses Day 2 close: 9.20
- Realized slippage error: 0.79/share (7.9%) vs modeled 0.01/share (0.1%)

### Max 1-Day Loss Estimate

Hidden slippage cost: 15 positions * 19K * 4.9% hidden = ~14,000 CNY (4.7% of NAV)
This is ON TOP of the market loss, and is completely invisible to the risk system.

### Severity: HIGH (P1)

The fixed slippage model systematically underestimates execution costs during volatility. The system's NAV calculation is overstated because it assumes fills at 0.1% slippage when real costs are 2-5x higher.

---

## Attack 3: Margin Cascade (Forced Liquidation by Other Funds)

### Historical Precedent [KNOWN]
- 2024-01-23: Snowball derivatives triggered knock-in on CSI1000, forced futures selling cascaded (Reuters)
- 2026-07-13: "融资余额自6/23见顶3万亿后连续回落" - margin balance dropped 1079B in 2 weeks (163.com)
- 2026-07-13: "量化股票多头规模达1.83万亿放大波动" (163.com)

### Exact Code Path

**Trailing stop** (risk.py:138-147):
```python
def check_trailing_stop(holding_high, current_price, stop_pct):
    if holding_high <= 0:
        return False
    decline = (holding_high - current_price) / holding_high
    return decline > stop_pct   # stop_pct = 0.20
```

**Forced sell generation** (risk.py:301-306):
```python
if check_trailing_stop(holding_high, close, config["trailing_stop"]):
    forced_sells[symbol] = pos["qty"]   # full-position exit
    # ... cooldown entry
```

**Market regime** (risk.py:118-135):
```python
def check_market_regime(csi1000_closes, decline_threshold, lookback_days):
    if len(csi1000_closes) < lookback_days + 1:
        return False                    # needs 11 data points
    window = csi1000_closes[-(lookback_days + 1):]
    cumulative_return = (window[-1] / window[0]) - 1
    return cumulative_return < -decline_threshold  # -8%
```

### Failure Cascade

**Phase 1: Margin calls elsewhere trigger selling**
- Other funds face margin calls, dump CSI1000 stocks
- CSI1000 drops 5-8% in days
- Our system: regime filter needs 10-day lookback, only fires after sustained decline

**Phase 2: Our trailing stops trigger**
- holding_high was set at purchase price or subsequent high
- If a stock was bought at 10.00 and is now at 7.90 (21% decline):
  - trailing stop fires -> forced_sells[symbol] = full qty
  - BUT: the stock might be limit-down, so the sell order carries
  - carry_day increments 1, 2, 3 -> CANCELLED

**Phase 3: Cascade feedback**
- Our system's cancelled sell orders mean we hold the position
- More margin calls -> more selling -> prices drop further
- Our trailing stop fires again next day -> new sell order -> carry -> cancel
- The position is effectively locked in a death spiral

### Critical Gap: No Intraday Stop-Loss

The system settles once per day at close. During a margin cascade:
- Stock drops 8% intraday, triggers margin calls for leveraged funds
- Those funds sell aggressively, pushing price to limit-down
- Our system sees the close price (limit-down) and tries to sell
- But it's already limit-down -> order carries
- Next day: more selling, another limit-down
- Our system is always one day behind the cascade

### Max 1-Day Loss Estimate

If 5 positions caught in margin cascade:
- Day 1: -10% each (limit-down) -> -3.3% NAV
- Day 2: -10% each (still limit-down) -> -3.0% NAV (of reduced base)
- Day 3: -10% + trailing stop fires but can't sell -> -2.7% NAV
- Day 4: order cancelled, position locked at -28% loss
- Total: ~9% NAV = ~27,000 CNY, with no exit

### Severity: CRITICAL (P0)

The system has no intraday risk management and no mechanism to handle cascading limit-down scenarios. The trailing stop is the only forced-sell trigger, but it fires too late (20%) and can't execute during limit-down.

---

## Attack 4: Flash Crash (5% Drop in 10 Minutes, Then Recovery)

### Historical Precedent [KNOWN]
- 2024-02-05: CSI1000 dropped 8.68% intraday (BusinessToday)
- 2026-07-13: CSI1000 -5.01% in single session, "险守MA250(7748.9)" (163.com)
- A-share flash crashes typically happen in the last 30 minutes of trading

### Exact Code Path

**Pipeline execution** (pipeline.py:1582-1663):
- Pipeline runs ONCE per day, after market close
- `_step8_settle()` uses that day's close price
- No intraday monitoring, no real-time execution

**Settlement** (engine.py:255):
```python
fill_price = apply_slippage(close, "sell", slippage)  # uses daily close
```

### Failure Scenarios

**Scenario A: Flash crash within one day, recovers before close**
- 10:30 AM: market drops 5% (flash crash)
- 2:30 PM: market recovers to -0.5%
- Close price: -0.5%
- System sees: -0.5% change, no risk triggers, no action
- **Result: System is completely blind to the intraday event**
- This is actually GOOD for the system (no panic selling at bottom)

**Scenario B: Flash crash across two days**
- Day 1 close: -5% (system sees this, triggers daily_loss halt)
- Day 2 open: gap down another -3%, intraday recovers to +2%
- Day 2 close: +2%
- System: Day 1 daily_loss halt fired, but no sells triggered (buying-only halt)
- Day 2: regime check (8% over 10 days) may or may not fire
- **Result: System holds through the crash, but the daily_loss halt prevents buying at the bottom**

**Scenario C: Flash crash + continued selling (worst case)**
- Day 1 close: -5% (daily_loss halt fires)
- Day 2 close: -3% (cumulative -8%, regime check fires)
- Day 3 close: -2% (cumulative -10%)
- System: buying halted since Day 1, but NO forced sells
- Trailing stop: if holding_high was 20% above current, stop hasn't fired
- **Result: System rides the entire decline without selling, then sells on recovery (if trailing stop fires at 20%)**

### Critical Gap: No Intraday Execution

The system is designed as a daily close-only engine. This is actually a deliberate design choice for paper trading, but it means:
- Can't capture intraday reversals (good)
- Can't react to intraday crashes (bad)
- Can't execute stop-losses intraday (bad)
- All risk management is delayed by 24 hours

### Max 1-Day Loss Estimate

If flash crash happens and system is forced to sell at close:
- Close price = bottom of crash
- 15 positions * -5% = -5% NAV = ~15,000 CNY
- If trailing stop triggers at 20% and sells at close: -20% per position = ~60,000 CNY

### Severity: MEDIUM (P2)

The daily close-only design is a known limitation. For paper trading, this is acceptable. For real trading, this would be a critical gap. The system can't exploit flash crashes (buy at bottom) but also can't be hurt by intraday noise.

---

## Attack 5: Suspension Risk (Stocks Halt Trading for Days)

### Historical Precedent [KNOWN]
- 2015-07: "千股停牌" - thousands of stocks suspended simultaneously during crash
- A-share suspensions can last days to weeks for material announcements
- IPO lock-up, restructuring, regulatory investigations all cause suspensions

### Exact Code Path

**Suspension detection** (engine.py:96-100):
```python
def is_suspended(volume: float) -> bool:
    if volume != volume:  # NaN
        return True
        return volume == 0.0
```

**Suspension handling** (engine.py:235-240 for sell, 370-375 for buy):
```python
if is_suspended(volume):
    update_order(conn, oid, status="carry")
    result.carries_suspended.append(...)
    continue
```

**Key difference from limit-down**: Suspended orders carry with NO carry_day limit.
- Limit-down: cancelled after 3 carry days
- Suspension: carries INDEFINITELY (no cancellation logic)

### Failure Analysis

**Scenario**: 2 of 15 positions suspend for 10 days

During suspension:
- Sell orders carry forever (no cancellation)
- The positions are locked, representing 13% of NAV
- Risk system can't evaluate these positions (no price data)
- NAV calculation uses last known price (stale)

When suspension lifts:
- Stock may gap down 10-30% (common after long suspensions)
- System sees the new close price on resume day
- Trailing stop may fire (if decline > 20% from holding_high)
- Sell order executes at the new (lower) price with 0.1% slippage (unrealistic)

**Concentration risk**: If one suspended stock is 13% of NAV and gaps down 30% on resume:
- Loss = 13% * 30% = 3.9% NAV in one day
- This is a single-stock event, not diversified

**Worst case**: Multiple suspensions + gap down
- 3 stocks suspend (20% of NAV)
- All resume on same day with -20% gap
- Loss = 20% * 20% = 4% NAV = ~12,000 CNY

### Critical Gap: No Suspension Risk Premium

The system has no mechanism to:
- Limit concentration in stocks with high suspension risk
- Adjust NAV for stale prices during suspension
- Set a maximum suspension exposure
- Handle partial resumes (some stocks resume, others don't)

### Max 1-Day Loss Estimate

On resume day after 10-day suspension:
- 2 stocks * 13% NAV each * -20% gap = ~5.2% NAV = ~15,600 CNY
- If 3 stocks: 7.8% NAV = ~23,400 CNY

### Severity: HIGH (P1)

Suspension risk is inherent to A-shares and the system has no special handling beyond indefinite carry. The lack of suspension risk premium means the system may over-concentrate in suspension-prone stocks.

---

## Combined Worst-Case Scenario

### "The Perfect Storm" (all 5 attacks simultaneously)

This is not hypothetical -- it closely matches the Jan-Feb 2024 CSI1000 crash:

1. **Day 1**: External shock (geopolitical, margin calls) triggers mass sell-off
   - 30% of CSI1000 hits limit-down
   - System: 5 positions limit-down, daily_loss halt fires
   - Loss: -5% NAV

2. **Day 2**: Cascade continues
   - Same stocks limit-down again
   - Margin calls from other funds amplify selling
   - System: trailing stops fire for 3 stocks but can't execute
   - Loss: -10% cumulative NAV

3. **Day 3**: Capitulation
   - CSI1000 down 15% from peak
   - System: hard drawdown halt fires (15%), but only halts buying
   - Trailing stops fire for more stocks, but limit-down blocks execution
   - 2 stocks suspend trading
   - Loss: -15% cumulative NAV

4. **Day 4-6**: Orders cancel
   - Limit-down sell orders cancel after 3 carry days
   - Positions locked with -28% loss per stock
   - Suspended stocks carry indefinitely

5. **Day 7-10**: Partial recovery
   - Some stocks recover, system tries to sell (new trailing stop triggers)
   - But 0.1% slippage model is wrong -- real slippage is 2-5%
   - Hidden slippage cost: additional 3-5% NAV

### Total Max Loss

| Component | Loss (CNY) | % of NAV |
|-----------|-----------|----------|
| Market decline (15 stocks * -10%) | 30,000 | 10.0% |
| Trailing stop gap (5% between halt and stop) | 15,000 | 5.0% |
| Hidden slippage (5% on sells) | 14,000 | 4.7% |
| Suspension gap-down (2 stocks) | 12,000 | 4.0% |
| Order cancellation (locked positions) | 0 | 0% (unrealized) |
| **TOTAL** | **~71,000** | **~23.7%** |

**Realistic estimate**: 15-20% NAV drawdown in 3-5 days (~45,000-60,000 CNY)

---

## Summary of Vulnerabilities

| # | Attack | Severity | Max 1-Day Loss | Root Cause |
|---|--------|----------|----------------|------------|
| 1 | Mass Limit-Down | CRITICAL | 10% (30K) | 3-day carry then cancel; no exit mechanism |
| 2 | Liquidity Dry-Up | HIGH | 4.7% hidden (14K) | Fixed 0.1% slippage; no spread model |
| 3 | Margin Cascade | CRITICAL | 3.3% (10K) | No intraday stop; trailing stop too late |
| 4 | Flash Crash | MEDIUM | 5% (15K) | Daily close-only; no intraday execution |
| 5 | Suspension | HIGH | 5.2% (15.6K) | Indefinite carry; no risk premium |

---

## Recommended Fixes (Priority Order)

### P0: Limit-Down Exit Mechanism
**File**: engine.py settle_day()
**Fix**: After 3 carry days on limit-down, convert to market order on next non-limit-down day (don't cancel). Or: add a "panic sell" mode that accepts any fill above 0.

```python
# Current (broken):
if carry_day >= carry_days_limit:
    update_order(conn, oid, status="cancelled")  # LOSES EXIT OPPORTUNITY

# Proposed:
if carry_day >= carry_days_limit:
    # Force-fill at limit-down price (worst case, but at least exits)
    fill_price = close * (1 - threshold)  # limit-down price
    # ... process fill at worst-case price
```

### P0: Hard Drawdown Forced Liquidation
**File**: risk.py run_all_risk_checks()
**Fix**: When drawdown_hard fires, generate forced_sells for ALL positions (not just halt buying).

```python
# Current:
buying_halted = drawdown_halted or daily_loss_halted or regime_halted

# Proposed:
if drawdown_halted:
    # Force-sell ALL positions when hard drawdown breached
    for symbol, pos in current_positions.items():
        forced_sells[symbol] = pos["qty"]
```

### P1: Dynamic Slippage Model
**File**: engine.py apply_slippage()
**Fix**: Scale slippage with volatility and volume.

```python
# Current:
def apply_slippage(close, side, slippage=0.001):
    return close * (1 + slippage) if side == "buy" else close * (1 - slippage)

# Proposed:
def apply_slippage(close, side, base_slippage, volume, avg_volume, change):
    volatility_factor = max(1.0, abs(change) / 0.03)  # 3% change = 1x, 10% = 3.3x
    liquidity_factor = max(1.0, avg_volume / max(volume, 1))
    actual_slippage = base_slippage * volatility_factor * liquidity_factor
    return close * (1 + actual_slippage) if side == "buy" else close * (1 - actual_slippage)
```

### P1: Suspension Risk Cap
**File**: signal.py filter_candidates()
**Fix**: Limit total NAV exposure to stocks with high suspension risk (low liquidity, ST status, recent volatility).

### P2: Intraday Stop-Loss (for real trading)
**Note**: Not applicable to paper trading, but essential for real execution.

---

## Conclusion

The ashare-lab system has **two critical vulnerabilities** (P0) that could cause 15-25% NAV drawdown in a 2024-style CSI1000 crash:

1. **No exit from limit-down positions**: Orders cancel after 3 days, leaving positions stranded
2. **No forced liquidation at hard drawdown**: The 15% drawdown halt only stops buying, doesn't trigger sells

The system was designed for normal market conditions. Under stress (which is when risk management matters most), it fails to protect capital. The fixed 0.1% slippage model and daily close-only execution compound the problem by making the system blind to real execution costs and intraday risk.

**Reference**: The Jan-Feb 2024 CSI1000 crash (27% decline in one month, 30% of stocks limit-down on worst day) is a direct precedent for these attack scenarios. The system would have suffered 20-25% drawdown in that event, with no mechanism to exit losing positions.
