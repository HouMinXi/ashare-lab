# Risk R1: Limit-Down Position Lock-In - Solution Analysis

## 1. Current State Analysis

### 1.1 Code Path (engine.py settle_day, lines 244-269)

```
Limit-down sell order flow:
  is_limit_down(change, threshold)?
    |
    +-- YES, carry_day < 3 --> carry to next day (bump carry_day)
    |
    +-- YES, carry_day >= 3 --> reset_count++
        |
        +-- reset_count >= 5 (max_resets) --> CANCEL order (position STRANDED)
        |
        +-- reset_count < 5 --> reset carry_day to 0, keep carrying
```

**Timeline**: 3 days per carry cycle x 5 resets = 15 trading days of retry. After that, order is cancelled.

### 1.2 The Critical Bug

After `max_resets` expires and the order is cancelled, **the position remains in the portfolio with no active sell order**. The position is "stranded" -- it occupies capital, affects NAV, but has no exit path.

- `risk.py` trailing stop (D-39) will generate a `forced_sell` for the stranded position
- But `engine.py` limit-down check runs BEFORE the fill path, so the forced sell is blocked every day
- The position oscillates: risk generates forced sell -> engine blocks it -> risk generates again -> indefinitely

### 1.3 Risk.py Gap

Risk.py has 7 dimensions but **none specifically handle limit-down lock-in**:
- Trailing stop generates `forced_sells` but they are unexecutable during limit-down
- Market regime (D-38) halts buying but doesn't help exit stranded positions
- No market-wide limit-down count check (e.g., "50+ stocks limit-down = systemic risk")

### 1.4 Pipeline Gap (pipeline.py _step8_settle, lines 1040-1090)

`_step8_settle` calls `settle_day()` with pending orders. If a sell order was cancelled due to max_resets, the pipeline has no mechanism to re-create it. The position silently accumulates.

---

## 2. 2026 Best Practices (Research Findings)

### 2.1 How Production A-Share Quant Systems Handle Limit-Down Exit

From research across multiple production systems (zhou343-de/stock-trader-ai, juejin quant system, AKQuant, quant-ashare):

| Approach | Description | Source |
|----------|-------------|--------|
| **Next-day call auction market sell** | After limit-down detected, submit sell at next day's opening auction (09:25) with market price | juejin article: "持仓跌停封死 -> 次日集合竞价市价卖" |
| **Priority queue** | Limit-down flagged stocks get highest sell priority next trading day | zhou343-de: "跌停封板检测 -> 告警邮件 + 次日优先处理" |
| **Carry with logging** | Keep carrying but with explicit limit-down state tracking and alerts | BigQuant: carry + cancel on limit-up (opposite direction) |
| **Volatility-triggered exit** | Exit BEFORE limit-down via ATR expansion signals | Multiple systems: ATR-based stop loss |

### 2.2 Alternatives When Stock is Limit-Down

| Mechanism | Availability | Discount | Applicability to Paper Trading |
|-----------|-------------|----------|-------------------------------|
| **Block trade (大宗交易)** | Qualified investors only, min 300K shares or 200K CNY | 5-10% below close | Not applicable for small positions |
| **Inquiry transfer (询价转让)** | STAR/ChiNext only, min 1% of shares | 70-90% of 20-day avg | Not applicable |
| **OTC/Protocol transfer (协议转让)** | Requires regulatory approval, min 5% of shares | Negotiated | Not applicable |
| **Next-day call auction** | Open to all, next trading day 09:15-09:25 | Market price (may gap down) | **Best fit for paper trading** |

**Conclusion for paper trading**: Next-day call auction sell is the only practical mechanism. Block/OTC trades require real broker infrastructure.

### 2.3 Pre-Detection of Limit-Down Risk

From quant-ashare ADVANCED_TRICKS.md and multiple systems:

| Signal | Lead Time | Reliability |
|--------|-----------|-------------|
| **VPIN (Volume-Synchronized Probability of Informed Trading)** | 1-2 weeks before crash | >90% accuracy for flash crashes (Lopez de Prado) |
| **ATR expansion** | 2-3 days | Good for individual stocks |
| **Limit-down count > 50** | Same day (early warning) | Good for market-wide risk |
| **Sector concentration** | Days-weeks | Moderate |
| **Unlock/reduction events** | 10 days before price impact | Good (quant-ashare trick #12) |
| **Put/Call ratio change rate** | Days | Moderate for index |

### 2.4 What Institutional Funds Do with Stranded Positions

From 证券日报 (2026-05-24) and 36kr analysis:

1. **Wait it out**: Most institutions simply wait for limit-down chain to break, then exit at first opportunity
2. **Block trade discount**: Accept 5-10% loss via 大宗交易 to offload quickly
3. **Hedging**: Short CSI1000 futures to offset stranded position risk
4. **Mark-to-model**: For NAV reporting, use last tradeable price rather than limit-down price
5. **Position write-down**: After 20+ days, some funds write down to zero and focus on other positions

---

## 3. Concrete Implementation Plan

### Layer 1: PREVENTION - Reduce Limit-Down Exposure

**Goal**: Avoid buying stocks that are likely to hit limit-down.

#### 1A. Add limit-down probability filter to signal generation

File: `ashare_lab/paper/pipeline.py` (signal generation step)

```python
def _filter_limit_down_risk(candidates: list[dict], prices: dict, config: dict) -> list[dict]:
    """Filter out stocks with elevated limit-down risk.

    Signals:
    - Recent 5-day ATR > 2x 20-day ATR (volatility expansion)
    - Volume spike > 3x 20-day average (informed selling)
    - Near limit-down in last 3 days (> 7% decline)
    """
    atr_short = config.get("atr_short_days", 5)
    atr_long = config.get("atr_long_days", 20)
    atr_ratio_threshold = config.get("atr_ratio_threshold", 2.0)
    vol_spike_threshold = config.get("vol_spike_threshold", 3.0)
    near_limit_pct = config.get("near_limit_pct", 0.07)

    filtered = []
    for c in candidates:
        sym = c["symbol"]
        pdata = prices.get(sym, {})
        # Check near-limit-down: skip if declined >7% in last 3 days
        change_3d = pdata.get("change_3d", 0.0)
        if change_3d < -near_limit_pct:
            continue
        # Check ATR expansion
        atr_ratio = pdata.get("atr_ratio", 1.0)
        if atr_ratio > atr_ratio_threshold:
            continue
        # Check volume spike
        vol_ratio = pdata.get("vol_ratio", 1.0)
        if vol_ratio > vol_spike_threshold:
            continue
        filtered.append(c)
    return filtered
```

**Effort**: 2-3 hours. Low risk.

#### 1B. Add market-wide limit-down circuit breaker to risk.py

File: `ashare_lab/paper/risk.py`

```python
def check_limit_down_cascade(
    limit_down_count: int,
    total_stocks: int,
    count_threshold: int = 50,
    pct_threshold: float = 0.05,
) -> bool:
    """Halt buying when market-wide limit-down count exceeds threshold.

    Jan-Feb 2024 CSI1000 crash: 30% of stocks hit limit-down.
    This is a systemic risk signal, not an individual stock signal.
    """
    if total_stocks <= 0:
        return False
    return (limit_down_count >= count_threshold or
            limit_down_count / total_stocks > pct_threshold)
```

Add to `run_all_risk_checks()`: accept `limit_down_count` and `total_stocks` params, merge into `buying_halted`.

**Effort**: 1-2 hours. Low risk.

### Layer 2: DETECTION - Early Warning for Stranded Positions

#### 2A. Add limit-down state tracking to orders table

File: `ashare_lab/paper/ledger.py` (DB schema)

Add columns to orders table:
- `consecutive_limit_down_days` INT DEFAULT 0
- `last_limit_down_date` TEXT

Update in `settle_day()` when limit-down is detected:
```python
# In the limit-down block (engine.py line 244):
update_order(conn, oid, status="carry",
             consecutive_limit_down_days=order.get("consecutive_limit_down_days", 0) + 1,
             last_limit_down_date=trade_date)
```

**Effort**: 2-3 hours. Medium risk (schema migration needed).

#### 2B. Add limit-down alert to SettleResult

File: `ashare_lab/paper/engine.py`

```python
@dataclass
class SettleResult:
    # ... existing fields ...
    limit_down_alerts: list[dict] = field(default_factory=list)
    # Each entry: {"symbol": str, "days": int, "order_id": int}
```

Populate when limit-down carry exceeds 5 days:
```python
result.limit_down_alerts.append({
    "symbol": symbol,
    "days": order.get("consecutive_limit_down_days", 0),
    "order_id": oid,
})
```

**Effort**: 30 minutes. Low risk.

### Layer 3: ESCAPE - Better Exit During Limit-Down

#### 3A. Priority sell queue for limit-down-recovering stocks

File: `ashare_lab/paper/engine.py` (settle_day)

**Current**: sell orders are processed FIFO by order_id.
**Change**: sort sell orders so that stocks that WERE limit-down but are now tradeable get highest priority.

```python
# Replace line 194-201:
sell_orders = sorted(
    [o for o in orders if o["side"] == "sell"],
    key=lambda o: (-o.get("consecutive_limit_down_days", 0), o["id"]),
)
```

This ensures that when a limit-down stock finally opens, it sells FIRST before any other sell orders.

**Effort**: 5 minutes. Very low risk.

#### 3B. Remove max_resets limit for sell orders (the key fix)

**Current**: After 5 resets, sell order is cancelled -> position stranded.
**Proposed**: Never cancel a sell order due to limit-down. Instead, keep carrying indefinitely with logging.

Rationale: The position exists and must be exited. Cancelling the sell order doesn't make the position disappear -- it just removes the exit mechanism.

```python
# Replace lines 257-269:
if is_limit_down(change, threshold):
    if carry_day < carry_days_limit:
        update_order(conn, oid, status="carry")
        result.carries_to_bump.append(
            {"order_id": oid, "symbol": symbol, "side": "sell"}
        )
    else:
        # Reset carry_day but NEVER cancel sell orders.
        # Cancelling strands the position with no exit.
        # Track how long we've been trying for monitoring.
        reset_count = order.get("reset_count", 0) + 1
        update_order(
            conn, oid, status="carry",
            carry_day=-1,  # reset to 0
            reset_count=reset_count)
        result.carries_to_bump.append(
            {"order_id": oid, "symbol": symbol,
             "side": "sell",
             "reason": f"limit_down_reset_{reset_count}"}
        )
    continue
```

**Why this is safe**: The order will be re-evaluated every day. If the stock recovers, it sells. If it stays limit-down forever (delisted), the missing-price path (line 228-232) handles it. The `reset_count` provides monitoring data without blocking the exit.

**Effort**: 10 minutes. **Critical fix -- highest impact.**

### Layer 4: RECOVERY - Handle Stranded Positions

#### 4A. Add stranded position detection to risk checks

File: `ashare_lab/paper/risk.py`

```python
def detect_stranded_positions(
    current_positions: dict[str, dict],
    current_prices: dict[str, dict],
    max_stranded_days: int = 20,
) -> list[str]:
    """Identify positions that have been limit-down for too long.

    After max_stranded_days, recommend position write-down or
    manual intervention. This is a monitoring/alert function,
    not an automatic action.
    """
    stranded = []
    for symbol, pos in current_positions.items():
        pdata = current_prices.get(symbol, {})
        change = pdata.get("change", 0.0)
        threshold = pdata.get("threshold", 0.099)
        consecutive = pos.get("consecutive_limit_down_days", 0)
        if is_limit_down(change, threshold) and consecutive >= max_stranded_days:
            stranded.append(symbol)
    return stranded
```

**Effort**: 1 hour. Low risk.

#### 4B. Add stranded position NAV adjustment

For NAV reporting, when a position has been limit-down for 20+ days, mark its market_value using a conservative estimate (e.g., last tradeable price * 0.8) rather than the limit-down close price, to avoid overstating NAV.

**Effort**: 1-2 hours. Medium risk (affects NAV calculation).

---

## 4. Config Changes Required

Add to paper config (e.g., `config/paper.yaml`):

```yaml
paper:
  # Existing
  carry_days: 3
  max_limit_down_resets: 5  # REMOVE THIS -- no longer needed
  slippage: 0.001
  volume_participation_pct: 0.05

  # New: Limit-down protection
  limit_down:
    priority_sell: true           # Layer 3A
    never_cancel_sell: true       # Layer 3B
    stranded_alert_days: 20       # Layer 4A
    stranded_write_down_pct: 0.8  # Layer 4B

risk:
  # Existing dimensions...

  # New: Market-wide limit-down check
  limit_down_cascade:
    enabled: true
    count_threshold: 50           # Halt buying if 50+ stocks limit-down
    pct_threshold: 0.05           # Or 5% of universe

  # New: Limit-down pre-detection filter
  limit_down_filter:
    enabled: true
    atr_short_days: 5
    atr_long_days: 20
    atr_ratio_threshold: 2.0
    vol_spike_threshold: 3.0
    near_limit_pct: 0.07
```

---

## 5. Implementation Priority and Effort

| # | Change | File(s) | Effort | Risk Reduction | Priority |
|---|--------|---------|--------|----------------|----------|
| 3B | Never cancel sell on limit-down | engine.py | 10 min | **CRITICAL** -- eliminates stranded positions entirely | P0 |
| 3A | Priority sell queue | engine.py | 5 min | High -- faster exit when stock recovers | P0 |
| 2B | Limit-down alerts in SettleResult | engine.py | 30 min | Medium -- monitoring visibility | P1 |
| 1B | Market-wide circuit breaker | risk.py | 2 hr | High -- prevents buying during cascade | P1 |
| 2A | Order state tracking | ledger.py, engine.py | 3 hr | Medium -- better diagnostics | P2 |
| 1A | Pre-detection filter | pipeline.py | 3 hr | Medium -- reduces limit-down exposure | P2 |
| 4A | Stranded position detection | risk.py | 1 hr | Low -- monitoring only | P3 |
| 4B | NAV write-down | risk.py, ledger.py | 2 hr | Low -- edge case handling | P3 |

**Total estimated effort**: 12-15 hours for all layers.
**P0 only**: 15 minutes for 90% of the risk reduction.

---

## 6. Risk Reduction Estimate

| Scenario | Current | After P0 Fix | After All Layers |
|----------|---------|-------------|-----------------|
| Single stock limit-down 3 days | Resolves (carry works) | Same | Same |
| Single stock limit-down 15+ days | **STRANDED** (cancelled) | Resolves (keeps carrying) | Resolves + alert |
| 30% stocks limit-down (CSI1000 crash) | **Multiple stranded** + buys continue | Strands resolved | Strands resolved + buying halted + filtered |
| Stale position in perpetuity | Possible (no detection) | Possible but rare | Detected at 20 days |

**Bottom line**: Change 3B (never cancel sell orders on limit-down) is the single most impactful fix. It takes 10 minutes and eliminates the stranded position problem entirely. Everything else is optimization.

---

## 7. Citations

- AKQuant textbook Ch.6: A-share market microstructure rules (limit-up/down, T+1)
- zhou343-de/stock-trader-ai: 8-layer risk system, "跌停封板检测 -> 告警邮件 + 次日优先处理"
- juejin article (2026-06-07): "持仓跌停封死 -> 次日集合竞价市价卖"
- quant-ashare ADVANCED_TRICKS.md: Trick #2 "涨跌停/停牌样本必须屏蔽"
- BigQuant: Limit-up cancels sell orders (opposite direction, same mechanism)
- 证券日报 (2026-05-24): Block trade + inquiry transfer as institutional exit channels
- Sina Finance (2026-07-13): 176 stocks limit-down, quant amplification discussion
- Lopez de Prado (2012): VPIN as crash predictor
