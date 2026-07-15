# Regulatory / Policy Shock Attack Analysis

**Date**: 2026-07-15
**System**: ashare-lab paper trading (CSI1000, NAV ~280K RMB)
**Scope**: 6 regulatory shock scenarios traced to exact code paths

---

## Scenario 1: IPO Suspension

**What happens**: CSRC halts all IPO approvals (historical precedent: 2015-07 to 2015-11, 2022-04, 2023-08). No new stocks list. Existing universe unchanged.

### Code path trace

**Impact on pipeline**: LOW -- system degrades gracefully.

1. **Step 5b** (`pipeline.py:623-630`): `_fetch_ipo_calendar(next_td)` calls eastmoney API for next trading day's listings. On IPO suspension, the API returns empty list. `ipo_calendar` = `[]`.

2. **Step 5b continued**: `ipo_won` = `[]` (no rows pass `check_ipo_subscription` filter). `ipo_listing_syms` = `set()`.

3. **Step 11** (`pipeline.py:1387-1487`): `_step11_ipo_processing` iterates `ipo_won` -- zero iterations. No cash deduction, no position creation, no sell orders generated for IPO stocks.

4. **Step 7** (`pipeline.py:1024-1038`): `_step7_csi1000_exits` checks if current positions left the CSI1000 universe. IPO suspension does NOT remove existing stocks from the index. No forced sells.

### Failure mode

**No failure.** The system's IPO handling is already designed for zero-IPO days. The `_fetch_ipo_calendar` function returns `[]` on HTTP failure (`pipeline.py:317-319`) or empty data (`pipeline.py:323-325`). IPO processing is purely additive -- when absent, the pipeline runs identically to a normal day with no new listings.

### Indirect risk

The TRA model was trained on historical data that includes IPO pop patterns. If IPOs resume after a long suspension, the model's IPO-day prediction patterns may be stale (model age > 7 days triggers a warning at `pipeline.py:1251-1255`). But this affects prediction quality, not pipeline correctness.

**Severity: LOW** | Max impact: None (pipeline operates normally)

---

## Scenario 2: T+1 Rule Change to T+0

**What happens**: CSRC changes settlement from T+1 to T+0. Stocks bought today can be sold today. This is a fundamental market structure change.

### Code path trace

**Impact on pipeline**: MEDIUM -- the T+1 sell guard becomes over-restrictive but not broken.

1. **Step 10** (`pipeline.py:1332-1335`): T+1 sell guard:
   ```python
   sell_syms = [
       s for s in sell_syms
       if ctx.current_positions.get(s, {}).get("buy_date") != ctx.trade_date
   ]
   ```
   Under T+0, this guard prevents same-day sells that are now LEGAL. The system will carry orders that could have been filled today.

2. **Engine settle** (`engine.py:208-332`): The settle_day function processes sells then buys. It does NOT enforce T+1 at the engine level -- it relies on the pipeline's T+1 sell guard to prevent same-day sells. Under T+0, the engine would correctly fill a same-day sell if the order reached it.

3. **buy_date tracking** (`engine.py:508`): New positions get `buy_date: trade_date`. This field exists solely for the T+1 guard.

4. **Order carry logic** (`engine.py:245-252`): If a sell order is blocked by limit-down, it carries for up to 3 days. Under T+0, a same-day sell that is blocked by the T+1 guard would NOT reach this logic -- it is filtered out before settle.

### Failure mode

**Over-restriction, not breakage.** The system will:
- NOT sell stocks bought on the same day (even though legal under T+0)
- Carry these orders to the next day, where they will execute normally
- This means 1 day of delayed execution for any stock that enters and exits the portfolio on the same day

**Impact estimate**: At n_drop=1, at most 1 stock per day could be affected. With a 15-stock portfolio, this is a <7% turnover delay. The system still functions; it just misses one day of execution speed for a small number of stocks.

### Required fix

Remove the T+1 sell guard in `pipeline.py:1332-1335` or make it conditional on a config flag. The engine itself is already T+0 compatible.

**Severity: MEDIUM** | Max impact: 1-day delayed execution for n_drop stocks per day, ~0.1% slippage cost

---

## Scenario 3: Short Selling Ban

**What happens**: CSRC bans or severely restricts short selling (historical: 2015-07 partial ban, 2024-01 restrictions on quant funds using margin/short). Market drops sharply as hedging mechanism disappears.

### Code path trace

**Impact on pipeline**: LOW for the system itself, HIGH for the portfolio via market impact.

1. **Hedge sleeve** (`pipeline.py:1116-1176`, `hedge.py`): The hedge sleeve uses LONG positions in defensive ETFs (treasury 511260, gold 518880, money market 511990). It does NOT use short selling. The hedge config at `baseline.yaml:112-126` shows `enabled: false` (locked to 08-02 per PM state). Even when enabled, hedge legs are all long ETF positions.

2. **No short positions anywhere**: The engine (`engine.py`) has no short-selling logic. All positions have positive `qty`. The settle function only processes sell orders for existing long positions.

3. **Market regime filter** (`risk.py:118-135`): If the ban triggers a market crash, the CSI1000 regime check fires:
   ```python
   cumulative_return = (window[-1] / window[0]) - 1
   return cumulative_return < -decline_threshold
   ```
   With `market_regime_decline: 0.08` and `market_regime_days: 10`, a >8% drop in 10 days halts buying.

4. **Drawdown breaker** (`risk.py:76-86`): Hard halt at 15% drawdown. A severe crash (like 2015's -40% in 3 weeks) would trigger this within days.

5. **Trailing stop** (`risk.py:138-147`): 20% trailing stop per position. In a crash, multiple positions hit stop simultaneously, generating forced sells.

### Failure mode

**The system itself does not fail.** It correctly activates risk controls:
- Buying halts after 8% market decline (regime filter)
- Hard halt at 15% drawdown
- Forced sells via trailing stops at 20% per-position decline
- Soft drawdown reduces topk from 15 to 7 at 10% drawdown

**The portfolio loses money** -- this is the intended behavior. The risk controls limit losses but cannot prevent them in a crash. With 15 stocks averaging 6.7% weight each, a 30% sector drop = ~10% portfolio loss before risk controls activate.

**Severity: HIGH (market risk, not system risk)** | Max impact: 15% drawdown (hard halt), then system stops buying

---

## Scenario 4: Program Trading Restrictions (Account Flagged)

**What happens**: Exchange flags the account for "disruptive program trading" (precedent: Lingjun Investment, Feb 2024 -- suspended 3 trading days). Account is restricted from placing new orders for N days.

### Code path trace

**Impact on pipeline**: The system is PAPER TRADING -- it does not place real orders. The restriction applies to a real brokerage account, not the simulation.

1. **Paper trading nature**: The engine (`engine.py`) simulates fills using `settle_day()` -- it writes to SQLite, not to a broker API. There is no QMT connection, no broker API call, no real order submission.

2. **Pipeline shell script** (`ashare-pipeline.sh`): The script runs `python3 -m ashare_lab.cli paper run-all` -- a local simulation. No broker connection.

3. **If this were real trading**: The system has no broker integration layer. There is no code path that submits orders to QMT, CTP, or any broker API. The gap between paper and real is the protection here.

### Failure mode for paper trading

**No impact.** The regulatory restriction targets real trading accounts. The paper engine continues to simulate normally.

### If real trading were added (future risk)

The system would need:
- A broker integration layer with order submission
- Error handling for account restriction responses (HTTP 403 / specific error codes)
- A fallback: if the broker rejects orders, the system should detect this and either:
  - Halt the pipeline (conservative)
  - Continue simulation but log "unfilled" status (optimistic)

**Severity: NONE (paper trading)** | Max impact: None

---

## Scenario 5: Sector Crackdown (Tech 2021 Style)

**What happens**: Government announces sweeping regulation of a sector (e.g., tech platform companies 2021, education 2021, real estate 2022). The affected sector drops 30-70% over days/weeks. Stocks in the sector may hit limit-down repeatedly.

### Code path trace

**Impact on pipeline**: MEDIUM -- system handles this through existing risk controls, but with specific gaps.

1. **Industry concentration** (`risk.py:169-179`): `check_industry_concentration` blocks NEW buys when industry exposure > 30% of NAV. But it does NOT force-sell existing positions in the affected industry. Existing positions are only sold via trailing stop (20% per-stock decline) or forced sells for concentration (>15% per-stock NAV cap).

2. **Limit-down blocking** (`engine.py:242-252`): Sell orders for limit-down stocks are carried for up to 3 days, then cancelled:
   ```python
   if is_limit_down(change, threshold):
       if carry_day < carry_days_limit:
           update_order(conn, oid, status="carry")
       else:
           update_order(conn, oid, status="cancelled")
   ```
   A stock that hits limit-down for 3 consecutive days has its sell order CANCELLED. The system stops trying to sell it. The position remains in the portfolio, continuing to lose value.

3. **Trailing stop** (`risk.py:138-147`): Triggers at 20% decline from holding high. But if the stock is limit-down, the sell order cannot execute (engine.py:242-252). The trailing stop generates the order, but the engine carries/cancels it.

4. **Specific failure chain**:
   - Day 1: Sector drops 10% (limit-down). Trailing stop does not trigger yet (need 20%).
   - Day 2: Sector drops another 10% (total -19%). Still below trailing stop threshold.
   - Day 3: Sector drops another 10% (total -27%). Trailing stop triggers. Sell order generated.
   - Day 3 settle: Stock is limit-down. Sell order carried (carry_day=0).
   - Day 4: Stock limit-down again. Sell order carried (carry_day=1).
   - Day 5: Stock limit-down again. Sell order carried (carry_day=2).
   - Day 6: Stock limit-down again. Sell order CANCELLED (carry_day=3 = limit).
   - Position remains in portfolio. No further sell attempt.

5. **Industry cap only blocks buys** (`pipeline.py:1310-1313`):
   ```python
   buy_syms = [
       s for s in buy_syms
       if ctx.industry_map.get(s) not in ctx.risk_result.blocked_industries
   ]
   ```
   This prevents buying MORE of the affected sector. But existing positions are not force-sold by the industry check.

### Failure mode

**The system cannot exit positions that hit limit-down for 3+ consecutive days.** This is the most dangerous regulatory scenario:

- The 3-day carry limit is hardcoded in `baseline.yaml:88` (`carry_days: 3`)
- After 3 days, the sell order is cancelled and the system gives up
- The position continues to lose value with no exit mechanism
- The system would need to re-generate a sell order on a future day when the stock is NOT limit-down, but the TopkDropout logic (`signal.py:29-61`) only generates sell signals for positions not in top-K -- if the crashed stock is already out of top-K, the sell was already generated on day 3 and cancelled

**Impact estimate**: If a stock drops 50% over 10 days (realistic in a sector crackdown), and the system exits after 3 days of limit-down at -27%, the position is stuck. The stock continues to drop to -50%, but the system has cancelled the sell order. Unrealized loss: ~50% of position value. For a 15-stock portfolio with one affected stock at ~6.7% weight: ~3.3% NAV impact beyond what risk controls captured.

### Required fix

After carry_days_limit is reached, do NOT cancel the sell order. Instead, reset carry_day to 0 and keep the order in "pending" status for the next trading day when the stock is not limit-down. The current logic (`engine.py:249-251`) permanently cancels the order, which is the bug.

**Severity: HIGH** | Max impact: ~3-5% additional NAV loss from stuck positions in a sector crackdown

---

## Scenario 6: QMT/Account Restriction (Broker Restricts Account Mid-Trade)

**What happens**: Broker restricts the account (margin call, compliance freeze, technical issue). This is the same as Scenario 4 for paper trading, but let's analyze the broader system resilience.

### Code path trace

**Impact on pipeline**: NONE for paper trading. The system has no broker connection.

1. **No broker integration**: As established in Scenario 4, the system is purely paper-based. No QMT, no CTP, no broker API calls exist in the codebase.

2. **Data pipeline dependency**: The system depends on:
   - qlib data (local, via subprocess) -- unaffected by broker restrictions
   - baostock API (remote, for industry/benchmark data) -- independent of broker
   - eastmoney API (for IPO calendar) -- independent of broker
   - GPU inference (local network, 192.168.100.11) -- independent of broker

3. **Alert delivery**: The report is sent via hermes-gateway to WeChat (`report.py:578-598`). This is independent of broker status.

### Failure mode

**No failure for paper trading.** The system continues to simulate, generate signals, and report to WeChat regardless of any broker restriction.

### If real trading were added (future risk)

The system would need:
- A health check on the broker connection before pipeline execution
- A "broker unreachable" error path that halts the pipeline (not silently continues)
- Position reconciliation: if the broker rejects some orders but not others, the paper state diverges from real state

**Severity: NONE (paper trading)** | Max impact: None

---

## Summary Matrix

| Scenario | Severity | System Impact | Portfolio Impact | Code Path |
|----------|----------|---------------|-----------------|-----------|
| 1. IPO Suspension | LOW | None (graceful) | None | pipeline.py:623-630 |
| 2. T+1 to T+0 | MEDIUM | Over-restrictive | ~0.1% slippage | pipeline.py:1332-1335 |
| 3. Short Selling Ban | HIGH | Risk controls activate | Up to 15% drawdown | risk.py:76-86, 118-135 |
| 4. Account Restriction | NONE | Paper trading unaffected | None | N/A (no broker) |
| 5. Sector Crackdown | HIGH | Stuck positions | ~3-5% extra loss | engine.py:249-251 |
| 6. QMT Restriction | NONE | Paper trading unaffected | None | N/A (no broker) |

---

## Recommendations

### Critical (fix before live trading)

1. **Carry-days limit bug** (`engine.py:249-251`): After 3 days of limit-down, sell orders are permanently cancelled. This is the most dangerous gap. Fix: reset carry_day and keep order pending instead of cancelling.

2. **No broker health check**: If real trading is ever added, the system needs broker connectivity verification before pipeline execution.

### Important (fix during paper trading)

3. **T+1 guard configurability** (`pipeline.py:1332-1335`): Make the T+1 sell guard conditional on a config flag so it can be disabled if rules change.

4. **Industry force-sell**: The industry concentration check blocks buys but does not force-sell existing over-concentrated positions. In a sector crackdown, this means the system holds losing positions indefinitely.

### Low priority

5. **Model retraining trigger**: Regulatory changes invalidate historical patterns. The IC-based retraining trigger (`pipeline.py:1284`) should also fire on regime-breaking events (e.g., sector drops > 30% in 5 days).

6. **Hedge sleeve activation**: Currently disabled (`enabled: false`). When enabled, it uses long ETF positions (treasury, gold, money market) that are independent of short-selling rules. No regulatory risk here.
