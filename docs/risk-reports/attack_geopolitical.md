# Geopolitical / War / Black Swan Attack Analysis

**Date**: 2026-07-15
**System**: ashare-lab paper trading engine
**Current NAV**: 300,000 RMB (100% CASH, zero positions)
**Universe**: CSI1000 (~2597 small-mid cap A-shares)
**Model**: w10.pt, trained 2026-07-01 (23.9 days old at analysis time)
**Hedge**: DISABLED (hedge.enabled=false in baseline.yaml:112)

---

## Executive Summary

The system is currently in **cold-start / zero-position state** (100% cash). This
dramatically limits first-order loss from any crash scenario. However, the analysis
exposes multiple **structural vulnerabilities** that would cause catastrophic losses
once the system accumulates positions, and several that matter even at zero-position
for the next trading cycle.

**Critical finding**: The system has no mechanism to detect or react to a black swan
event BETWEEN pipeline runs. All risk checks execute post-close (pipeline.py step 9).
If the system buys on day T-1 and a crash hits at day T open, the damage is already
done before any risk gate fires.

---

## Scenario 1: Taiwan Military Escalation (5-8% Single-Day Crash, Mass Limit-Down)

### Historical Precedent [KNOWN]
- 2015 Chinese crash: SSE dropped 40% in ~1 month, thousands of stocks hit limit-down
  simultaneously across multiple days (Bian et al. 2018, NBER WP 25040).
- 2024 quant quake: Bloomberg reported quant models "went haywire" during a 2-week
  crash; Man Group compared it to the 2007 US quant quake.
- Taiwan misfired-missile event: academic paper (Li et al. 2023, Australian Economic
  Papers) confirmed causal decline in Taiwan-related A-share stocks.

### Code Path Trace (assuming system holds 15 positions at ~18K each)

**Step 8 - settle_day (engine.py:170-583)**:
1. Sell orders processed first (engine.py:209). For each held stock:
   - `is_limit_down(change, threshold)` at engine.py:89-93 fires when
     change <= -threshold (0.099 for regular, 0.199 for ChiNext/STAR).
   - If limit-down: order gets CARRIED if carry_day < 3, else CANCELLED
     (engine.py:243-251).
   - **FAILURE MODE**: The sell order does NOT fill. The position is STUCK.
     After 3 carry days, the order is cancelled entirely -- the system
     gives up trying to sell.

2. Buy orders processed after sells (engine.py:337). For buy candidates:
   - `is_limit_up(change, threshold)` at engine.py:82-86 blocks buying.
   - In a crash, buy orders are irrelevant (prices dropping, not rising).

3. Volume cap `cap_fill_by_volume` (engine.py:128-143): even if a stock
   is NOT at limit-down, volume participation cap of 5% (config:89) means
   a 18K-position stock with normal volume can only partially fill.

**Step 9 - risk_checks (risk.py:218-360)**:
1. `check_daily_loss` (risk.py:89-99): threshold = 3% (config:101).
   A 5-8% crash on a 15-stock portfolio concentrated in CSI1000 small-caps
   (which fall harder than the index) likely exceeds 3% daily loss.
   -> `buying_halted = True` for the NEXT day.

2. `check_drawdown_breaker` (risk.py:76-86): threshold = 15% (config:100).
   Only fires after cumulative 15% drawdown from peak NAV.

3. `check_trailing_stop` (risk.py:138-147): threshold = 20% (config:105).
   Only fires after 20% decline from holding-period high. In a single 5-8%
   day, this would NOT fire (positions drop 5-8%, not 20%).

4. `check_market_regime` (risk.py:118-135): threshold = 8% decline over
   10 trading days (config:103-104). A single-day 5-8% crash may or may
   not trigger this depending on prior 9 days.

5. `check_soft_drawdown` (risk.py:182-210): threshold = 10% (config:108).
   If cumulative drawdown >10%, reduces topk from 15 to 7.

### Failure Mode Summary

**P1 -- Sell-side paralysis (limit-down lock-in)**: In a mass limit-down
event, most/all held stocks hit limit-down. Sell orders CANNOT execute
(engine.py:243: `is_limit_down(change, threshold)` -> carry/cancel).
The system is locked into positions for up to 3 days. During those 3 days,
the stocks can drop another 10% per day. Worst case: 3 consecutive
limit-down days = ~27% cumulative loss on regular stocks, ~49% on
ChiNext/STAR stocks.

**P2 -- Trailing stop too slow**: Trailing stop at 20% (config:105) is
designed for gradual declines, not gap events. A single 10% limit-down
day does NOT trigger it. The system needs 2+ days of limit-down before
the trailing stop fires, but by then sell orders cannot execute anyway.

**P2 -- No intraday risk management**: All risk checks are post-close.
The system cannot react to an intraday crash. If it entered positions
yesterday and a geopolitical event hits at market open, the damage is
done before the pipeline runs at 18:00.

### Maximum 1-Day Loss Estimate [COMPUTED]

**If system held 15 positions (full deployment)**:
- Each position: 300K * 0.95 / 15 = 19,000 RMB
- CSI1000 small-mid caps in a geopolitical crash: beta ~1.3-1.5x index
- A-share index drops 5-8%, CSI1000 drops ~7-12%
- Average position loss: ~7-12% * 19,000 = 1,330 - 2,280 per stock
- Total portfolio loss: 15 * 1,330 to 2,280 = **19,950 - 34,200 RMB**
- As % of NAV: **6.7% - 11.4%**
- Risk gate: daily_loss (3%) fires -> buying halted next day
- But positions are STUCK due to limit-down -> cannot sell

**If positions held for multiple days (worse case)**:
- Day 1: -7% (limit-down lock-in)
- Day 2: -10% (continued limit-down, sell orders carried)
- Day 3: -10% (sell orders cancelled after 3-day carry limit)
- Cumulative: ~25% on regular stocks, ~49% on ChiNext/STAR
- Total loss: **75,000 - 147,000 RMB** (25-49% of NAV)
- Severity: **P1** (>50% loss possible on ChiNext-heavy portfolio)

### Current State Mitigation [KNOWN]
The system is currently 100% cash (zero positions). A crash today causes
ZERO direct loss. However, if the pipeline runs today and generates buy
orders for tomorrow, those orders execute into the crash -- buying stocks
that immediately limit-down. The next-day settle would lock in losses.

---

## Scenario 2: Trade War 2.0 (Sudden Sector Rotation)

### Code Path Trace

**Prediction layer (predict.py:29-284)**:
- Model w10.pt trained on 2018-2025 data (walk-forward windows).
- Alpha158 features (predict.py:168-175): 158 technical factors based on
  price/volume history. These are BACKWARD-LOOKING -- they capture the
  regime the model was trained on.
- TRA routing (3 states, config:29): the model has 3 "regime" patterns.
  A sudden sector rotation (yesterday's winners become today's losers)
  may not match any trained regime.

**Blend layer (blend.py:29-105)**:
- 60% raw TRA + 40% neutralized predictions.
- Style neutralization (blend.py:89-90) regresses against Barra-style
  size/volatility/momentum factors. In a regime change, the factor
  relationships learned during training break down.
- **FAILURE MODE**: The neutralization assumes stable factor loadings.
  A geopolitical shock that re-prices entire sectors (e.g., tech sanctions)
  invalidates the regression coefficients.

**Signal layer (signal.py:29-61)**:
- TopkDropout with n_drop=1 (config:59): only 1 stock exits per day.
- **FAILURE MODE**: In a sudden rotation, the system holds 14 yesterday's
  winners that are now crashing, and can only exit 1 per day.
- At n_drop=1, it takes 14 trading days to fully rotate the portfolio.
  In a crash, 14 days = permanent capital destruction.

**Risk layer**:
- `check_market_regime` (risk.py:118-135): 8% decline over 10 days
  halts buying. Does NOT halt selling or force liquidation.
- `check_trailing_stop` (risk.py:138-147): 20% trailing stop fires
  full-position exit. But by the time a stock drops 20% in a rotation,
  the damage is done.

### Failure Mode Summary

**P1 -- n_drop=1 is dangerously slow for regime changes**: The system
can only exit 1 position per day (signal.py:51-53, sell_cands[:n_drop]).
In a sector rotation where all 15 positions are wrong, it takes 14 days
to rotate. Each day, the remaining positions lose more. This is the
single most dangerous design choice for black swan events.

**P2 -- Model stale + no PSI monitoring**: Model is 23.9 days old
(predict.py:119: _MODEL_STALE_DAYS=7 threshold already exceeded).
No Population Stability Index (PSI) monitoring exists. The model cannot
detect that the factor distribution has shifted.

**P2 -- Neutralization assumes stationarity**: The blend's style
neutralization (blend.py:89-90) uses factor loadings from the training
window. In a regime change, these loadings are stale, potentially
AMPLIFYING the wrong signal.

### Maximum 1-Day Loss Estimate [COMPUTED]

**If system held 15 positions after a sector rotation trigger**:
- Day 1: positions drop 3-5% (rotation, not crash)
- n_drop=1: only 1 stock exits, 14 remain
- Day 2-5: remaining positions continue declining 2-3% per day
- Cumulative 5-day loss: ~15-25% on the portfolio
- Total loss: **45,000 - 75,000 RMB** (15-25% of NAV)
- Severity: **P2** (20-50% loss)

---

## Scenario 3: COVID-Style Black Swan (Market Halt, Liquidity Vanishes)

### Code Path Trace

**Settle engine (engine.py:170-583)**:
1. `is_suspended(volume)` at engine.py:96-100: returns True when volume=0.
   - Sell orders on suspended stocks -> CARRY (engine.py:235-240).
   - Buy orders on suspended stocks -> CARRY (engine.py:370-375).

2. After 3 carry days, orders are CANCELLED (engine.py:248-251).
   - **FAILURE MODE**: Stocks can be suspended for weeks (2015 crash
     saw suspensions lasting months). The system gives up after 3 days.

3. `cap_fill_by_volume` (engine.py:128-143): 5% participation cap.
   - In a liquidity crisis, volume drops to near-zero.
   - Even non-suspended stocks may have volume so low that fill_qty=0.
   - -> CARRY -> CANCEL after 3 days.

**Risk layer**:
- `check_market_regime` (risk.py:118-135): fires after 8% decline over
  10 days. In COVID-style crash, this fires within days.
- `check_drawdown_breaker` (risk.py:76-86): 15% hard halt.
- `check_daily_loss` (risk.py:89-99): 3% daily halt.

**Hedge sleeve (hedge.py:110-195)**:
- Currently DISABLED (config:112: hedge.enabled=false).
- Even if enabled: `max_single_day_rebalance=0.30` (config:119) caps
  daily rebalancing at 30% of NAV. Cannot flee to safety in one day.
- `anti_whipsaw_days=10` (config:117): cannot exit hedge for 10 days
  after activation, even if the crash is over.

### Failure Mode Summary

**P0 -- Total position lock-in**: In a COVID-style halt:
1. All held stocks suspended (volume=0).
2. Sell orders CARRY for 3 days, then CANCEL.
3. System cannot exit ANY position.
4. Positions remain in portfolio at stale/zero valuations.
5. NAV computation (compute_nav) uses last known prices, which are
   stale. The reported NAV is a LIE -- it overstates true value.

**P1 -- Liquidity death spiral**: Even stocks that aren't officially
suspended may have volume so low that cap_fill_by_volume returns 0.
The 5% participation cap (config:89) means a stock with 100K daily
volume can only sell 5,000 shares per day. For a 19K position at
10 RMB/share = 1,900 shares, this is fine. But for a stock at 1 RMB/share
= 19,000 shares, the cap blocks the full exit.

**P1 -- Hedge sleeve too slow**: Even if enabled, the hedge sleeve
cannot protect against a sudden halt. The ramp (config:120-123) moves
from 95% equity to 20% equity linearly across 3-15% drawdown. In a
COVID crash where the market drops 8% in one day, the sleeve would
target ~80% equity -- still heavily exposed.

### Maximum 1-Day Loss Estimate [COMPUTED]

**COVID-style: market drops 8%, then halts for 3 days**:
- Day 1: -8% on all positions. Loss: 24,000 RMB (8% of 300K).
- Days 2-4: market halted. Positions frozen. No pipeline runs.
- Day 5: market reopens, another -5%. Loss: additional 12,000 RMB.
- Total: **36,000 RMB** (12% of NAV).
- But if suspended stocks resume lower: **50,000-80,000 RMB** (17-27%).
- Severity: **P1** (>50% loss possible if halt lasts >1 week)

---

## Scenario 4: Sanctions on China Tech (Overnight Factor Regime Change)

### Code Path Trace

**Model layer (predict.py)**:
- Alpha158 features (predict.py:168-175) include momentum, volatility,
  and volume factors. Sanctions that re-price tech stocks overnight
  create a discontinuity in these factors.
- TRA routing (3 states, config:29): the model's 3 regime patterns were
  trained on historical data that does NOT include a sanctions regime.
- **FAILURE MODE**: The model has never seen a world where tech stocks
  are sanctioned. Its predictions are based on patterns from a pre-sanctions
  universe. It will continue recommending tech stocks because that's what
  worked in training.

**Blend layer (blend.py)**:
- Style neutralization against size/volatility/momentum factors.
- Sanctions would cause: (a) tech sector crash, (b) non-tech rally,
  (c) factor correlations break.
- The neutralization regression (blend.py:89-90) uses historical factor
  loadings. Post-sanctions, these loadings are meaningless.

**Universe layer**:
- CSI1000 (config:2: primary universe) includes tech stocks.
- No sector exclusion mechanism exists in the signal layer.
- The industry concentration check (risk.py:169-179, config:107: 30% cap)
  limits single-industry exposure to 30% of NAV, but does not exclude
  sanctioned sectors entirely.

### Failure Mode Summary

**P1 -- Model blind spot**: The TRA model has 3 regime patterns, none
of which include a sanctions regime. It will continue generating positive
scores for tech stocks that are fundamentally impaired. The 60/40 blend
offers no protection because neutralization assumes stable factor
relationships.

**P2 -- Slow exit**: n_drop=1 means the system exits 1 tech stock per
day. If 8 of 15 positions are tech, it takes 8 days to exit. Each day,
the remaining tech positions lose more.

**P2 -- No fundamental overlay**: The system is purely quantitative
(Alpha158 technical factors). It has no fundamental analysis, no sector
blacklist, no news-driven override. The sentiment module
(pipeline.py:1314-1329) exists but is optional and its LLM-based veto
may not catch sanctions news fast enough.

### Maximum 1-Day Loss Estimate [COMPUTED]

**If 8 of 15 positions are tech-related**:
- Tech stocks drop 10% (limit-down) on sanctions news.
- Non-tech stocks flat or +2%.
- Portfolio loss: 8 * 19,000 * 10% = 15,200 (tech) - 7 * 19,000 * 2%
  = 2,660 (non-tech gain) = **12,540 RMB net loss** (4.2% of NAV).
- Over 5 days (slow exit via n_drop=1): tech drops another 10-20%.
- Total: **30,000 - 50,000 RMB** (10-17% of NAV).
- Severity: **P2** (20-50% loss if tech concentration is high)

---

## Cross-Cutting Vulnerabilities

### V1: No Intraday Risk Management [KNOWN]
All risk checks execute post-close (pipeline.py step 9). The system
cannot react to intraday events. If a crash happens at 10:00 AM, the
system doesn't know until 18:00 PM when the pipeline runs.

**Impact**: 8 hours of unmonitored exposure during market hours.

### V2: T+1 Constraint + n_drop=1 = Trapped [COMPUTED]
A-share T+1 rule (stocks bought today cannot be sold until tomorrow)
combined with n_drop=1 means:
- Day T: buy 15 stocks
- Day T+1: crash hits. Can only sell 1 stock (n_drop=1). T+1 blocks
  stocks bought on day T.
- Day T+2: can sell 1 more stock. But 13 positions still trapped.
- Full exit takes 14+ trading days.

### V3: Hedge Disabled [KNOWN]
hedge.enabled=false (baseline.yaml:112). The defensive sleeve (treasury
ETF 511260, gold ETF 518880, money market 511990) is not active. Even
if enabled, the ramp is too slow for gap events (hedge.py docstring:
"soft floor -- protects against progressive drawdowns but cannot
guarantee against a single-day gap").

### V4: Model Staleness [KNOWN]
Model w10.pt is 23.9 days old (predict.py:119 threshold: 7 days).
The model was trained on data ending ~2026-06-30. Any structural change
after that date (geopolitical event, policy shift) is invisible to the
model.

### V5: IC Always Null for Live Predictions [KNOWN]
predict.py:236-255: IC is NaN for out-of-sample predictions. The system
has NO signal quality metric for live trading. It cannot detect when
the model's predictions have become meaningless.

### V6: Price Data Dependency on qlib Subprocess [INFERRED]
pipeline.py:674-728: prices are fetched via a qlib subprocess with 90s
timeout. In a market crash:
- qlib data feed may be delayed (exchange data feeds lag during crashes).
- The 90s timeout may expire, leaving prices empty.
- Empty prices -> trailing stops cannot compute -> risk checks degrade.

### V7: Report Gate Blocks Operator Notification [KNOWN]
pipeline.py:1512: `if prev_nav > 0 and abs(today_nav - prev_nav) / prev_nav > 0.20`
-> `return` (line 1515). Report quality gate skips report if NAV changed
>20% in one day. In a black swan where NAV drops >20%, the report is
SUPPRESSED. The operator does not get notified of the crash via WeChat.

This is a critical design flaw: the gate was designed to prevent false
reports from data errors (e.g., qlib returning normalized $close), but
it also blocks legitimate crash alerts. The `abs()` means it also blocks
suspiciously large gains, which is correct -- but blocking large losses
is the opposite of what an operator needs in a crash.

**Fix**: change to `abs(today_nav - prev_nav) / prev_nav > 0.20 and
today_nav > prev_nav` -- only suppress suspiciously large GAINS, never
suppress large LOSSES. Or: send a "CRASH ALERT" instead of suppressing.

---

## Risk Priority Matrix

| Scenario | Max 1-Day Loss (RMB) | Max Multi-Day Loss | Severity | Likelihood |
|----------|---------------------|-------------------|----------|------------|
| Taiwan escalation (positions held) | 19,950 - 34,200 | 75,000 - 147,000 | P1 | Low |
| Taiwan escalation (current: cash) | 0 | 0 (but buys into crash) | P3 | Low |
| Trade war rotation | 5,700 - 9,500 | 45,000 - 75,000 | P2 | Medium |
| COVID black swan | 24,000 | 50,000 - 80,000 | P1 | Low |
| Tech sanctions | 12,540 | 30,000 - 50,000 | P2 | Medium |

---

## Recommended Mitigations (Priority Order)

1. **Enable hedge sleeve** (baseline.yaml:112: hedge.enabled=true).
   Even imperfect protection beats zero protection.

2. **Increase n_drop to 3-5** during high-volatility regimes. The current
   n_drop=1 (config:59) is a death trap in rapid rotations. Dynamic
   n_drop based on market_regime check would help.

3. **Add intraday circuit breaker**: monitor real-time prices via akshare
   or eastmoney API during market hours. If portfolio drops >5% intraday,
   generate emergency sell orders immediately.

4. **Fix report gate**: do NOT suppress reports when NAV drops >20%.
   Instead, tag as "CRASH ALERT" and send with higher priority.

5. **Add PSI monitoring**: detect when the model's feature distribution
   has shifted from training data. Auto-halt buying when PSI > 0.2.

6. **Reduce trailing stop to 10-12%** for CSI1000 small-caps. The
   current 20% threshold is too wide for stocks with 10% daily limits.

7. **Add sector blacklist mechanism**: allow config-driven sector
   exclusions (e.g., "exclude tech if sanctions detected").

8. **Shorten carry_days_limit from 3 to 1** for sell orders. In a crash,
   3 days of carry = 3 days of additional losses. The system should
   retry sells more aggressively.
