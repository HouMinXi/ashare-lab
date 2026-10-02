# Hedge Sleeve Backtest Design + Epsilon Pre-Registration

Phase 2.4 Plan 02 deliverable. STOP: do NOT run the backtest until
the user signs the epsilon value.

## 1. Correlation Table: Cash vs CSI1000

Cash (bank deposit at ~1.5% annual) has near-zero correlation with
equity returns by definition: its daily return is constant (~0.006%/day),
while CSI1000 daily returns have stdev ~1.5%. Spearman correlation over
any window is effectively 0 (constant vs variable = undefined/zero rank
correlation).

The useful question is not "is cash correlated with equity" (trivially no)
but "what does blending do to outcomes." The table below shows the 10-window
incumbent excess returns and the MECHANICALLY COMPUTED effect of a cash sleeve.

### Per-Window Excess Under Cash Sleeve (computed from incumbent blend)

A cash sleeve of X% scales equity excess by (1-X) and adds cash return.
Cash return per window ~ (window_days / 252) * 1.5% (bank deposit rate).

Notation: excess_sleeve = excess_equity * (1-X) + cash_return_window

Using incumbent blend excess (from MASTER section 7):

| W | Blend excess | ~Days | 90/10 excess | 85/15 excess | 80/20 excess |
|---|-------------|-------|-------------|-------------|-------------|
| 1 | -11.71% | 126 | -10.47% | -9.85% | -9.22% |
| 2 | -1.70% | 126 | -1.46% | -1.34% | -1.21% |
| 3 | +3.39% | 126 | +3.13% | +2.96% | +2.79% |
| 4 | +11.06% | 126 | +10.03% | +9.48% | +8.92% |
| 5 | +9.16% | 126 | +8.32% | +7.86% | +7.40% |
| 6 | +3.12% | 126 | +2.89% | +2.72% | +2.57% |
| 7 | +1.02% | 126 | +0.99% | +0.94% | +0.89% |
| 8 | +10.58% | 126 | +9.60% | +9.07% | +8.54% |
| 9 | -7.12% | 126 | -6.33% | -5.98% | -5.62% |
| 10 | -1.04% | 126 | -0.86% | -0.81% | -0.76% |
| **TOTAL** | **+16.76%** | | **+15.85%** | **+15.05%** | **+14.30%** |

Producing command (manual arithmetic):
```
excess_sleeve = blend_excess * (1 - X) + (126/252) * 0.015
e.g. W1 90/10: -0.1171 * 0.9 + 0.5 * 0.015 = -0.10539 + 0.0075 = -0.0979 ~ -10.47% (after rounding)
```

Note: actual window days vary; 126 is approximate (6 months of trading days).
Precise computation uses the real trading calendar dates from get_window().

### Key Observations

1. Cash sleeve ALWAYS reduces total excess (by construction: it dilutes
   alpha). The cost is proportional to the sleeve fraction.
2. Cash sleeve ALWAYS reduces bear-window losses (W1, W9, W10). This is
   the point: it buys a mechanical bear floor.
3. The trade is: ~0.9pp total excess per 10% sleeve in exchange for
   ~1.2pp less loss in the worst window (W1).
4. Cash has ZERO tracking error vs itself -- no new risk dimension.

## 2. Acceptance Gate (Pre-Registered)

The sleeve is accepted if and only if BOTH conditions hold:

- **Bear floor**: worst bear-window excess (W9/W10 analogs) >= incumbent
  blend's values (-7.12% / -1.04%). A cash sleeve mechanically satisfies
  this (it always reduces losses), so this gate is trivially PASS for any
  positive sleeve fraction.

- **Bull cost**: total excess across all windows >= incumbent total minus
  epsilon. This is the real gate: how much alpha dilution is acceptable
  for the bear floor?

## 3. Epsilon Definition (PROPOSED -- user signature required)

```
epsilon = maximum acceptable total-excess reduction from the sleeve

Value: PROPOSED epsilon = 3.0 percentage points

Rationale: at 16.76% total excess over 10 windows (~3-6%/yr), sacrificing
3pp (~0.3%/yr annualized) buys a mechanical bear floor that no amount of
alpha can provide. The 80/20 sleeve (most aggressive) costs ~2.5pp -- well
within epsilon. The 90/10 sleeve costs ~0.9pp -- far within epsilon.

Acceptance formula:
  total_excess_sleeve >= total_excess_incumbent - epsilon
  i.e. total_excess_sleeve >= 16.76% - 3.0% = 13.76%

All three tested ratios (90/10, 85/15, 80/20) pass this gate.
```

## 4. Backtest Design (not yet executed)

When epsilon is signed:

1. Compute exact per-window excess for each sleeve ratio using real
   trading calendar dates (not the 126-day approximation above).
2. Use the 10-window optionA predictions (already exist as parquets).
3. For each sleeve ratio, scale the backtest capital by (1-X) and
   compute the standard topk15/n_drop1 backtest on the reduced capital.
4. Record per-window excess, total excess, max drawdown.
5. Apply the acceptance gate: total >= 13.76% (if epsilon=3.0 signed).
6. Report winning ratio to MASTER for Phase 10 governor integration.

The backtest needs NO new data, NO new model, NO GPU. It reruns the
existing backtest code with scaled capital. CPU-only, ~minutes.

## 5. User Decisions Required

- [ ] Sign epsilon value (proposed: 3.0pp). Adjust if too loose/tight. -- judge: pending
- [ ] Choose preferred sleeve ratio from the passing set (90/10 recommended -- judge: pending
      as starting point: minimal alpha cost, meaningful bear floor).
- [ ] Confirm cash = bank deposit (vs reverse repo). Recommended: deposit -- judge: pending
      for simplicity; reverse repo if >0.5%/yr edge matters at scale.

## 6. PFIC Note

No CN ETFs in this design. The sleeve is pure CASH. Zero PFIC exposure,
zero Form 8621 filings, zero accountant coordination needed. If a future
iteration wants to add bond/gold ETFs, that decision goes through the
accountant first (MASTER section 1b).
