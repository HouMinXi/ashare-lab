# ML Model Retraining Frequency in Quantitative Trading: Research Synthesis

Generated: 2026-07-15 | Sources: 4 Exa searches, 40+ results analyzed

---

## 1. Academic/Industry Consensus on Retraining Frequency

### 1.1 Strategy-Horizon-Dependent Frequencies

The consensus is clear: retraining frequency must match the strategy's time horizon.
There is no universal "best" frequency -- it depends on how fast the market
patterns the model learned become obsolete.

| Strategy Horizon | Typical Retraining Frequency | Rationale |
|---|---|---|
| Intraday / HFT | Daily to weekly | Fast-changing microstructure patterns |
| Short-term (days) | Weekly to monthly | Balance adaptation with stability |
| Medium-term (weeks) | Monthly to quarterly | Longer-lasting market regimes |
| Long-term (months) | Quarterly to annually | Fundamental relationships more stable |

Source: Breaking Alpha Insights (breakingalpha.io)

### 1.2 Two Dominant Paradigms

**A. Scheduled (Rolling Window) Retraining**
- Retrain on a fixed calendar schedule (e.g., every month, every quarter)
- Walk-forward validation is the standard methodology (Lopez de Prado, 2018)
- Common window: 3-year training data, monthly retrain, 6-month validation
- Simple, predictable, but may lag behind sudden regime changes

**B. Triggered / Event-Driven Retraining**
- Retrain only when drift detection signals degradation
- Combines performance monitoring (IC, Sharpe) with distributional tests (PSI, KS)
- More responsive but higher operational complexity

**C. Hybrid (Recommended in Production)**
- Scheduled baseline (e.g., monthly) PLUS triggered override
- Production systems overwhelmingly favor this approach
- Source: Wayland Zhang "AI Quantitative Trading" textbook; ML4Trading (Stefan Jansen)

### 1.3 Key Academic References

- **Geertsema & Lu (2022), SSRN 3965171**: Information in financial markets
  decays at ~7% per month (exponential decay). A 1-month delay in using
  information = 7% decrease in expected returns. Does not vary substantially
  across stock size/liquidity. For "stable" stocks, first-month decay is ~60%
  but near zero afterward.
- **Regol et al. (2025), PMLR v267**: Principled formulation of the retraining
  problem as an uncertainty-based decision under limited information. Proposes
  forecasting model performance evolution rather than threshold-based triggers.
- **Pan et al. (2026), arXiv 2606.00143 (ReCAP)**: Regime-adaptive continual
  learning outperforms rolling-window retraining. Uses a policy library across
  detected regimes rather than periodic full retraining. On NAS100: ReCAP
  achieves 164.89% CR / 1.14 SR vs. best rolling-window baseline at 124.24% /
  0.92 SR (90-day retraining frequency, 360-day window).
- **AlphaCon (OpenReview, 2025)**: In-context adaptation at inference time
  without retraining. Trained once, adapts to regimes via context encoding.
  "Significantly outperforms strong baselines that require periodic retraining."

---

## 2. IC Decay Rates: How Quickly Do Models Degrade?

### 2.1 Quantified IC Decay

**A-shares Specific (from Wayland Zhang's textbook, Lesson 17):**
- Example: IC dropped from 0.05 to 0.01 over 24 months
- Monthly decay rate: ~6.6% per month (calculated as compound decay)
- Formula: IC(t) = IC(0) * (1-r)^t, where r = 0.066/month

**Information Decay (Geertsema & Lu, 2022):**
- Stock return predictability follows asymptotic exponential decay
- ~7% per month overall
- For stable stocks: 60% decay in first month, near zero after

**General Decay Types (from multiple sources):**

| Decay Type | Manifestation | Cause | Typical Cycle |
|---|---|---|---|
| Sudden | Fails one day | Policy change, black swan | Unpredictable |
| Gradual | Returns slowly decline | Alpha arbitraged, structure changes | 6-18 months |
| Cyclical | Good sometimes, bad others | Market state switching | Economic cycle |

Source: Wayland Zhang "AI Quantitative Trading", Lesson 17

### 2.2 IC Quality Benchmarks (Production Systems)

From MarketSentinel (production XGBoost system for S&P 500):

| IC Range | Quality Assessment |
|---|---|
| > 0.08 | Strong alpha |
| 0.04 - 0.08 | Moderate alpha |
| 0.02 - 0.04 | Weak alpha |
| < 0.02 | Near noise -- retrain |

Source: github.com/muhammedshihab1001/MarketSentinel

### 2.3 Model Decay for Neural Networks (LSTM/Transformer)

- LSTM/Transformer models are especially vulnerable to concept drift because
  they learn complex nonlinear relationships that can invert during regime
  changes (Mental Momentum Research, 2026)
- The nonstationarity-complexity tradeoff: complex models need larger training
  windows but larger windows increase exposure to regime shifts
- Modern mitigation: frequency-domain decomposition (VMD, DERITS) before
  feeding into LSTMs; regime-switching architectures; continual learning

---

## 3. Production System Thresholds

### 3.1 Drift Detection Thresholds

**IC Thresholds (Wayland Zhang / ML4Trading):**

| Metric | Warning Threshold | Critical Threshold | Action |
|---|---|---|---|
| IC | < 0.02 | < 0.01 | Trigger retraining |
| PSI | > 0.10 | > 0.25 | Distribution shift |
| Rolling Sharpe (20d) | < 0.5 | < 0.0 | Performance decay |
| Rolling Win Rate | < 45% | < 40% | Check signal quality |

**Retraining Trigger Rules (Production Example):**
1. IC < 0.01 for 3 consecutive days --> Trigger retraining
2. PSI > 0.25 single occurrence --> Trigger immediately
3. 20-day Sharpe < 0 --> Trigger retraining

Source: Wayland Zhang "Model Drift and Retraining Strategies"

**MarketSentinel Drift Severity Scale:**

| Severity | State | Exposure Scale | Action |
|---|---|---|---|
| 0-4 | None/Low | 1.0 (full) | Monitor only |
| 5-7 | Soft Drift | 0.6 | Consider retrain |
| 8-9 | Soft+Flag | 0.6 | retrain_required = true |
| 10-14 | Hard Drift | 0.25 | Immediate retrain |
| 15 | Critical | 0.25 | Consider halting |

Source: MarketSentinel production system

**Credit Risk (ACRM, ACL 2026):**
- KS decay > 0.03 triggers automated model refresh
- PSI boundary is a hard constraint (non-negotiable)
- Shifted from quarterly to monthly maintenance cadence

### 3.2 Drift Detection Methods

| Method | What It Detects | Sensitivity | Cost | Use Case |
|---|---|---|---|---|
| Performance Monitoring | Strategy returns | Medium | Low | All strategies (essential) |
| K-S Test | Feature distribution | High | Medium | Periodic checks |
| Chi-Square Test | Categorical features | High | Low | Regime labels |
| CUSUM | Prediction errors | High | Low | Continuous monitoring |
| PSI | Population stability | High | Low | Production (standard) |
| KL Divergence | Distribution divergence | High | Medium | Autonomous systems |
| SHAP Monitoring | Feature importance shift | High | High | Advanced diagnostics |
| ADWIN / DDM | Concept drift in errors | High | Medium | Online learning |

**Four-Quadrant Diagnostic (ML4Trading / Jansen):**

| | Performance Decay: YES | Performance Decay: NO |
|---|---|---|
| **Drift Detected: YES** | Retrain on recent data | Model is robust (monitor) |
| **Drift Detected: NO** | Monitoring coverage incomplete | All clear |

### 3.3 Model Update Workflow (Best Practice)

1. Shadow mode: challenger processes live data without trading
2. Champion-challenger evaluation with explicit promotion criteria
3. Minimum effect size: 0.2-0.3 Sharpe improvement required
4. Deflated Sharpe ratio or bootstrap comparison for statistical rigor
5. White's Reality Check when multiple challengers compete
6. Explicit rollback procedures tested before deployment

Source: ML4Trading (Stefan Jansen), MLOps and Governance chapter

---

## 4. A-Shares Specific Considerations

### 4.1 Higher Volatility and Regime Frequency

- A-share market exhibits "relatively higher volatility and lower VaR/ES than
  most alternative markets" (Liu & Lee, 2025, IJFE 70108)
- Markov regime-switching GARCH model shows frequent regime transitions
- RCM statistic: 3.60 (Shanghai), 21.20 (Shenzhen) -- both well below 50
  threshold, indicating efficient regime classification
- Global factors (oil price, gold, VIX, US 10Y yield) drive A-share regime
  changes -- exogenous shocks are common

### 4.2 Herding Behavior Amplifies Regime Shifts

- Herding is prominent in volatile A-share regimes; adverse herding in tranquil
  regimes (regime-switching model, all listed A-shares 1999-2016)
- Sector-level herding shows rotational patterns during turbulent regimes
- This means factor relationships can invert more abruptly than in developed
  markets, requiring faster model adaptation

Source: ScienceDirect, "Regime-switching herd behavior in Chinese A-share market" (2021)

### 4.3 Meta-Learning Outperforms Periodic Retraining for A-Shares

- Wang & Lera (2026, Journal of Financial Markets): Meta-learning framework
  (FinPFN) evaluated on daily Chinese A-shares
- Conditions forecasts on recent feature-return relationships
- "Significantly outperforms benchmarks during regime changes proxied by
  large volatility shifts"
- Does NOT require explicit regime labels or frequent re-estimation

### 4.4 Recommended A-Shares Retraining Parameters

Based on synthesized evidence:

| Strategy Type | Recommended Frequency | Training Window | Notes |
|---|---|---|---|
| Daily factor model | Weekly to biweekly | 1-2 years rolling | A-shares factor decay faster |
| Medium-frequency | Monthly | 2-3 years | Include regime detection |
| Low-frequency/Value | Quarterly | 3-5 years | Fundamental factors more stable |

**Additional A-Shares Specific Triggers:**
- Policy change events (regulatory announcements) --> immediate evaluation
- VIX spike or US rate moves --> check regime state
- Retail investor participation surge --> momentum factor may shift
- Industry rotation speed change --> regime transition likely

### 4.5 A-Shares IC Decay Expectations

Based on the Wayland Zhang textbook example and adjusted for A-share volatility:
- Typical IC decay rate: 5-7% per month (consistent with global 7% from
  Geertsema & Lu, but potentially faster during policy-driven regime shifts)
- Gradual decay cycle: 6-18 months before model becomes ineffective
- Sudden decay: can happen overnight on policy changes (unpredictable)
- IC effectiveness threshold for A-shares: 0.03 (below this, signal is weak)

---

## 5. Summary: Key Takeaways for ashare-lab

1. **Monthly retraining is a reasonable default** for medium-frequency A-share
   factor models, but should be supplemented with triggered retraining.

2. **IC decays at ~6-7% per month** on average. After 6 months without
   retraining, expect ~35-40% of original predictive power lost.

3. **Production thresholds**: IC < 0.02 = warning, IC < 0.01 = critical,
   PSI > 0.25 = immediate retrain, 20d Sharpe < 0 = retrain.

4. **A-shares need faster adaptation** than US markets due to: higher regime
   frequency, policy-driven shocks, retail herding, and factor crowding from
   quant proliferation.

5. **Continual learning / online learning** is superior to periodic batch
   retraining for A-shares (ReCAP, AlphaCon, meta-learning evidence).

6. **Hybrid strategy is the production standard**: scheduled monthly baseline
   PLUS drift-triggered override. Monitor IC, PSI, Sharpe simultaneously with
   multi-window (5d/20d/60d) tracking.

7. **Training window should be longer than monitoring window** (SAS simulation
   finding: 18-month train / 6-month monitor was optimal in 25-year study).

---

## Sources

1. Geertsema & Lu (2022). "Measuring information decay in financial markets."
   SSRN 3965171.
2. Breaking Alpha Insights. "Machine Learning in Strategy Development."
   breakingalpha.io.
3. Garg (2026). "ML Techniques for Quantitative Stock Trading Strategies."
   IJFMR v08i01.
4. Mental Momentum Research (2026). "Non-Stationarity and Concept Drift in
   Quantitative Trading."
5. Pan et al. (2026). "ReCAP: Regime-Adaptive Continual Learning for Portfolio
   Management." arXiv 2606.00143.
6. Regol et al. (2025). "When to retrain a machine learning model." PMLR v267.
7. Wayland Zhang. "AI Quantitative Trading: From Zero to One." waylandz.com.
   Lessons 09, 17, and Model Drift chapter.
8. Lucena Research (2019). "Dynamically Retraining Models for Stock
   Forecasting."
9. MarketSentinel. github.com/muhammedshihab1001/MarketSentinel.
10. Jansen, Stefan. "ML for Trading" (3rd ed). ml4trading.io. MLOps chapter.
11. ACRM (2026). "Multi-Agent Trajectory Learning for Automated Credit Risk
    Model Refreshing." ACL Industry 65.
12. Liu & Lee (2025). "Capturing Risk Dynamics of A-Share Market Based on
    Markov Regime-Switching." IJFE 70108.
13. Wang & Lera (2026). "Meta-learning for return prediction in shifting market
    regimes." Journal of Financial Markets v79.
14. "Regime-switching herd behavior in Chinese A-share market." ScienceDirect
    (2021).
15. Shen et al. (2026). "Regime-Dependent Industry Rotation Strategy in
    China's A-Share Market." Atlantis Press.
16. AlphaCon (2025). "In-Context Adaptation for Dynamic Alpha Generation."
    OpenReview.
17. SAS (2020). "Turning the Crank: A Simulation of Optimizing Model
    Retraining." Proceedings 4612-2020.
18. Duling, David R. SAS Global Forum 2020.
