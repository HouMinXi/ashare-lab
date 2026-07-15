# R9 Solution: Regime Inversion Mitigation for ashare-lab

## Problem Statement

- 3-year rolling training window with locked 60/40 TRA/nTRA blend weight
- No PSI monitoring on features or factor ICs
- If value factor (or any Alpha158 feature relationship) inverts after 3 years, model learns wrong relationship
- A-shares have more frequent regime transitions than developed markets (Wang et al. 2026: 3 bubble + 2 downturn + 2 random-walk periods in past decade)
- Current TRA has implicit regime routing (3 states in routing head), but blend weight is hardcoded

## What Already Exists

The ashare-lab baseline already has partial regime awareness:

- **TRA model** (`configs/baseline.yaml`): `routing.num_states: 3` with LR_TPE routing signal. The TRA backbone implicitly learns regime-dependent predictions through its routing head.
- **Blend module** (`ashare_lab/research/blend.py`): `tra_weight=0.60` is the Phase 2.1 lock. Parameter exists for unit tests only, never exposed to config.
- **Walk-forward**: 3-year rolling window, 6-month step, min 5 windows.

What is MISSING: explicit regime detection, PSI monitoring, adaptive blend weights, factor IC tracking.

---

## Layer 1: PSI Monitoring (3 days, do first)

### Algorithm

Population Stability Index (PSI) on feature distributions, computed monthly against training window baseline.

```
PSI = sum( (P_i - Q_i) * ln(P_i / Q_i) )  for i in bins
```

Standard thresholds (credit risk origin, confirmed by darwintIQ 2026):
- PSI < 0.10: stable (green)
- 0.10 <= PSI < 0.25: drift warning (yellow)
- PSI >= 0.25: significant shift, retrain recommended (red)

### What to Monitor

1. **Feature-level PSI**: Each of 158 Alpha158 features, binned into 10 deciles
2. **Factor IC drift**: Rolling 6-month IC of TRA predictions vs realized returns
3. **Blend component IC ratio**: IC(TRA) / IC(nTRA) -- if ratio flips below 1.0, the 60/40 blend is wrong

### Implementation

```python
# New file: ashare_lab/monitoring/psi.py
def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI between two distributions."""
    # Use baseline quantiles as bin edges
    edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    base_counts = np.histogram(baseline, bins=edges)[0] / len(baseline)
    curr_counts = np.histogram(current, bins=edges)[0] / len(current)
    # Avoid log(0)
    base_counts = np.clip(base_counts, 1e-6, None)
    curr_counts = np.clip(curr_counts, 1e-6, None)
    return float(np.sum((curr_counts - base_counts) * np.log(curr_counts / base_counts)))

def monitor_features(train_features: pd.DataFrame, live_features: pd.DataFrame) -> dict:
    """PSI for each feature column. Returns {col_name: psi_value}."""
    return {col: compute_psi(train_features[col].values, live_features[col].values)
            for col in train_features.columns}

def compute_rolling_ic(predictions: pd.Series, returns: pd.Series, window: int = 126) -> pd.Series:
    """Rolling rank IC (Spearman correlation) over window trading days."""
    return predictions.groupby(level=1).rolling(window).corr(returns).droplevel(0)
```

### Alert Rules

- Any single feature PSI >= 0.25: log WARNING, include in monthly report
- Mean feature PSI >= 0.15: log CRITICAL, recommend retrain
- Rolling IC < 0.02 for 3 consecutive months: log CRITICAL, factor relationship may have inverted

### Effort: 3 days
- Day 1: PSI computation module + unit tests
- Day 2: Integration into blend_verdict / report pipeline
- Day 3: Alert thresholds + monthly report additions

---

## Layer 2: Explicit Regime Detection (10 days)

### Algorithm: Three-Detector Voting

Proven approach from eastmoney-monthly (381 stars, validated on real A-share data with 24-timepoint unbiased pool). Three independent detectors, majority vote, no single point of failure.

**Detector 1: HMM (2-state Gaussian)**
- Features: [market_volatility, market_turnover, advance_decline_ratio]
- Implementation: `hmmlearn.hmm.GaussianHMM(n_components=2)`
- Walk-forward: retrain every 6 months on rolling 3-year window
- Output: state label (0 or 1) + posterior probability

**Detector 2: EWMA Volatility Ratio**
- Compute: `vol_60d / vol_252d` on CSI1000 (universe benchmark)
- Threshold: ratio > 1.5 = high-vol regime (bear), ratio < 0.8 = low-vol regime (bull)
- Simple, no training needed, interpretable

**Detector 3: Trend Strength**
- Compute: `(price - MA200) / MA200` + slope of MA200 over past 20 days
- Positive + rising slope = bull, negative + falling slope = bear
- Also no training needed

**Voting**: 2-of-3 agreement required. If all three disagree, mark as "uncertain" and default to conservative (bear) allocation.

### Regime Labels

| Regime | HMM | Vol Ratio | Trend | Meaning |
|--------|-----|-----------|-------|---------|
| strong_bear | 1 (high-vol) | > 1.5 | negative + falling | High conviction bear |
| bear | 1 | > 1.5 | mixed | Default bear |
| bull | 0 (low-vol) | < 0.8 | positive + rising | Default bull |
| strong_bull | 0 | < 0.8 | positive + rising | High conviction bull |
| uncertain | mixed | mixed | mixed | Conservative default |

### Implementation

```python
# New file: ashare_lab/regime/detector.py
class RegimeDetector:
    def __init__(self, hmm_retrain_months: int = 6):
        self.hmm_model = None
        self.last_hmm_train = None
        self.hmm_retrain_months = hmm_retrain_months

    def detect(self, market_data: pd.DataFrame) -> tuple[str, float]:
        """Returns (regime_label, confidence)."""
        hmm_vote = self._hmm_detect(market_data)
        vol_vote = self._volatility_detect(market_data)
        trend_vote = self._trend_detect(market_data)

        votes = [hmm_vote, vol_vote, trend_vote]
        bull_votes = sum(1 for v in votes if v == "bull")
        bear_votes = sum(1 for v in votes if v == "bear")

        if bull_votes >= 2:
            return "bull", bull_votes / 3.0
        elif bear_votes >= 2:
            return "bear", bear_votes / 3.0
        else:
            return "uncertain", 0.33

    def _hmm_detect(self, data: pd.DataFrame) -> str:
        # Retrain if stale
        # Fit 2-state HMM on [vol, turnover, adv_ratio]
        # Return "bull" if low-vol state, "bear" if high-vol state
        ...

    def _volatility_detect(self, data: pd.DataFrame) -> str:
        vol_60 = data['close'].pct_change().tail(60).std() * np.sqrt(252)
        vol_252 = data['close'].pct_change().tail(252).std() * np.sqrt(252)
        ratio = vol_60 / vol_252
        if ratio > 1.5:
            return "bear"
        elif ratio < 0.8:
            return "bull"
        return "uncertain"

    def _trend_detect(self, data: pd.DataFrame) -> str:
        ma200 = data['close'].rolling(200).mean()
        position = (data['close'].iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]
        slope = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20]
        if position > 0.02 and slope > 0:
            return "bull"
        elif position < -0.02 and slope < 0:
            return "bear"
        return "uncertain"
```

### Effort: 10 days
- Day 1-3: HMM detector with walk-forward retraining
- Day 4-5: Volatility and trend detectors
- Day 6-7: Voting logic + unit tests
- Day 8-9: Integration with walk-forward backtest pipeline
- Day 10: Validation on historical regime periods (2015 crash, 2018 bear, 2020 bull, 2021-2023 downturn)

---

## Layer 3: Adaptive Blend Weight (5 days)

### Option A: Regime-Conditioned Lookup Table (Recommended for Phase 4)

Replace the locked 60/40 with a regime-dependent table:

| Regime | tra_weight | rule_weight | Rationale |
|--------|-----------|-------------|-----------|
| strong_bull | 0.70 | 0.30 | ML learns momentum in bull markets |
| bull | 0.65 | 0.35 | Slight ML tilt |
| uncertain | 0.50 | 0.50 | Equal weight, no conviction |
| bear | 0.45 | 0.55 | Rule-based more robust in bear |
| strong_bear | 0.35 | 0.65 | Defensive, trust rules more |

Interpolation: when regime confidence is between 0.33 and 0.67, linearly interpolate between adjacent regime weights.

```python
# Modification to ashare_lab/research/blend.py
REGIME_WEIGHTS = {
    "strong_bull": 0.70,
    "bull": 0.65,
    "uncertain": 0.50,
    "bear": 0.45,
    "strong_bear": 0.35,
}

def blend_tra_ntra(
    pred: pd.Series,
    window: dict,
    *,
    tra_weight: float | None = None,  # None = use regime
    regime: str = "uncertain",
    regime_confidence: float = 0.5,
) -> pd.Series:
    if tra_weight is None:
        tra_weight = REGIME_WEIGHTS.get(regime, 0.50)
        # Confidence-based interpolation toward 0.50
        tra_weight = 0.50 + (tra_weight - 0.50) * min(regime_confidence / 0.67, 1.0)
    # ... rest of existing logic
```

### Option B: Rolling IC-Weighted (Enhancement, after Phase 4)

More adaptive, less interpretable:

```python
def compute_adaptive_weight(
    ml_ic_6m: float,
    rule_ic_6m: float,
    clip_low: float = 0.20,
    clip_high: float = 0.80,
) -> float:
    """IC-weighted blend: w_ml = IC_ml / (IC_ml + IC_rule)."""
    if ml_ic_6m <= 0 and rule_ic_6m <= 0:
        return 0.50  # both bad, equal weight
    w = ml_ic_6m / (ml_ic_6m + rule_ic_6m)
    return np.clip(w, clip_low, clip_high)
```

### Effort: 5 days
- Day 1-2: Modify blend.py to accept regime parameter + lookup table
- Day 3: Wire regime detector output into blend call chain
- Day 4: Unit tests + backtest with regime-adaptive blend
- Day 5: Compare fixed 60/40 vs adaptive on 2018-2024 walk-forward

---

## Layer 4: Meta-Learning (Deferred, 2-3 weeks)

### Wang & Lera 2026 (FinPFN) -- Most Directly Applicable

**Paper**: "Meta-learning for return prediction in shifting market regimes"
**Journal**: Journal of Financial Markets, vol. 79, 2026
**Authors**: Yicheng Wang, Sandro Claudio Lera (SUSTech)
**Tested on**: Chinese A-shares (daily) + US equities (monthly)

**Core idea**: Instead of learning a fixed mapping from features to returns, condition forecasts on recent feature-return relationships. Uses a Transformer-based Bayesian predictor (FinPFN) that meta-learns by treating the previous day as a prior for predicting the current day.

**Key result**: Significantly outperforms random forests during regime changes (proxied by large volatility shifts).

**Feasibility for ashare-lab**:
- PRO: Tested on A-shares directly, handles regime shifts explicitly
- PRO: No explicit regime labels needed (learns from recent context)
- CON: Requires Transformer training infrastructure (GPU)
- CON: Paper is 19 pages, implementation complexity is moderate
- CON: Daily-frequency design; ashare-lab runs at daily but walk-forward is 6-month step

**Verdict**: Promising but requires GPU infrastructure. Defer to after Phase 4 live validation.

### RAM-Net (KAIST 2025) -- Lighter Alternative

**Paper**: "Adaptive Sample Weighting with Regime-Aware Meta-Learning Framework"
**Core idea**: Wraps existing model as black box. Meta-learns sample weights based on regime. Preprocessing step, not architecture change.
**Result**: +10% improvement across 6 baseline models.
**Feasibility**: Can be added as preprocessing to existing TRA training. No architecture change needed.

### ReCAP (KDD 2026) -- Overkill for Now

Full continual learning framework with policy library + regime-gate. Designed for production portfolio management. Overkill for paper-trading system.

### Effort: 2-3 weeks (deferred)
- Week 1: Implement RAM-Net sample weighting wrapper
- Week 2: Integrate with walk-forward pipeline + backtest
- Week 3: If promising, explore FinPFN for direct replacement

---

## Implementation Priority

```
Phase 4 (immediate):
  [1] PSI Monitor          -- 3 days  -- catches drift before it hurts
  [2] Regime Detection     -- 10 days -- explicit regime labels
  [3] Adaptive Blend       -- 5 days  -- regime-conditioned weights

Phase 5 (after live validation):
  [4] Meta-learning (RAM-Net) -- 2 weeks -- if regime detection shows value
  [5] FinPFN integration      -- 3 weeks -- if GPU infra available
```

Total Phase 4 effort: ~18 working days (3.5 weeks part-time).

---

## Key References

1. **Wang, Feng (2026)** - "Comparing factor models across different market regimes: Evidence from China" - Pacific-Basin Finance Journal. mBSDF bubble detection, DHS3/HXZ4 regime-conditional performance.

2. **Liu, Lee (2025)** - "Capturing the Risk Dynamics of the A-Share Market Based on Markov Regime-Switching" - Int J Financial Econ. MSGARCH + quantile regression for A-share regime identification.

3. **Wang, Lera (2026)** - "Meta-learning for return prediction in shifting market regimes" - J Financial Markets. FinPFN on A-shares, Transformer Bayesian predictor.

4. **Pan et al. (2026)** - "ReCAP: Regime-aware Continual Adaptive Portfolio management" - KDD 2026. Policy library + regime-gate for continual learning.

5. **Jang et al. (2025)** - "RAM-Net: Adaptive Sample Weighting with Regime-Aware Meta-Learning" - ACM. +10% improvement via regime-aware sample re-weighting.

6. **eastmoney-monthly (2026)** - github.com/yuu-ramsey/eastmoney-monthly. Three-detector voting (HMM/vol/trend) proven on A-shares, 381 stars.

7. **Ma (2026)** - "Breaks and Trends in Factor Premia" - Lancaster FoFI. TV regularization for structural break detection in factor premia, adaptive SDF construction.

8. **AlphaCrafter (2026)** - Multi-agent framework with regime-conditioned factor ensembles on CSI 300 + S&P 500.
