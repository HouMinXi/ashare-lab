# R4b: Neutralization Quality During Crisis -- Solution Design

## Problem Statement

`neutralize.py` uses `np.linalg.lstsq` (pure OLS) for cross-sectional factor
neutralization. During crisis days:

- Residuals are correlated and heavy-tailed (volatility clustering)
- Survivor bias distorts cross-sectional distributions
- `_MIN_INSTRUMENTS=30` is below the recommended minimum even in normal times
- No quality check on neutralization output -- bad residuals silently propagate

## 1. Robust Regression Replacement: Huber M-Estimator

### Recommendation: Huber T via statsmodels.RLM

**Why Huber over WLS/RANSAC:**

- Huber M-estimator (IRLS algorithm) is the standard robust method for
  cross-sectional factor models in quantitative finance (Martin 2022,
  "Robust Regression Estimator for Asset Factor Models").
- Outlier down-weighting is automatic: observations with |residual| > t
  (default t=1.345*MAD) get weight t/|z|; observations within threshold
  get weight 1.0. This handles the heavy-tailed, correlated residuals
  during crisis without discarding data.
- WLS requires knowing or estimating the variance structure -- useful but
  secondary. Romano & Wolf (2017) show ALS (Adaptive LS, which switches
  OLS/WLS via Breusch-Pagan pre-test) reduces HC3 standard errors vs OLS
  in 79-94% of cases, especially during financial turmoil.
- RANSAC is designed for structural outliers (wrong model), not
  distributional outliers (heavy tails). It discards too aggressively
  during crisis when most observations are "outliers" by OLS standards.
- The 2025 ScienceDirect paper on Huber PCA for large-dimensional factor
  models confirms Huber loss is the right tradeoff: robust to heavy tails,
  statistically efficient under normality, computationally cheap (IRLS
  converges in ~10 iterations).

### Implementation Design

```python
import statsmodels.api as sm

# Current: OLS via numpy
beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)

# Proposed: Huber M-estimator via statsmodels
model = sm.RLM(y, x_mat, M=sm.robust.norms.HuberT(t=1.345))
result = model.fit()
beta = result.params
residuals = result.resid
weights = result.weights  # diagnostic: see which obs were downweighted
```

**Key parameters:**
- `t=1.345` (default): 95% asymptotic efficiency at the normal distribution.
  Higher t = less robust, more efficient. Lower t = more robust, less efficient.
- `scale_est=sm.robust.scale.HuberScale()`: robust scale estimator (default
  MAD works well for cross-sectional data).
- `maxiter=50`: IRLS convergence limit (typically converges in 5-10 iterations).

**Fallback strategy:** If Huber fails to converge (rare, but possible with
pathological data), fall back to OLS and log a warning. This keeps the
function safe for production.

### Data Structure Compatibility

The current code works with numpy arrays extracted from pandas. statsmodels
RLM accepts the same numpy arrays -- no structural change needed. The only
addition is importing statsmodels (already a dependency of the project).

## 2. Crisis Detection Trigger for Regression Switch

### Recommendation: Two-Pronged Signal

Leverage the existing `regime.py` infrastructure (ma20_above_pct) plus a
cross-sectional residual quality check.

**Trigger 1: Regime Signal (from existing regime.py)**
- `ma20_above_pct < 0.30` indicates crisis breadth -- use Huber unconditionally
- `ma20_above_pct >= 0.30` indicates normal market -- Huber still default,
  but OLS acceptable as fallback

**Trigger 2: Cross-Sectional Heteroscedasticity Test**
- After running regression, compute Breusch-Pagan test on residuals
- If p-value < 0.05, log a warning that heteroscedasticity is present
- This is informational (for quality metrics), not a method switch --
  Huber handles heteroscedasticity by design

**Design decision:** Do NOT switch between OLS and Huber based on regime.
Use Huber as the universal default. The rationale:
- Huber is 95% efficient at the normal -- negligible cost in normal times
- Regime detection introduces look-ahead risk and complexity
- The cost of one bad neutralization day during crisis exceeds the
  efficiency loss of Huber on 1000 normal days

### Integration with regime.py

```python
def neutralize_predictions(
    pred: pd.Series,
    factors: pd.DataFrame,
    regime_date: str | None = None,  # optional: pass date for regime check
) -> pd.Series:
    # ... existing alignment logic ...
    # Huber regression always used (no regime switch needed)
    # But log regime context for diagnostics
    if regime_date:
        from ashare_lab.research.regime import compute_regime_signals
        signals = compute_regime_signals(regime_date)
        if signals["ma20_above_pct"] < 0.30:
            log.info("neutralize: %s crisis regime (breadth=%.3f), "
                     "using robust regression", regime_date,
                     signals["ma20_above_pct"])
```

## 3. Neutralization Quality Metrics

### Recommended Metrics (computed per date, logged, not blocking)

| Metric | What It Measures | Crisis Threshold | Implementation |
|--------|-----------------|------------------|----------------|
| **Cross-sectional R-squared** | How much variance do factors explain? | R2 < 0.10 = factors explain nothing; R2 > 0.90 = near-perfect fit (data issue) | `1 - SS_res / SS_tot` |
| **Huber weight statistics** | Fraction of observations downweighted | > 30% downweighted = distribution is very non-normal | `np.mean(result.weights < 0.95)` |
| **Residual skewness** | Asymmetry of residuals | \|skew\| > 2.0 = severe asymmetry | `scipy.stats.skew(residuals)` |
| **Residual kurtosis** | Tail heaviness | kurtosis > 10 = extreme tails | `scipy.stats.kurtosis(residuals)` |
| **Jarque-Bera test** | Joint normality test | p < 0.01 = reject normality | `scipy.stats.jarque_bera(residuals)` |
| **Breusch-Pagan test** | Heteroscedasticity | p < 0.05 = heteroscedastic | Manual implementation (regress |residuals| on X) |
| **Factor coefficient stability** | Are betas stable? | beta change > 50% from rolling median | Rolling window comparison |

### Implementation Pattern

```python
from dataclasses import dataclass
from scipy import stats

@dataclass
class NeutralizationQuality:
    """Quality metrics for a single date's neutralization."""
    date: str
    r_squared: float
    n_instruments: int
    n_downweighted: int  # Huber weights < 0.95
    residual_skewness: float
    residual_kurtosis: float
    jarque_bera_pvalue: float
    breusch_pagan_pvalue: float
    max_abs_residual: float
    method_used: str  # "huber" or "ols_fallback"

    @property
    def is_crisis_quality(self) -> bool:
        """True if quality metrics suggest crisis-day problems."""
        return (
            self.n_downweighted / self.n_instruments > 0.30
            or self.jarque_bera_pvalue < 0.01
            or abs(self.residual_skewness) > 2.0
        )


def compute_quality_metrics(
    y: np.ndarray,
    x_mat: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    date: str,
    method: str,
) -> NeutralizationQuality:
    """Compute quality metrics for neutralization output."""
    n = len(y)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_downweighted = int(np.sum(weights < 0.95))
    skew = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))  # excess kurtosis
    jb_stat, jb_p = stats.jarque_bera(residuals)

    # Breusch-Pagan: regress |residuals| on X
    abs_resid = np.abs(residuals)
    bp_model = sm.OLS(abs_resid, x_mat).fit()
    bp_stat = n * bp_model.rsquared
    bp_p = 1.0 - stats.chi2.cdf(bp_stat, df=x_mat.shape[1] - 1)

    return NeutralizationQuality(
        date=date,
        r_squared=r_squared,
        n_instruments=n,
        n_downweighted=n_downweighted,
        residual_skewness=skew,
        residual_kurtosis=kurt,
        jarque_bera_pvalue=float(jb_p),
        breusch_pagan_pvalue=float(bp_p),
        max_abs_residual=float(np.max(np.abs(residuals))),
        method_used=method,
    )
```

### Logging Strategy

- **Every date:** Log at DEBUG level (available but not noisy)
- **Crisis-quality dates:** Log at WARNING level
- **Summary stats:** Log at INFO level every 20 trading days (rolling window)
  -- median R2, fraction of crisis-quality days, fraction of Huber fallbacks

### Alerting Integration

```python
# In the main pipeline, after neutralization:
if quality.is_crisis_quality:
    log.warning(
        "neutralize: %s crisis-quality residuals "
        "(R2=%.3f, downweighted=%d/%d, skew=%.2f, kurt=%.2f)",
        quality.date, quality.r_squared,
        quality.n_downweighted, quality.n_instruments,
        quality.residual_skewness, quality.residual_kurtosis,
    )
```

## 4. Minimum Instrument Count During Crisis

### Current State

```python
_MIN_INSTRUMENTS = 30  # hardcoded, no crisis adjustment
```

### Problem

- 30 instruments for 3 regressors + intercept = 7.5:1 ratio. Standard
  econometric rule of thumb is 10:1 minimum, so 40 is the bare minimum.
- Chen, Connor, Korajczyk (2018): "Differences across estimators are most
  pronounced when cross-sectional sample sizes n have fewer than 4,000
  assets." For our CSI1000 universe (~1000 stocks), we are always in the
  small-n regime where estimator choice matters most.
- During crisis: correlated residuals, survivor bias, and volatility
  clustering reduce the effective sample size. 30 real instruments during
  crisis behave like ~15-20 independent observations.

### Recommendation

```python
# Normal times
_MIN_INSTRUMENTS_NORMAL = 50   # 50:4 = 12.5:1 ratio

# Crisis detection: if regime signal available and ma20_above_pct < 0.30
_MIN_INSTRUMENTS_CRISIS = 80   # higher bar: 80:4 = 20:1 ratio

# Absolute floor: never run regression below this
_MIN_INSTRUMENTS_FLOOR = 20    # below this, return predictions unchanged
```

**Why 50/80 instead of 30:**
- 50 gives 12.5:1 ratio for 4 parameters -- comfortable margin above 10:1
- 80 during crisis accounts for effective sample size reduction from
  correlated residuals
- 20 is the absolute floor: below this, the regression is meaningless
  regardless of method

**Dynamic adjustment (alternative):** Instead of fixed thresholds, compute
the effective sample size using the autocorrelation of residuals:

```python
def effective_sample_size(residuals: np.ndarray) -> float:
    """Compute effective n accounting for residual autocorrelation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(residuals, lags=5, return_df=True)
    # If any lag is significant, reduce effective n
    if (lb["lb_pvalue"] < 0.05).any():
        # Simple correction: n_eff = n / (1 + 2*sum(rho_k))
        acf = np.correlate(residuals, residuals, mode="full")
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        # Sum first 5 autocorrelations
        rho_sum = np.sum(np.abs(acf[1:6]))
        n_eff = len(residuals) / (1 + 2 * rho_sum)
        return max(n_eff, 1.0)
    return float(len(residuals))
```

This is more principled but adds complexity. Start with fixed thresholds,
add dynamic adjustment if crisis-day neutralization still produces bad
quality metrics.

## 5. Complete Revised neutralize_predictions Flow

```python
def neutralize_predictions(
    pred: pd.Series,
    factors: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional robust neutralization of predictions against style factors.

    Uses Huber M-estimator (IRLS) instead of OLS for robustness to
    heavy-tailed residuals and correlated outliers during market crises.
    Falls back to OLS if Huber fails to converge.

    Quality metrics are computed per date and logged. Crisis-quality
    dates (high downweighting, non-normal residuals) trigger warnings.

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float.
        factors: DataFrame from compute_style_factors.

    Returns:
        Series with same structure as pred. For dates with fewer than
        _MIN_INSTRUMENTS valid observations, predictions are returned
        unchanged.
    """
    if pred.empty:
        return pred.copy()

    results = []
    quality_log: list[NeutralizationQuality] = []
    dates = pred.index.get_level_values("datetime").unique()

    for date in dates:
        # ... existing alignment and NaN filtering logic (unchanged) ...

        if len(pred_valid) < _MIN_INSTRUMENTS:
            # ... return unchanged ...
            continue

        y = pred_valid.values.astype(np.float64)
        x_mat = np.column_stack([
            np.ones(len(pred_valid)),
            factors_valid["size"].values,
            factors_valid["volatility"].values,
            factors_valid["momentum"].values,
        ]).astype(np.float64)

        # Huber M-estimator (replaces np.linalg.lstsq)
        method = "huber"
        try:
            rlm_model = sm.RLM(y, x_mat, M=sm.robust.norms.HuberT(t=1.345))
            rlm_result = rlm_model.fit(maxiter=50)
            beta = rlm_result.params
            residuals = y - x_mat @ beta
            weights = rlm_result.weights
        except Exception:
            # Fallback to OLS if Huber fails
            log.warning("neutralize: Huber failed for %s, falling back to OLS", date)
            beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            residuals = y - x_mat @ beta
            weights = np.ones(len(y))
            method = "ols_fallback"

        # Compute quality metrics
        quality = compute_quality_metrics(
            y, x_mat, residuals, weights, str(date), method
        )
        quality_log.append(quality)

        if quality.is_crisis_quality:
            log.warning(
                "neutralize: %s crisis-quality (R2=%.3f, downweighted=%d/%d)",
                date, quality.r_squared,
                quality.n_downweighted, quality.n_instruments,
            )

        # ... build result series (unchanged) ...

    # Periodic summary logging
    if quality_log and len(quality_log) % 20 == 0:
        _log_quality_summary(quality_log[-20:])

    return pd.concat(results) if results else pred.copy()
```

## 6. Implementation Effort Estimate

| Task | LOC | Effort | Risk |
|------|-----|--------|------|
| Replace OLS with Huber RLM | ~15 lines changed | 1h | Low -- statsmodels.RLM is well-tested |
| Add quality metrics dataclass | ~40 lines new | 1h | Low -- pure computation |
| Add compute_quality_metrics function | ~30 lines new | 1.5h | Low -- scipy.stats is standard |
| Update _MIN_INSTRUMENTS to 50 | 1 line | 5min | Low |
| Add quality logging to neutralize_predictions | ~20 lines | 1h | Low |
| Add fallback logic (Huber -> OLS) | ~10 lines | 30min | Low |
| Update tests for robust regression | ~50 lines | 2h | Medium -- need to verify residuals differ |
| Test with known crisis dates (2015, 2020) | manual | 1h | Medium -- need real data |
| **Total** | **~165 lines** | **~8h** | **Low-Medium** |

## 7. Dependencies

- `statsmodels` (already in project dependencies -- used by regime.py indirectly)
- `scipy.stats` (already in project dependencies)
- No new external dependencies required

## 8. References

1. Martin, R.D. (2022). "Robust Regression Estimator for Asset Factor Models."
   Journal of Asset Management. (mOpt estimator -- theoretical foundation)
2. Romano & Wolf (2017). "Resurrecting weighted least squares."
   Journal of Econometrics. (ALS/WLS with HC3 std errors)
3. Chen, Connor, Korajczyk (2018). "A Performance Comparison of Large-n
   Factor Estimators." Review of Asset Pricing Studies. (small-n regime)
4. Kan, Robotti, Shanken (2013). "Pricing Model Performance and the
   Two-Pass Cross-Sectional Regression Methodology." JF. (R2 distribution)
5. Huber, P.J. (1964). "Robust Estimation of a Location Parameter."
   Annals of Math Stats. (Huber loss function)
6. statsmodels RLM documentation:
   https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html
7. facmodcs library (github.com/Druhayes/facmodcs):
   Production implementation of OLS/WLS/Robust/W-Rob cross-sectional models.
