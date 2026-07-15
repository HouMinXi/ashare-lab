# R8: Factor Crowding / Alpha Decay - Solution Design

## Problem Statement

Alpha158 features are public and widely replicated by thousands of A-share quant funds. Without crowding detection, alpha decay remains invisible until walk-forward validation catches up weeks or months later. The hyperbolic decay model alpha(t) = K/(1+lambda*t) describes how factor alpha erodes as more agents discover and trade the same signal.

**A-share specific constraints:**
- No 13F filings (no quarterly institutional position disclosure)
- No short interest data (limited short-selling, no comparable metric)
- Retail-dominated market (~80% retail turnover)
- Public fund overweight ratios available quarterly (lagging)
- ETF fund flows available daily (real-time proxy)

---

## 1. Factor Crowding Index (FCI) for A-shares

### 1.1 Design: Adapted 3-Signal Composite

The US FCI model (Quant Decoded, 2026-04) uses short interest + ETF flows + factor correlation. For A-shares, we replace short interest with available proxies:

| Signal | US Version | A-share Adaptation | Data Source | Frequency |
|--------|-----------|-------------------|-------------|-----------|
| S1: Position Concentration | Short interest HHI | Public fund overweight ratio + margin balance by sector | Wind/CSMAR quarterly + margin daily | Quarterly + Daily |
| S2: ETF Flow Intensity | Factor ETF flows / broad ETF flows | Sector/thematic ETF net flows / CSI 300 ETF flows | Eastmoney push2 API (~1,468 ETFs) | Daily |
| S3: Factor Return Correlation | Pairwise factor return correlation | Same - directly applicable | Factor return series | Daily |

### 1.2 Signal Construction

**S1 - Position Concentration (replaces short interest):**

```python
def position_concentration_signal(date):
    """
    A-share proxy for institutional crowding.
    Combines quarterly fund overweight + daily margin data.
    """
    # Quarterly: public fund overweight ratio by sector
    # Source: Wind/CSMAR, available ~15 days after quarter end
    fund_overweight = get_fund_overweight_ratio(sector, quarter)

    # Daily: margin balance concentration (HHI)
    # Margin data available T+1 from exchanges
    margin_by_sector = get_margin_balance_by_sector(date)
    margin_hhi = sum((s / total) ** 2 for s in margin_by_sector)

    # Combine: fund data when fresh, margin as daily proxy
    days_since_quarter = (date - last_quarter_end).days
    if days_since_quarter < 30:
        signal = 0.7 * fund_overweight + 0.3 * margin_hhi
    else:
        signal = margin_hhi  # degrade to margin-only after 30 days

    return signal
```

**S2 - ETF Flow Intensity:**

```python
def etf_flow_intensity_signal(date, window=20):
    """
    Ratio of factor-specific ETF inflows to broad market ETF inflows.
    A-share has rich sector/thematic ETF universe.
    """
    # Classify ETFs
    factor_etfs = get_etfs_by_category('sector')  # ~800+ sector ETFs
    broad_etfs = get_etfs_by_category('index')     # CSI 300, CSI 500, etc.

    # Rolling net inflows
    factor_inflows = sum(net_flow(etf, date, window) for etf in factor_etfs)
    broad_inflows = sum(net_flow(etf, date, window) for etf in broad_etfs)

    ratio = factor_inflows / max(broad_inflows, 1e-8)

    # Z-score vs 12-month history
    hist = get_rolling_history(ratio, window=252)
    z = (ratio - mean(hist)) / std(hist)

    return z
```

**S3 - Factor Return Correlation:**

```python
def factor_correlation_signal(date, window=60):
    """
    Mean pairwise correlation of Alpha158 factor returns.
    Rising correlation = behavioral convergence = crowding.
    """
    # Get daily returns for top-N most-traded factor portfolios
    factors = get_alpha158_factor_portfolios()
    returns = {f: get_returns(f, date, window) for f in factors}

    # Compute pairwise correlation matrix
    corr_matrix = compute_correlation_matrix(returns)

    # Mean off-diagonal correlation
    n = len(factors)
    mean_corr = (corr_matrix.sum() - n) / (n * (n - 1))

    # Historical z-score
    hist_corr = get_rolling_mean_corr_history(window=252*5)
    z = (mean_corr - mean(hist_corr)) / std(hist_corr)

    return z
```

### 1.3 Composite FCI

```python
def compute_fci(date):
    """
    Composite Factor Crowding Index for A-shares.
    Threshold: FCI > 1.5 = high crowding regime.
    """
    z1 = position_concentration_signal(date)
    z2 = etf_flow_intensity_signal(date)
    z3 = factor_correlation_signal(date)

    # Equal weight (can be optimized)
    fci = (z1 + z2 + z3) / 3.0

    return fci
```

### 1.4 Crowding-Adjusted Portfolio Rule

```
If FCI > 1.5:
    - Cut momentum/turnover factor exposure by 50%
    - Shift to judgment-based factors (value, quality) which decay slower
    - Increase cash buffer by 10%
If FCI > 2.0:
    - Cut momentum exposure to 25%
    - Activate tail-risk hedging (put spreads on crowded sectors)
```

---

## 2. Factor Return Correlation Monitoring (Real-Time)

### 2.1 Architecture

```
Daily Pipeline (T+1 after market close):
  1. Fetch factor returns for Alpha158 factors
  2. Compute rolling 60-day pairwise correlation matrix
  3. Extract: mean correlation, max eigenvalue (PCA), condition number
  4. Compare against 5-year rolling distribution
  5. Alert if any metric exceeds 95th percentile
```

### 2.2 Key Metrics

| Metric | What It Captures | Alert Threshold |
|--------|-----------------|-----------------|
| Mean pairwise corr | General factor convergence | > 0.25 (long-run avg ~0.05) |
| Max eigenvalue (PCA) | Dominant common factor strength | > 40% of total variance |
| Condition number | Correlation matrix instability | > 10x |
| Rolling IC decay | Individual factor alpha erosion | IC drops below 0.5x trailing avg |

### 2.3 Implementation

```python
class FactorCorrelationMonitor:
    def __init__(self, factor_universe, lookback=60, history_years=5):
        self.factors = factor_universe  # Alpha158 or subset
        self.lookback = lookback
        self.history_years = history_years

    def daily_update(self, date):
        returns = self._get_factor_returns(date, self.lookback)
        corr_matrix = np.corrcoef(returns)

        metrics = {
            'mean_corr': self._mean_offdiag(corr_matrix),
            'max_eigenvalue_pct': self._max_eigenvalue_pct(corr_matrix),
            'condition_number': np.linalg.cond(corr_matrix),
        }

        # Compare against historical distribution
        alerts = {}
        for name, value in metrics.items():
            threshold = self._get_percentile(name, 95)
            if value > threshold:
                alerts[name] = {
                    'value': value,
                    'threshold': threshold,
                    'percentile': self._to_percentile(name, value)
                }

        return metrics, alerts

    def _max_eigenvalue_pct(self, corr_matrix):
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        return eigenvalues[-1] / eigenvalues.sum()
```

---

## 3. Alpha Decay Rate Estimation

### 3.1 Hyperbolic Decay Model

Based on Lee (2025) "Not All Factors Crowd Equally":

```
alpha(t) = K / (1 + lambda * t)

Where:
  K = initial alpha capacity (factor-specific)
  lambda = strategy discovery rate
  t = time since factor became known/tradeable
```

**Key findings from the paper:**
- Momentum: R^2 = 0.65 (best fit), hyperbolic > exponential > linear
- Reversal: R^2 = 0.30 (moderate fit)
- Value/Quality: R^2 < 0.10 (judgment factors, poor fit)
- Crowding accelerated post-2015 (correlates with factor ETF growth, rho = -0.63)
- Crowding predicts TAIL RISK, not average returns (critical distinction)

### 3.2 Estimation Procedure

```python
import numpy as np
from scipy.optimize import curve_fit

def hyperbolic_decay(t, K, lambda_):
    """alpha(t) = K / (1 + lambda * t)"""
    return K / (1 + lambda_ * t)

def estimate_decay_params(factor_returns, window=36):
    """
    Estimate K and lambda for a factor using rolling Sharpe as alpha proxy.

    Args:
        factor_returns: monthly factor return series
        window: rolling window for Sharpe calculation (months)

    Returns:
        K, lambda_, R_squared, residual_alpha (current)
    """
    # Rolling 36-month Sharpe as alpha proxy
    rolling_sharpe = []
    for i in range(window, len(factor_returns)):
        rets = factor_returns[i-window:i]
        sr = np.mean(rets) / np.std(rets) * np.sqrt(12)
        rolling_sharpe.append(sr)

    # Filter to positive alpha periods (crowding relevant)
    positive_mask = np.array(rolling_sharpe) > 0
    t = np.arange(len(rolling_sharpe))[positive_mask]
    alpha_obs = np.array(rolling_sharpe)[positive_mask]

    # Fit hyperbolic model
    try:
        popt, pcov = curve_fit(hyperbolic_decay, t, alpha_obs,
                               p0=[alpha_obs[0], 0.01],
                               maxfev=10000)
        K, lambda_ = popt

        # R-squared
        alpha_pred = hyperbolic_decay(t, K, lambda_)
        ss_res = np.sum((alpha_obs - alpha_pred) ** 2)
        ss_tot = np.sum((alpha_obs - np.mean(alpha_obs)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Current residual alpha
        current_t = t[-1]
        residual_alpha = hyperbolic_decay(current_t, K, lambda_)

        return K, lambda_, r_squared, residual_alpha

    except RuntimeError:
        return None, None, None, None


def classify_factor_type(factor_name):
    """
    Mechanical factors: momentum, reversal, turnover (fits hyperbolic)
    Judgment factors: value, quality, growth (poor fit, slower decay)
    """
    mechanical = ['momentum', 'reversal_short', 'reversal_long',
                  'turnover', 'illiquidity']
    judgment = ['value', 'quality', 'growth', 'profitability', 'investment']

    if any(m in factor_name.lower() for m in mechanical):
        return 'mechanical'  # Expect hyperbolic decay, faster
    elif any(j in factor_name.lower() for j in judgment):
        return 'judgment'    # Slower decay, harder to arbitrage
    return 'unknown'
```

### 3.3 Decay-Adjusted Alpha Forecast

```python
def decay_adjusted_alpha(factor_name, current_alpha, months_since_discovery):
    """
    Adjust alpha forecast for crowding decay.

    Key insight from Lee (2025): use crowding for TAIL RISK, not mean return timing.
    """
    factor_type = classify_factor_type(factor_name)

    if factor_type == 'mechanical':
        # Fit hyperbolic model, get lambda
        K, lambda_, r2, residual = estimate_decay_params(factor_returns)

        if lambda_ and r2 > 0.3:
            # Apply decay adjustment
            decayed_alpha = hyperbolic_decay(months_since_discovery, K, lambda_)

            # But: average returns are efficiently priced (Lee 2025)
            # Use decay signal for RISK, not return timing
            tail_risk_multiplier = 1.0 + 0.5 * max(0, lambda_ - 0.02)

            return {
                'decayed_alpha': decayed_alpha,
                'tail_risk_multiplier': tail_risk_multiplier,
                'action': 'reduce_position_size' if lambda_ > 0.03 else 'hold'
            }

    elif factor_type == 'judgment':
        # Judgment factors: slower decay, use IC half-life instead
        ic_half_life = estimate_ic_half_life(factor_name)
        return {
            'decayed_alpha': current_alpha * (0.5 ** (months_since_discovery / ic_half_life)),
            'tail_risk_multiplier': 1.0,  # Lower crash risk
            'action': 'hold'  # More robust to crowding
        }
```

### 3.4 Key Insight: Crowding Predicts Crashes, Not Returns

From Lee (2025):

| Factor Type | Crowded State | Crash Probability | Action |
|-------------|--------------|-------------------|--------|
| Reversal (mechanical) | Crowded | 1.7-1.8x higher | Reduce, hedge |
| Momentum (mechanical) | Crowded | 0.38x lower | Can hold (trend reinforced) |
| Value/Quality (judgment) | N/A | Poor model fit | Monitor IC decay instead |

**Critical rule for Alpha158:** Most Alpha158 features are mechanical (turnover, momentum, reversal variants). They are HIGHLY susceptible to hyperbolic decay. The model predicts alpha should be ~50% of initial within 2-3 years of widespread adoption.

---

## 4. A-Shares Specific: ETF Flow Proxy Design

### 4.1 ETF Universe for Crowding Detection

| ETF Category | Count | Use Case |
|-------------|-------|----------|
| Broad index (CSI 300/500/1000) | ~50 | Denominator for flow ratio |
| Sector/thematic | ~800+ | Numerator, sector crowding |
| Smart beta/factor | ~100 | Direct factor exposure tracking |
| Cross-border (HK/US) | ~200 | Sentiment gauge |

**Data source:** Eastmoney push2 API via Apify actor (~1,468 ETFs, 4-tier order breakdown: super-large/large/medium/small).

### 4.2 Sector Crowding Map

```python
def build_sector_crowding_map(date):
    """
    Map ETF flows to factor exposures.
    Identifies which factors are seeing unusual capital concentration.
    """
    # Get sector ETF flows
    sector_flows = get_etf_flows(date, category='sector')  # ~496 sectors

    # Map sectors to factor tilts
    sector_factor_map = {
        'semiconductor': ['momentum', 'growth'],
        'consumer_electronics': ['momentum', 'turnover'],
        'banking': ['value', 'low_vol'],
        'real_estate': ['value', 'size_small'],
        'new_energy': ['growth', 'momentum'],
        # ... more mappings
    }

    # Aggregate flows by factor
    factor_flows = defaultdict(float)
    for sector, flow in sector_flows.items():
        if sector in sector_factor_map:
            for factor in sector_factor_map[sector]:
                factor_flows[factor] += flow

    # Compute crowding percentile for each factor
    crowding = {}
    for factor, flow in factor_flows.items():
        hist = get_historical_flows(factor, years=5)
        percentile = stats.percentileofscore(hist, flow)
        crowding[factor] = {
            'flow': flow,
            'percentile': percentile,
            'alert': percentile > 95
        }

    return crowding
```

### 4.3 Real-Time Signal: Super-Large vs Small Order Split

The Eastmoney API provides 4-tier order breakdown. The ratio of super-large orders to total flow is a strong institutional signal:

```python
def institutional_flow_signal(etf_ticker, date, window=20):
    """
    Super-large order net flow as % of total = informed money.
    Rising ratio = institutional crowding building.
    """
    flows = get_etf_flow_breakdown(etf_ticker, date, window)

    super_large_pct = flows['super_large_net'] / flows['total_net']

    # Compare to historical
    hist = get_historical_super_large_pct(etf_ticker, years=2)
    z = (super_large_pct - np.mean(hist)) / np.std(hist)

    return z
```

---

## 5. Integration with Existing ashare-lab Pipeline

### 5.1 Where FCI Fits in Alpha158 Workflow

```
Current:  Alpha158 features -> model -> signals -> backtest
With FCI: Alpha158 features -> model -> signals -> FCI gate -> position sizing
                                                        |
                                                   FCI > 1.5?
                                                   Yes: reduce 50%
                                                   No:  full size
```

### 5.2 Walk-Forward Enhancement

```python
def crowding_aware_walk_forward(model, factor_data, fci_series):
    """
    Standard walk-forward with crowding-adjusted alpha forecasts.
    """
    for test_period in walk_forward_periods:
        # Standard prediction
        raw_alpha = model.predict(factor_data[test_period])

        # Get FCI for this period
        fci = fci_series[test_period]

        # Adjust for crowding
        if fci > 1.5:
            # Mechanical factors: reduce exposure
            for factor in mechanical_factors:
                raw_alpha[factor] *= 0.5

            # Judgment factors: keep or increase (rotation)
            for factor in judgment_factors:
                raw_alpha[factor] *= 1.0  # unchanged

        elif fci > 2.0:
            # Extreme crowding: defensive posture
            raw_alpha *= 0.25  # across the board

        # Position sizing with decay forecast
        positions = optimize_positions(raw_alpha, risk_budget)

        yield test_period, positions
```

---

## 6. Estimated Effort

| Component | Scope | Effort | Dependencies |
|-----------|-------|--------|--------------|
| **ETF Flow Data Pipeline** | Eastmoney API integration, daily ETL | 3-5 days | Apify subscription or direct API |
| **FCI Signal Construction** | S1+S2+S3 implementation | 3-4 days | ETF data pipeline |
| **Factor Correlation Monitor** | Daily correlation matrix + PCA | 2-3 days | Factor return data |
| **Alpha Decay Estimation** | Hyperbolic model fitting + classification | 2-3 days | Historical factor returns |
| **Crowding-Adjusted Backtest** | Walk-forward with FCI gate | 2-3 days | FCI signals |
| **Alert System** | Threshold monitoring + notification | 1-2 days | All above |
| **TOTAL** | | **13-20 days** | |

### Phased Approach

**Phase 1 (1 week):** ETF flow data pipeline + basic FCI (S2 only)
**Phase 2 (1 week):** Full 3-signal FCI + correlation monitor
**Phase 3 (1 week):** Alpha decay estimation + crowding-aware backtest

---

## 7. Key References

1. **Lee, C.J. (2025)** - "Not All Factors Crowd Equally" - arXiv:2512.11913
   - Hyperbolic decay model, mechanical vs judgment factor taxonomy
2. **Quant Decoded (2026-04)** - "A Crowding Index Warned 2-4 Weeks Before Factor Crashes"
   - 3-signal FCI framework, US implementation
3. **Industrial Securities / Liu Yu (2026-05)** - A-share sector crowding tracking
   - Percentile-based crowding measurement, returns-vs-crowding quadrant
4. **UBS (2026-06)** - A-share tech stock crowding analysis
   - Fund overweight ratio as crowding proxy, 3-year style cycle
5. **Guosheng Securities (2026-07)** - ETF flow factor research
   - ETF flow factor IC=0.078, ICIR=2.50, multi-factor construction methodology
6. **Apify China ETF Flow Tracker** - ~1,468 ETFs, 4-tier order breakdown
   - Real-time data source for S2 signal
