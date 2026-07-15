# Deep Research: 12 Detection Blind Spots for ashare-lab

**Date**: 2026-07-15
**System**: A-share paper trading lab (Phase 3 COMPLETE, 424 tests, Alpha158+LGBModel, TRA)

---

## 1. Signal Quality Degradation — IC Always Null for Live Predictions

### Problem
The system computes IC (Information Coefficient) during backtesting walk-forward windows but cannot compute IC for live predictions because future returns are unknown. This means the system has zero visibility into whether its signals are degrading in real-time.

### Industry Standard

Production quant systems use **proxy metrics** that do not require future labels:

1. **Rolling IC of past predictions vs realized returns**: Once T+5 returns materialize, compute IC on the lagged prediction-return pairs. This creates a 5-day feedback delay but gives real signal quality data (Macrosynergy, 2023).

2. **Prediction distribution monitoring**: Log every prediction's distribution (mean, std, percentiles). If the distribution narrows abnormally or shifts mean, something is wrong. This is the "cheapest detector" — requires zero infrastructure beyond a prediction log (datarekha, 2026).

3. **Signal Quality Score (SQS)**: A composite 0-100 score combining data freshness, regime deviation, spread quality, and signal agreement. Below threshold blocks new positions (AlgoKing, 2026).

4. **NannyML CBPE (Confidence-Based Performance Estimation)**: Estimates model performance without ground truth labels by using prediction confidence distributions. Specifically designed for the "label delay" problem (NannyML).

5. **Meta-labeling (Lopez de Prado)**: A secondary classifier that predicts whether the primary signal's prediction will be correct, using features of the signal environment rather than future returns (SignalFlow, 2026).

### Detection Method

```python
# Tier 1: Lagged IC (available T+N days later)
def compute_lagged_ic(predictions_df, returns_df, lag_days=5):
    """Compute IC once returns materialize."""
    merged = predictions_df.merge(
        returns_df.shift(-lag_days),  # future returns become available
        left_index=True, right_index=True
    )
    return spearmanr(merged['prediction'], merged['return'])[0]

# Tier 2: Prediction distribution monitoring (real-time)
def monitor_prediction_distribution(predictions, baseline_stats):
    """Alert if prediction distribution shifts."""
    current_mean = predictions.mean()
    current_std = predictions.std()
    mean_zscore = (current_mean - baseline_stats['mean']) / baseline_stats['std']
    std_ratio = current_std / baseline_stats['std']
    if abs(mean_zscore) > 2.0 or std_ratio < 0.5 or std_ratio > 2.0:
        alert("prediction_distribution_shift", mean_z=mean_zscore, std_ratio=std_ratio)

# Tier 3: Effective rank collapse (detects model producing identical scores)
def detect_score_collapse(predictions, threshold=0.1):
    """Detect if predictions lose discriminative power."""
    unique_ratio = len(set(predictions)) / len(predictions)
    entropy = -sum(p * np.log(p + 1e-10) for p in np.histogram(predictions, bins=50)[0] / len(predictions))
    return unique_ratio < threshold or entropy < 1.0
```

### Implementation Complexity
- Tier 1 (lagged IC): **2-3 hours** — just add a delayed computation job
- Tier 2 (distribution monitoring): **4-6 hours** — log predictions to SQLite, compute daily stats
- Tier 3 (SQS composite): **1-2 days** — requires data freshness, regime, spread, agreement components

### A-Shares Specific
- A-share T+1 settlement means returns are available T+2 at earliest; use T+5 for 5-day holding period IC
- A-share market has limit-up/down (10%/20% for ChiNext/STAR), which distorts IC — filter out limit-hit days
- Baostock data has 1-2s/query latency; batch fetch prediction dates' returns rather than real-time

### Citations
- Macrosynergy (2023): "How to measure the quality of a trading signal"
- AlgoKing (2026): "Signal quality scoring: building a market-aware trade gate"
- datarekha (2026): "Drift — the silent killer of deployed models"
- NordVarg (2025): "Using ML & AI for Alpha Discovery"
- ml4t/diagnostic: "Statistical validation and diagnostics for quantitative trading strategies"

---

## 2. Factor Crowding — 1000 Quant Funds Using Identical Alpha158 Features

### Problem
Alpha158 is a well-known qlib feature set. Thousands of quant funds in China use identical or near-identical features, creating crowding risk. When crowded factors unwind, the drawdown is sudden and severe (2007 quant crisis, 2020 momentum crash).

### Industry Standard

1. **Factor Crowding Index (FCI)**: Composite of three signals — short interest concentration, factor ETF flow intensity, and pairwise factor return correlation. FCI > 1.5 sigma triggers exposure reduction. Provided 2-4 weeks advance warning before 2007, 2018, 2020, and 2022 dislocations (Quant Decoded, 2026).

2. **Factor Centrality (PCA-based)**: Centrality_i = lambda_i / sum(lambda_j) where lambda is factor loading on first principal component. Higher centrality = more crowded = higher drawdown risk (FinLab).

3. **Five-metric crowding model** (Hedge Fund Alpha, 2018): volatility, dispersion, correlation, momentum, valuations — each computed daily, standardized, equally weighted. Crowded factors show 18% probability of 15%+ drawdown vs 1% for uncrowded.

4. **Hyperbolic alpha decay model** (Lee, 2025): alpha(t) = K/(1+lambda*t) — mechanical factors (momentum, reversal) fit this model (R²=0.65); judgment factors (value, quality) do not. Crowding accelerated post-2015, correlating with factor ETF growth (rho=-0.63).

5. **Order flow imbalance correlation** (Bouchaud et al., 2020): Correlate anonymous market order imbalances with expected factor rebalancing flow. Momentum rebalancing explains 1-2% of order flow, increasing over time.

### Detection Method

```python
# Method 1: Factor return correlation (real-time, no lag)
def compute_factor_crowding_correlation(factor_returns, window=60):
    """Pairwise correlation of factor returns that should be uncorrelated."""
    corr_matrix = factor_returns.rolling(window).corr()
    # Value and momentum historically have corr ~-0.30
    # If corr rises toward 0 or positive = crowding signal
    return corr_matrix.mean().mean()

# Method 2: Factor centrality (PCA-based)
def compute_factor_centrality(factor_returns, window=12):
    """PCA-based crowding measure."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(factor_returns.rolling(window).cov())
    # Centrality = eigenvalue share of first PC
    return pca.explained_variance_ratio_[0]

# Method 3: Prediction concentration (specific to ashare-lab)
def detect_prediction_crowding(predictions_history, window=20):
    """If many funds use Alpha158, predictions will correlate."""
    # Track cross-sectional correlation of our predictions with
    # market consensus (e.g., index weight changes as proxy)
    rolling_corr = predictions_history.rolling(window).corr(market_consensus)
    return rolling_corr
```

### Implementation Complexity
- Factor correlation monitor: **4-6 hours** — compute daily, store in SQLite
- FCI composite: **2-3 days** — need short interest data (not available in China via baostock), ETF flow data (available via akshare), factor correlation (easy)
- Centrality: **4-6 hours** — PCA on factor returns

### A-Shares Specific
- China has no 13F equivalent; holdings data arrives with 1-quarter lag via fund reports
- Short interest data not available for A-shares (no centralized short-selling data)
- **Proxy for crowding**: Track CSI300/CSI500/CSI1000 index fund inflows (available via akshare), compute factor ETF flow ratio
- A-share factor crowding is particularly acute because: (a) limited hedging tools, (b) 10% limit-up/down amplifies momentum crowding, (c) retail-dominated market with herding behavior
- Use **turnover rate of Alpha158 top-decile stocks** as a crowding proxy — high turnover = crowded factor

### Citations
- Quant Decoded (2026): "A Crowding Index Warned 2-4 Weeks Before Factor Crashes"
- Lee (2025): "Not All Factors Crowd Equally" (arXiv:2512.11913)
- Bouchaud et al. (2020): "Zooming In on Equity Factor Crowding" (arXiv:2001.04185)
- Hedge Fund Alpha (2018): "Factor Crowding Model: Mob Management Measures"
- FinLab: "Factor Crowding Analysis" documentation

---

## 3. Feature Distribution Drift — No PSI Monitoring on Alpha158 Features

### Problem
The system has no monitoring for when Alpha158 feature distributions shift from their training-time baseline. If the market regime changes (e.g., from growth to value, from low-vol to high-vol), the model's learned patterns become invalid.

### Industry Standard

1. **Population Stability Index (PSI)**: The production standard. PSI = sum((actual% - expected%) * ln(actual%/expected%)). Thresholds: <0.1 stable, 0.1-0.25 moderate shift (investigate), >=0.25 significant drift (retrain). Originated in credit scoring, now universal (SentryML, 2026; Fiddler AI).

2. **Kolmogorov-Smirnov (KS) test**: Per-feature, per-day. Alert when statistic > threshold (more useful than p-value alone on large samples). Best for continuous features (datarekha, 2026).

3. **Jensen-Shannon Divergence**: Symmetric KL divergence, good for comparing distributions.

4. **Aggregate drift monitoring**: Track share of drifted features rather than individual features. If >30% of features show PSI > 0.1, the model is in trouble (Evidently AI).

5. **Prediction distribution monitoring**: The cheapest detector — just log predictions and alert when daily mean moves. Catches more real problems than any KS test (datarekha, 2026).

6. **whylogs**: Lightweight statistical sketches per request, <1ms overhead. Used by WhyLabs for continuous PSI computation (WhyLabs).

### Detection Method

```python
def compute_psi(baseline, current, bins=10):
    """Population Stability Index."""
    # Bin both distributions using baseline quantiles
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # Avoid division by zero
    baseline_counts = np.clip(baseline_counts, 0.001, None)
    current_counts = np.clip(current_counts, 0.001, None)

    psi = np.sum((current_counts - baseline_counts) * np.log(current_counts / baseline_counts))
    return psi

def monitor_alpha158_drift(current_features, baseline_features):
    """PSI monitoring for all Alpha158 features."""
    alerts = {}
    for col in current_features.columns:
        psi = compute_psi(baseline_features[col].dropna(), current_features[col].dropna())
        if psi >= 0.25:
            alerts[col] = {'psi': psi, 'status': 'CRITICAL'}
        elif psi >= 0.1:
            alerts[col] = {'psi': psi, 'status': 'WARNING'}
    return alerts
```

### Implementation Complexity
- Per-feature PSI: **4-6 hours** — compute daily after pipeline run, store in SQLite
- KS test: **2-3 hours** — scipy.stats.ks_2samp
- Aggregate dashboard: **1 day** — store history, alert on thresholds
- Alert pipeline: **4 hours** — integrate with existing WeChat report (Phase 4)

### A-Shares Specific
- Alpha158 has ~158 features; monitor top 20-30 by SHAP importance (from training)
- A-share features have strong seasonality (January effect, quarter-end window dressing) — use rolling 252-day baseline, not static training baseline
- ST/STAR board regime changes (20% vs 10% limit) create distribution shifts — detect and flag
- PSI thresholds may need calibration: A-share market is more volatile than US, so 0.25 may be too tight for some features

### Citations
- SentryML (2026): "Model Monitoring in Production: A Four-Layer Framework"
- datarekha (2026): "Drift — the silent killer of deployed models"
- Evidently AI (2025): "What is data drift in ML, and how to detect and handle it"
- aioutlooks (2026): "How to Detect Model Drift: PSI, KS Tests, and Concept Drift"
- AWS (2026): "Monitoring discriminative ML models using Amazon SageMaker AI with MLflow"

---

## 4. Intraday Crashes — All Risk Checks Post-Close

### Problem
The pipeline runs once daily after market close. If the market crashes intraday (e.g., -7% circuit breaker triggers in A-shares), the system has no mechanism to detect or respond until the next day's run.

### Industry Standard

1. **Circuit Breaker architecture** (RustyBT, 2026): Multiple breaker types — DrawdownCircuitBreaker (10% from 30-day peak), DailyLossCircuitBreaker (absolute $ loss), OrderRateCircuitBreaker (runaway algo protection), ManualCircuitBreaker (human kill switch). Coordinated by CircuitBreakerManager.

2. **State machine design** (QuantPython, 2026): ARMED -> MONITORING -> TRIGGERED -> COOLING_DOWN -> HALTED. Key: use Decimal (not float) for P&L, threading.Lock for HWM updates, idempotent trigger, non-blocking on_trigger callback.

3. **Pre-trade + post-trade checks** (algo-trading-platform): Pre-trade: leverage, concentration, drawdown. Post-trade: portfolio drawdown halt at 15%, daily loss halt. RiskAction enum: ALLOW/WARN/REJECT/HALT.

4. **Tiered circuit breakers** (Claude Trading Skills): Max daily loss 2%, losing streak cooldown 24h, weekly drawdown 5%, monthly drawdown 8%. Each has different release conditions.

5. **Volatility targeting** (RustyBT): scaling_factor = target_vol / current_vol. Rebalance monthly when current vol deviates >20% from target.

### Detection Method

```python
class IntradayRiskMonitor:
    """Intraday risk monitoring for paper trading."""

    def __init__(self, capital=300000):
        self.capital = capital
        self.peak_value = capital
        self.day_start_value = capital
        self.max_drawdown_pct = 0.10  # 10% from peak
        self.max_daily_loss_pct = 0.03  # 3% daily

    def check_intraday(self, current_value):
        """Call this with real-time NAV if available."""
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value
        daily_loss = (self.day_start_value - current_value) / self.day_start_value

        if drawdown >= self.max_drawdown_pct:
            return "HALT", f"Drawdown {drawdown:.1%} >= {self.max_drawdown_pct:.0%}"
        if daily_loss >= self.max_daily_loss_pct:
            return "HALT", f"Daily loss {daily_loss:.1%} >= {self.max_daily_loss_pct:.0%}"
        if drawdown >= self.max_drawdown_pct * 0.7:
            return "WARN", f"Drawdown {drawdown:.1%} approaching limit"
        return "OK", ""
```

### Implementation Complexity
- Post-close drawdown check: **1-2 hours** — add to existing pipeline step 9a
- Intraday monitoring (requires real-time data): **2-3 days** — need akshare real-time quote integration, scheduled polling during market hours
- Circuit breaker state machine: **1-2 days** — state persistence in SQLite, alert integration

### A-Shares Specific
- A-share circuit breakers: -7% halts trading for rest of day (since 2016 rule change, only price limits remain at ±10%/±20%)
- No pre-market or after-hours trading in A-shares — intraday monitoring only needed 9:30-15:00 CST
- akshare provides real-time quotes with ~3s delay (eastmoney backend)
- For paper trading: intraday monitoring is aspirational; the pipeline runs post-close. The immediate fix is to compute drawdown at pipeline run time and alert if >5%

### Citations
- RustyBT (2026): "Circuit Breakers" documentation
- QuantPython (2026): "Day 11 — Risk Limits: Hard-Coding Max-Drawdown Circuit Breakers"
- algo-trading-platform: "Risk Management" documentation
- Oyamori (2026): "Drawdown Limits — How to Build a Kill Switch"
- Claude Trading Skills: "Drawdown Circuit Breaker"

---

## 5. Real Execution Costs — Fixed 0.1% Slippage vs Real 2-5% in Crisis

### Problem
The backtest uses a fixed 0.1% slippage assumption. In reality, A-share slippage during crisis periods (limit-up/down, low liquidity) can be 2-5%. This means the backtest significantly overestimates real-world performance.

### Industry Standard

1. **Dynamic slippage models** (Hasbrouck, 1991; DCS extension 2022): Market impact = f(trade_size, volatility, liquidity). Time-varying instantaneous impact adapts to recent price and trade history.

2. **Square-root impact model**: Impact = sigma * sqrt(Q / V) where sigma = volatility, Q = trade size, V = daily volume. Standard at Renaissance, Two Sigma scale.

3. **Almgren-Chriss model**: Permanent + temporary impact. Optimal execution trajectory minimizes expected cost + risk penalty.

4. **Transaction Cost Analysis (TCA)**: Compare expected vs realized execution. Track implementation shortfall = paper return - actual return.

5. **Crisis-adjusted slippage**: During high-volatility regimes, multiply base slippage by volatility ratio. VIX-equivalent scaling.

### Detection Method

```python
def dynamic_slippage(trade_size, adv, volatility, spread_pct, regime='normal'):
    """Dynamic slippage model for A-shares."""
    # Base: square-root impact
    base_impact = volatility * np.sqrt(trade_size / max(adv, 1))

    # Spread component
    spread_cost = spread_pct / 2  # half spread

    # Crisis multiplier
    crisis_mult = {
        'normal': 1.0,
        'elevated': 2.0,  # VIX > 25 equivalent
        'crisis': 5.0,    # limit-up/down, circuit breaker
    }.get(regime, 1.0)

    total_slippage = (base_impact + spread_cost) * crisis_mult

    # Cap at reasonable bounds
    return np.clip(total_slippage, 0.0005, 0.05)  # 0.05% to 5%

def estimate_adv(stock_code, lookback=20):
    """Average daily volume from baostock."""
    # Use baostock query_history_k_data_plus for volume
    pass
```

### Implementation Complexity
- Square-root impact model: **4-6 hours** — simple formula, need ADV data
- Regime-adjusted multiplier: **2-3 hours** — use market volatility as proxy
- Full TCA pipeline: **3-5 days** — need order-level data, execution timestamps
- Integration with backtest engine: **1-2 days** — replace fixed slippage in engine.py

### A-Shares Specific
- A-share tick size: 0.01 CNY (fixed), so spread cost is relatively predictable
- Daily volume data available via baostock (query_history_k_data_plus with volume field)
- A-share specific: ST stocks have 5% limit, ChiNext/STAR have 20% limit — adjust slippage model per board
- Commission: stamp tax 0.05% (sell only) + broker commission 0.025% (min 5 CNY) — these are fixed, not slippage
- For paper trading: use conservative 0.3% total cost (vs current 0.1%) as interim fix

### Citations
- Hasbrouck (1991); DCS extension (2022): "Measuring price impact and information content of trades"
- Almgren & Chriss (2001): "Optimal execution of portfolio transactions"
- Gârleanu & Pedersen (2013): "Dynamic trading with predictable returns and transaction costs"

---

## 6. Prediction Collapse — TRA Produces Identical Scores for All Stocks

### Problem
If the TRA model degenerates to producing identical or near-identical predictions for all stocks, the system cannot distinguish good from bad stocks. This is a form of representational collapse.

### Industry Standard

1. **Effective Rank** (Roy & Vetterli, 2007): eff_rank = exp(-sum(p * log(p))) where p = singular values / sum. Score in [0,1] where 1.0 = full collapse. Used by MONAI's EmbeddingCollapseMetric (MONAI, 2026).

2. **Silent Collapse detection** (2025): Three precursors — contraction of anchor entropy (overconfidence), freezing of representation drift, erosion of tail coverage. These manifest multiple iterations before validation metrics change.

3. **SIGMA framework** (2026): Track spectral contraction of prediction Gram matrix. Deterministic + stochastic bounds on eigenvalue spectrum.

4. **Prediction entropy monitoring**: If prediction entropy drops below threshold, model is producing degenerate outputs. Simple and effective.

5. **Unique ratio**: len(unique(predictions)) / len(predictions). If < 0.1, predictions have collapsed.

### Detection Method

```python
def detect_prediction_collapse(predictions, baseline_entropy=None):
    """Multi-metric collapse detection."""
    metrics = {}

    # 1. Unique ratio
    unique_ratio = len(set(np.round(predictions, 4))) / len(predictions)
    metrics['unique_ratio'] = unique_ratio
    metrics['unique_ratio_alert'] = unique_ratio < 0.1

    # 2. Prediction entropy
    hist, _ = np.histogram(predictions, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist)) * (predictions.max() - predictions.min()) / 50
    metrics['entropy'] = entropy

    # 3. Effective rank (on prediction vector)
    centered = predictions - predictions.mean()
    sv = np.linalg.svd(centered.reshape(-1, 1), compute_uv=False)
    sv_norm = sv / sv.sum()
    eff_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-10)))
    metrics['effective_rank'] = eff_rank

    # 4. Standard deviation ratio vs baseline
    if baseline_entropy:
        std_ratio = predictions.std() / baseline_entropy['std']
        metrics['std_ratio'] = std_ratio
        metrics['std_ratio_alert'] = std_ratio < 0.3

    # Alert if any metric indicates collapse
    metrics['collapse_detected'] = (
        metrics.get('unique_ratio_alert', False) or
        metrics.get('std_ratio_alert', False) or
        entropy < 1.0
    )

    return metrics
```

### Implementation Complexity
- Basic collapse detection (unique ratio, std): **2-3 hours** — add to pipeline after predict.py
- Effective rank: **4-6 hours** — numpy SVD
- Entropy monitoring with baseline: **4-6 hours** — need to establish baseline from first N predictions
- Alert integration: **2 hours** — add to WeChat report

### A-Shares Specific
- A-share universe (CSI1000) has ~1000 stocks; if effective rank drops below 100, predictions are degenerate
- TRA model uses Alpha158 features which are highly correlated; some loss of diversity is expected
- Monitor per-window: if a walk-forward window produces IC < 0.01, flag as potential collapse
- Compare prediction std across windows: sudden std drop = model converged to trivial solution

### Citations
- MONAI (2026): "EmbeddingCollapseMetric" — SVD effective rank, centroid similarity, separation
- SIGMA (2026): "Scalable Spectral Insights for LLM Model Collapse" (arXiv:2601.03385)
- Silent Collapse (2025): "Silent Collapse in Recursive Learning Systems" (arXiv:2605.14588)
- Kalinowski (2026): "Monitoring Neural Training with Topology" (arXiv:2604.26984)
- Roy & Vetterli (2007): "The effective rank"

---

## 7. Crash Alerts Suppression — Report Gate Blocks >20% NAV Changes

### Problem
The report system has a gate that blocks reporting of NAV changes >20%, presumably to prevent alarming the user. But this means the most critical events (crashes) are exactly the ones that get suppressed.

### Industry Standard

1. **Tiered alerting** (industry standard): INFO/WARN/CRITICAL/EMERGENCY levels. Critical and Emergency always alert, regardless of magnitude. The bigger the change, the MORE urgent the alert.

2. **Never suppress crash alerts**: Every production trading system treats large drawdowns as mandatory alerts. Suppression is only for normal operations (routine P&L updates).

3. **Multi-channel alerting**: Critical alerts go to multiple channels (email + SMS + push). If one channel fails, others catch it.

4. **Alert fatigue management**: Reduce noise on normal operations, amplify on anomalies. This is the opposite of suppressing large changes.

### Detection Method

```python
# Fix: invert the gate logic
def should_alert(nav_change_pct, threshold_normal=0.05, threshold_critical=0.10):
    """Alert gating logic — OPPOSITE of current behavior."""
    abs_change = abs(nav_change_pct)

    if abs_change >= threshold_critical:
        return 'CRITICAL', True  # ALWAYS alert — never suppress
    elif abs_change >= threshold_normal:
        return 'WARNING', True
    else:
        return 'INFO', False  # Suppress routine updates
```

### Implementation Complexity
- Fix the gate logic: **1 hour** — simple conditional inversion
- Multi-channel alerting: **4-6 hours** — add iLink priority for critical alerts
- Alert history tracking: **2 hours** — store in SQLite

### A-Shares Specific
- A-share market has 10% daily limit; a >10% NAV change in one day is extraordinary and must alert
- Use the existing iLink Bot infrastructure (Phase 4) for critical alerts — it already works
- Consider: if NAV drops >5% in a day, send immediate alert (don't wait for pipeline run)

### Citations
- Oyamori (2026): "It halts unconditionally. A circuit breaker that generates an alert but does not prevent order submission is a notification, not a circuit breaker."
- RustyBT: "Always use DrawdownCircuitBreaker and DailyLossCircuitBreaker in live trading"

---

## 8. Database Corruption — No Integrity Check Before Backup

### Problem
The paper.db SQLite database has no integrity verification. If corruption occurs (WAL bug, disk error, unclean shutdown), the backup will contain corrupted data, and the system won't know.

### Industry Standard

1. **PRAGMA integrity_check**: SQLite's built-in corruption detector. Run before every backup. Returns "ok" or list of corruption details (SQLite docs).

2. **WAL mode + proper checkpointing**: Use WAL mode (journal_mode=WAL) for crash safety. Checkpoint after critical transactions. Known WAL-reset bug in SQLite 3.7.0-3.51.2 (fixed 3.51.3) — upgrade if on affected version.

3. **Page-level checksums**: cksm VFS extension adds Fletcher checksums to every database page. Detects bit-rot that WAL checksums miss. SQLITE_IOERR_DATA on mismatch (gosqlite.org).

4. **Never use fs.copyFile() for WAL-mode databases**: Always use SQLite's native backup API or .backup() method. fs.copyFile() copies only .db, missing .db-wal and .db-shm = instant corruption (Scott Spence, 2025).

5. **Aggressive checkpointing**: Don't let WAL grow indefinitely. Default autocheckpoint is 1000 pages; for critical data, checkpoint after every critical transaction.

6. **Application-level checksums**: Store hash of critical columns (NAV, positions) for verification on retrieval.

### Detection Method

```python
import sqlite3

def verify_database_integrity(db_path):
    """Run before every backup."""
    conn = sqlite3.connect(db_path)
    result = conn.execute("PRAGMA integrity_check").fetchone()
    conn.close()
    if result[0] != 'ok':
        raise DatabaseCorruptionError(f"integrity_check failed: {result[0]}")
    return True

def safe_backup(source_path, backup_path):
    """Safe SQLite backup using native API."""
    import shutil
    # First verify integrity
    verify_database_integrity(source_path)

    # Use SQLite backup API (not file copy)
    src = sqlite3.connect(source_path)
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

    # Verify backup integrity
    verify_database_integrity(backup_path)

def compute_nav_hash(conn):
    """Application-level integrity check."""
    row = conn.execute("SELECT nav, cash FROM nav ORDER BY date DESC LIMIT 1").fetchone()
    import hashlib
    return hashlib.sha256(f"{row[0]}:{row[1]}".encode()).hexdigest()
```

### Implementation Complexity
- PRAGMA integrity_check: **1 hour** — add to backup function
- Safe backup (native API): **2-3 hours** — replace any file-copy backups
- Page-level checksums (cksm): **4-6 hours** — requires VFS extension, may not be available in Python sqlite3
- Application-level hash: **2-3 hours** — add to ledger.py

### A-Shares Specific
- X500 (192.168.100.10) runs the pipeline; if it crashes mid-write, WAL may be left in inconsistent state
- X500 uses ext4 filesystem; no known corruption issues, but always verify before backup
- Backup frequency: after every pipeline run (daily), keep 7 days of backups
- Current cleanup_old_backups in ledger.py should be extended with integrity verification

### Citations
- SQLite (2026): "Write-Ahead Logging" documentation
- SQLite (2026): "How To Corrupt An SQLite Database File"
- Scott Spence (2025): "SQLite Corruption with fs.copyFile() in WAL Mode"
- gosqlite.org: "cksm package — corruption-detection VFS for SQLite"
- avi.im (2025): "PSA: SQLite WAL checksums fail silently and may lose data"
- db-news.com (2025): "The Hidden Risks of Silent WAL Truncation"

---

## 9. Stale Prediction Cutoff — No Age Limit on Fallback Predictions

### Problem
If the prediction pipeline fails, the system may use old predictions without any staleness check. Predictions from 30 days ago are worthless for a 5-day holding period strategy.

### Industry Standard

1. **Freshness SLAs** (Feast, 2024): Define maximum acceptable staleness per feature/prediction. `freshness_ms: 3600000` = 1 hour. Reject any value older than SLA (Theneuralbase).

2. **TTL (Time-To-Live) on predictions**: Every prediction file carries a timestamp. If age > holding_period * 2, reject.

3. **Prediction age monitoring**: Log prediction timestamps. Alert if fallback predictions older than threshold are used.

4. **Fail-closed design**: If no fresh predictions available, do NOT trade. Better to miss a day than trade on stale signals.

### Detection Method

```python
from datetime import datetime, timedelta

def check_prediction_freshness(prediction_date, max_age_days=10):
    """Reject predictions older than 2x holding period."""
    pred_dt = datetime.strptime(prediction_date, '%Y-%m-%d')
    age = (datetime.now() - pred_dt).days

    if age > max_age_days:
        return False, f"STALE: prediction is {age} days old (max {max_age_days})"
    if age > max_age_days * 0.7:
        return True, f"WARNING: prediction is {age} days old"
    return True, "FRESH"

def get_latest_fresh_predictions(predictions_dir, max_age_days=10):
    """Get most recent predictions within freshness window."""
    import glob
    pred_files = sorted(glob.glob(f"{predictions_dir}/pred_*.parquet"), reverse=True)
    for f in pred_files:
        pred_date = extract_date_from_filename(f)
        fresh, msg = check_prediction_freshness(pred_date, max_age_days)
        if fresh:
            return f, msg
    return None, "NO FRESH PREDICTIONS AVAILABLE — DO NOT TRADE"
```

### Implementation Complexity
- Freshness check: **1-2 hours** — add to pipeline step that loads predictions
- Fail-closed logic: **2-3 hours** — modify pipeline to halt if no fresh predictions
- Alert on stale fallback: **1 hour** — add to WeChat report

### A-Shares Specific
- 5-day holding period means predictions should be max 10 days old (2x)
- GPU machine (192.168.100.11) generates predictions; X500 (192.168.100.10) runs pipeline — sync failures can leave stale predictions
- baostock data availability: T+1 for daily data, so predictions for day T use data through T-1
- Holiday periods (Chinese New Year, National Day) can cause 7-10 day gaps — adjust max_age for holidays

### Citations
- Feast (2024): "Freshness SLAs" — freshness_ms parameter
- SentryML (2026): "Silent breakage: Null rate on a key feature goes from 0.1% to 40%"
- Industry best practice: "Fail-closed" design for production ML systems

---

## 10. Retraining Necessity — Dead Retrain Gate

### Problem
The system has a retrain gate but it never fires because the metrics it monitors are always null or stale. The model never gets retrained, and there's no trigger to know when it should be.

### Industry Standard

1. **Hybrid retraining** (Arize, 2023; SmartDev, 2025): Baseline schedule (e.g., monthly) + accelerated triggers (PSI > 0.25, IC drop > 30%, prediction distribution shift). Minimum interval between retrains prevents churn.

2. **Three-tier trigger system** (SmartDev, 2025):
   - Primary: Performance metric drops below threshold (IC < 0.02 for 5 consecutive days)
   - Secondary: PSI > 0.25 on top features
   - Tertiary: External events (market regime change, new regulations)

3. **Uncertainty-based retraining** (Regol et al., 2025): Forecast model performance evolution using bounded metrics. Retrain when predicted performance crosses threshold. Outperforms drift-detection baselines on 7 datasets.

4. **Champion-challenger**: Always have a challenger model in shadow mode. Promote challenger when it consistently outperforms champion on recent data.

5. **Never auto-retrain on drift alone** (datarekha, 2026): Drift can be upstream bug, seasonality, or concept drift requiring new labels. Investigate before retraining.

### Detection Method

```python
class RetrainGate:
    """Multi-signal retraining trigger."""

    def __init__(self, config):
        self.min_interval_days = config.get('min_interval_days', 14)
        self.ic_threshold = config.get('ic_threshold', 0.02)
        self.psi_threshold = config.get('psi_threshold', 0.25)
        self.last_retrain_date = None

    def should_retrain(self, current_ic, feature_psi, prediction_drift):
        """Check all triggers."""
        # Respect minimum interval
        if self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since < self.min_interval_days:
                return False, f"Too soon ({days_since}d < {self.min_interval_days}d)"

        # Primary trigger: IC degradation
        if current_ic is not None and current_ic < self.ic_threshold:
            return True, f"IC degraded to {current_ic:.4f} < {self.ic_threshold}"

        # Secondary trigger: feature drift
        drifted_features = [f for f, psi in feature_psi.items() if psi >= self.psi_threshold]
        if len(drifted_features) > len(feature_psi) * 0.3:
            return True, f"{len(drifted_features)} features drifted (PSI > {self.psi_threshold})"

        # Tertiary trigger: prediction distribution shift
        if prediction_drift.get('collapse_detected'):
            return True, "Prediction collapse detected"

        return False, "All signals nominal"
```

### Implementation Complexity
- Basic retrain gate with PSI + IC: **1-2 days** — need to fix IC computation (Blind Spot #1)
- Champion-challenger: **1 week** — need shadow model infrastructure
- Uncertainty-based: **2-3 days** — implement Regol et al. method
- Full hybrid pipeline: **1-2 weeks** — schedule + triggers + data validation + model promotion

### A-Shares Specific
- Walk-forward windows are ~6 months; retraining should happen at least every 3 months
- A-share market has strong seasonal patterns; retrain after Chinese New Year, after earnings season
- GPU training requires Windows machine (192.168.100.11) — manual process currently
- Consider: automated retrain trigger sends alert to user, user initiates training (semi-automatic)

### Citations
- Arize (2023): "A Guide To Optimized Retraining"
- SmartDev (2025): "AI Model Drift & Retraining: A Guide for ML System Maintenance"
- Regol et al. (2025): "When to retrain a machine learning model" (MLR Proceedings)
- datarekha (2026): "Drift — the silent killer of deployed models"
- AWS (2026): "Monitoring discriminative ML models using Amazon SageMaker AI with MLflow"

---

## 11. Neutralization Quality During Crisis — OLS Breaks Down

### Problem
The neutralization step uses OLS regression to remove market-cap and industry exposure. During crisis periods, returns have extreme outliers that bias OLS estimates, causing neutralization to fail precisely when it's needed most.

### Industry Standard

1. **MM-estimator** (Yohai, 1987): High breakdown point (50%) + high efficiency (95%). The gold standard for robust regression in finance. Used by Axioma for fundamental factor models. Outperforms OLS on 26% of microcap stocks, 14% of smallcaps (Martin, 2022).

2. **mOpt estimator** (Martin, 2022): Minimizes maximum bias over Tukey-Huber family. 99% normal distribution efficiency. Smooth outlier rejection for scaled residuals > 3.0. Superior to Huber estimator which can have arbitrarily large bias.

3. **Hausman test for LS vs Robust** (Martin & Simin, 2022): Statistical test to determine if OLS and robust estimates differ significantly. If they do, use robust. Available in R package `robust`.

4. **Winsorization as fallback**: Clip returns at 1st/99th percentile before OLS. Simple but loses information about outlier magnitude.

5. **Quantile regression**: Median regression (L1) is naturally robust to outliers. Less efficient than MM but simpler to implement.

### Detection Method

```python
def robust_neutralize(returns, market_cap, industry, method='mm'):
    """Robust neutralization replacing OLS."""
    from sklearn.linear_model import HuberRegressor

    # Construct design matrix
    X = construct_design_matrix(market_cap, industry)

    if method == 'huber':
        model = HuberRegressor(epsilon=1.35)  # 95% efficiency
        model.fit(X, returns)
        residuals = returns - model.predict(X)

    elif method == 'mm':
        # Use statsmodels robust linear model
        import statsmodels.api as sm
        rlm = sm.RLM(returns, X, M=sm.robust.norms.TukeyBiweight())
        result = rlm.fit()
        residuals = result.resid

    elif method == 'winsorize_ols':
        # Simple fallback: winsorize then OLS
        from scipy.stats.mstats import winsorize
        returns_w = winsorize(returns, limits=[0.01, 0.01])
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, returns_w)
        residuals = returns - model.predict(X)

    return residuals
```

### Implementation Complexity
- Huber regression: **2-3 hours** — sklearn.linear_model.HuberRegressor
- MM-estimator: **4-6 hours** — statsmodels.RLM with TukeyBiweight
- Hausman test: **4-6 hours** — R package `robust` or custom implementation
- Full robust neutralization pipeline: **1-2 days** — replace OLS in adjust.py, add fallback

### A-Shares Specific
- A-share returns have heavier tails than US (retail-driven, limit-up/down effects)
- Industry classification: use SW (Shenwan) industry classification, not GICS
- Market-cap neutralization: log(market_cap) is standard; A-share small-caps are extremely volatile
- Crisis periods: 2015 A-share crash (-45% in 3 months), 2018 trade war, 2020 COVID — all show OLS neutralization failure
- The existing neutralize_backtest_results show R14 neutralization hurt W1/W3/W4 but rescued W5/W6 — robust methods would improve W5/W6 further

### Citations
- Martin (2022): "Robust estimation of factor models in finance" (UW dissertation)
- Martin & Simin (2022): "A Hausman Type Test for Differences between LS and Robust Factor Model Betas"
- Martin (2022): "mOpt estimator" — Journal of Portfolio Management
- Yohai (1987): "High Breakdown-Point and High Efficiency Robust Estimates for Regression"
- EUR Thesis (2024): "On the robustness of factor p-values in multi-factor asset pricing models"
- Sorokina et al. (2013): "Robust Methods in Event Studies"

---

## 12. Double Pipeline Execution — No Mutex

### Problem
If the pipeline runs twice (cron overlap, manual trigger during scheduled run), both executions will read/write the same SQLite database, potentially causing corruption or double trades.

### Industry Standard

1. **File-based mutex (flock)**: Standard Unix file locking. Pipeline acquires exclusive lock on PID file at start, releases at end. If lock is held, exit immediately.

2. **Database-level locking**: SQLite supports EXCLUSIVE transactions. But application-level mutex is preferred because DB locks can deadlock.

3. **Distributed lock** (Redis/etcd): For multi-machine setups. Not needed for ashare-lab (single X500).

4. **Idempotency keys**: Each pipeline run gets a unique run_id. All writes include run_id; duplicate runs are detected by run_id collision.

5. **Process-level singleton**: Use PID file + process existence check. If PID file exists and process is alive, exit.

### Detection Method

```python
import fcntl
import os
import sys

class PipelineMutex:
    """File-based mutex for pipeline execution."""

    def __init__(self, lock_path='/tmp/ashare_pipeline.lock'):
        self.lock_path = lock_path
        self.lock_fd = None

    def acquire(self):
        """Try to acquire exclusive lock. Returns True if acquired."""
        try:
            self.lock_fd = open(self.lock_path, 'w')
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(str(os.getpid()))
            self.lock_fd.flush()
            return True
        except (IOError, OSError):
            # Lock is held by another process
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return False

    def release(self):
        """Release the lock."""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            os.unlink(self.lock_path)

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Pipeline already running — mutex held")
        return self

    def __exit__(self, *args):
        self.release()

# Usage in pipeline.py:
def main():
    with PipelineMutex():
        run_pipeline()
```

### Implementation Complexity
- File-based mutex: **1-2 hours** — add to pipeline.py main()
- Idempotency keys: **4-6 hours** — add run_id to all database writes
- Database-level locking: **2-3 hours** — SQLite BEGIN EXCLUSIVE

### A-Shares Specific
- Single execution host (X500) means file-based mutex is sufficient
- Cron schedule: pipeline runs at market close (15:00 CST) — add mutex to prevent overlap
- Manual runs: user may trigger pipeline manually during market hours for testing — mutex prevents conflict
- Consider: add run_id to paper.db tables (orders, trades, nav) for audit trail

### Citations
- POSIX flock(2): Standard Unix file locking
- SQLite documentation: "WAL-mode File Format" — locking protocol
- Industry standard: Every production pipeline uses process-level mutex

---

## Priority Ranking for ashare-lab

| Priority | Blind Spot | Risk Level | Effort | Impact |
|----------|-----------|------------|--------|--------|
| P0 | #7 Crash alerts suppression | CRITICAL | 1h | Fixes the exact inverse of correct behavior |
| P0 | #8 Database corruption | CRITICAL | 3h | Prevents silent data loss |
| P0 | #12 Double pipeline | HIGH | 2h | Prevents data corruption from concurrent runs |
| P1 | #9 Stale predictions | HIGH | 2h | Prevents trading on worthless signals |
| P1 | #6 Prediction collapse | HIGH | 4h | Detects model failure |
| P1 | #3 Feature drift (PSI) | HIGH | 6h | Core monitoring capability |
| P2 | #1 Signal quality (lagged IC) | MEDIUM | 3h | Delayed but real signal quality |
| P2 | #5 Dynamic slippage | MEDIUM | 6h | More realistic backtest |
| P2 | #10 Retraining triggers | MEDIUM | 2d | Model freshness |
| P2 | #11 Robust neutralization | MEDIUM | 1d | Crisis resilience |
| P3 | #4 Intraday monitoring | LOW | 2d | Paper trading doesn't need real-time |
| P3 | #2 Factor crowding | LOW | 3d | Long-term risk, not immediate |

**Recommended order**: Fix P0 items first (3 hours total), then P1 items (12 hours total), then P2 items (5 days total).
