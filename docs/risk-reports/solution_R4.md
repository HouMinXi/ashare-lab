# Risk R4 Solution: Model Signal Quality Monitoring

## Problem Statement

The TRA neural network model produces daily predictions for ~1000 CSI1000 stocks.
The model is 23.9 days old. IC (Information Coefficient) is always `None` for live
predictions because the TRA model's internal IC metric requires labels (forward
returns) that do not exist at prediction time. The system has zero signal quality
feedback and could be trading on pure noise without knowing.

## Root Cause Analysis

### Why IC Is Always Null

In `predict.py` lines 230-255, IC capture attempts:
```python
ic_attr = getattr(model, "_ic_list", None)  # TRA internal state
```
For live/OOS predictions, TRA's `test_epoch(-1)` runs without labels, so `_ic_list`
is empty or NaN. The `append_ic_history()` in pipeline.py records `None` to
`data/ic_history.tsv` every day -- a monitoring infrastructure that monitors nothing.

### What Is Missing

| Gap | Impact | Detection |
|-----|--------|-----------|
| No lagged IC | Cannot verify signal quality | IC always None |
| No PSI monitoring | Cannot detect distribution drift | Silent degradation |
| No prediction entropy | Cannot detect score collapse | Model outputs uniform noise |
| No feature drift detection | Cannot detect covariate shift | Silent alpha decay |
| No meta-labeling | Cannot filter false positives | Trades on noise |

## Solution Architecture: Three-Layer Defense

```
Layer 1 (T+0, no labels)     Layer 2 (T+5, lagged)       Layer 3 (future)
Prediction Distribution       Lagged IC Computation       Meta-Labeling
Monitoring (PSI + entropy)    (Spearman rank corr)        (secondary classifier)
     |                              |                           |
     v                              v                           v
  meta.json                   ic_lagged.tsv               position sizing
  + pipeline alert            + pipeline alert            + signal gating
```

---

## Layer 1: Prediction Distribution Monitoring (Immediate, No Labels)

### Design

Run after every prediction in `predict.py`. Compares today's score distribution
against a rolling reference window. Requires NO future labels.

### Metrics

#### 1a. Population Stability Index (PSI) on Score Distribution

Reference: rolling 20-day window of prediction scores from `predictions/*.parquet`.
Current: today's prediction scores.

```
PSI = SUM( (pct_current[i] - pct_reference[i]) * ln(pct_current[i] / pct_reference[i]) )
```

Binning: quantile-based (10 bins from reference distribution) to handle heavy tails.

Thresholds:
- PSI < 0.10: Stable
- 0.10 <= PSI < 0.20: Investigate (log warning)
- PSI >= 0.20: Critical (alert + flag in meta.json)

#### 1b. Prediction Entropy

Shannon entropy of binned prediction scores. Detects score collapse (model outputs
near-identical scores = no discriminating power).

```
H = -SUM( p[i] * ln(p[i] )  over 10 quantile bins
```

Reference entropy: computed from training/validation set scores.
Threshold: current_entropy < 0.5 * reference_entropy = ALERT (score collapse).

#### 1c. Score Concentration (Herfindahl Index)

Measures how concentrated the top-K scores are. If the model puts all weight on
a few stocks, concentration spikes.

```
HHI = SUM( (score[i] / SUM(scores))^2 )
```

Threshold: HHI > 2x training HHI = ALERT (over-concentration).

#### 1d. Prediction Count Stability

Simple check: does the number of valid (finite) predictions match the universe size?
A sudden drop indicates data pipeline failure.

Threshold: count < 0.9 * expected_count = ALERT.

### Implementation

File: `ashare_lab/research/monitor.py` (new module)

```python
"""Prediction distribution monitoring (label-free).

Provides compute_prediction_diagnostics() which compares today's
prediction scores against a rolling reference window using PSI,
entropy, and concentration metrics.  No future labels required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_lab.config import PREDICTIONS_DIR

log = logging.getLogger(__name__)

REFERENCE_WINDOW = 20  # rolling window in trading days
N_BINS = 10            # quantile bins for PSI


@dataclass
class PredictionDiagnostics:
    """Label-free prediction quality metrics."""
    psi: float              # Population Stability Index
    entropy: float          # Shannon entropy of score distribution
    entropy_ratio: float    # current / reference entropy
    hhi: float              # Herfindahl concentration index
    n_valid: int            # count of finite scores
    psi_alert: bool         # PSI >= 0.20
    entropy_alert: bool     # entropy < 50% of reference
    concentration_alert: bool  # HHI > 2x reference
    overall_status: str     # "ok" | "warning" | "critical"


def _load_reference_scores(n_days: int = REFERENCE_WINDOW) -> np.ndarray:
    """Load scores from the last n_days prediction parquet files."""
    pred_files = sorted(PREDICTIONS_DIR.glob("*.parquet"))
    # Exclude today (will be the "current" distribution)
    pred_files = [f for f in pred_files if not f.name.endswith(".meta.json")]
    pred_files = pred_files[-(n_days + 1):-1]  # exclude last (today)

    scores = []
    for f in pred_files:
        try:
            df = pd.read_parquet(f)
            scores.extend(df["score"].dropna().values)
        except Exception:
            continue
    return np.array(scores) if scores else np.array([])


def _compute_psi(reference: np.ndarray, current: np.ndarray,
                 n_bins: int = N_BINS) -> float:
    """Compute Population Stability Index with quantile-based bins."""
    if len(reference) < n_bins or len(current) < n_bins:
        return 0.0

    # Quantile-based bins from reference distribution
    edges = np.percentile(reference,
                          np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)

    # Avoid log(0)
    eps = 1e-4
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    # Normalize
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _compute_entropy(scores: np.ndarray, n_bins: int = N_BINS) -> float:
    """Shannon entropy of binned score distribution."""
    if len(scores) < n_bins:
        return 0.0
    edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(scores, bins=edges)[0]
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _compute_hhi(scores: np.ndarray) -> float:
    """Herfindahl-Hirschman Index on normalized scores."""
    if len(scores) == 0:
        return 0.0
    abs_scores = np.abs(scores)
    total = abs_scores.sum()
    if total == 0:
        return 0.0
    shares = abs_scores / total
    return float(np.sum(shares ** 2))


def compute_prediction_diagnostics(
    current_scores: np.ndarray,
) -> PredictionDiagnostics:
    """Compute label-free diagnostics for today's predictions.

    Args:
        current_scores: Array of finite prediction scores for today.

    Returns:
        PredictionDiagnostics with all metrics and alert flags.
    """
    ref_scores = _load_reference_scores()

    # PSI
    psi = _compute_psi(ref_scores, current_scores) if len(ref_scores) > 0 else 0.0

    # Entropy
    current_entropy = _compute_entropy(current_scores)
    ref_entropy = _compute_entropy(ref_scores) if len(ref_scores) > 0 else current_entropy
    entropy_ratio = current_entropy / ref_entropy if ref_entropy > 0 else 1.0

    # Concentration
    current_hhi = _compute_hhi(current_scores)
    ref_hhi = _compute_hhi(ref_scores) if len(ref_scores) > 0 else current_hhi

    # Alerts
    psi_alert = psi >= 0.20
    entropy_alert = entropy_ratio < 0.5
    concentration_alert = ref_hhi > 0 and current_hhi > 2 * ref_hhi

    # Overall status
    if psi_alert or entropy_alert:
        status = "critical"
    elif psi >= 0.10 or (ref_hhi > 0 and current_hhi > 1.5 * ref_hhi):
        status = "warning"
    else:
        status = "ok"

    return PredictionDiagnostics(
        psi=round(psi, 4),
        entropy=round(current_entropy, 4),
        entropy_ratio=round(entropy_ratio, 4),
        hhi=round(current_hhi, 6),
        n_valid=len(current_scores),
        psi_alert=psi_alert,
        entropy_alert=entropy_alert,
        concentration_alert=concentration_alert,
        overall_status=status,
    )
```

### Integration Point

In `predict.py` after step 8 (line ~222), before writing the parquet:

```python
from ashare_lab.research.monitor import compute_prediction_diagnostics

diag = compute_prediction_diagnostics(day.to_numpy(dtype=float))
if diag.overall_status == "critical":
    log.error(
        "PREDICTION QUALITY CRITICAL: PSI=%.3f entropy_ratio=%.2f "
        "HHI=%.4f -- consider halting trading",
        diag.psi, diag.entropy_ratio, diag.hhi,
    )
elif diag.overall_status == "warning":
    log.warning(
        "Prediction quality warning: PSI=%.3f entropy_ratio=%.2f",
        diag.psi, diag.entropy_ratio,
    )
```

Add to `meta.json` (line ~265):
```python
meta["diagnostics"] = {
    "psi": diag.psi,
    "entropy": diag.entropy,
    "entropy_ratio": diag.entropy_ratio,
    "hhi": diag.hhi,
    "status": diag.overall_status,
}
```

### Pipeline Integration

In `pipeline.py` `_step10_signal_generation()`, after reading meta.json
(line ~1242), add:

```python
diag = meta.get("diagnostics", {})
if diag.get("status") == "critical":
    logger.critical(
        "Signal quality CRITICAL for %s: PSI=%.3f -- "
        "TRADING HALTED until model review",
        ctx.trade_date, diag.get("psi", 0),
    )
    ctx.risk_result = replace(ctx.risk_result, buying_halted=True)
```

---

## Layer 2: Lagged IC Computation (T+5 Delayed Labels)

### Design

Compute Spearman rank correlation between T-day predictions and T+5 realized
returns. This is the standard Information Coefficient used in quantitative finance.
The 5-day lag matches the TRA model's natural holding period.

### Data Flow

```
Day T:  predictions/T.parquet  (instrument, score)
Day T+5: qlib $close for T and T+5  ->  5-day return per instrument
IC_T = SpearmanCorr(score_T, return_T_to_T+5)  across instruments
```

### Metrics

#### 2a. Lagged IC (per day)

```python
IC_T = spearman(scores_T, returns_T_to_T+5)
```

Computed cross-sectionally across all instruments with valid data.

#### 2b. Rolling IC (20-day window)

```python
rolling_IC = mean(IC_T for T in last 20 days)
```

#### 2c. ICIR (IC Information Ratio)

```python
ICIR = mean(IC) / std(IC)
```

Annualized: ICIR_annual = ICIR * sqrt(252) for daily signals.

#### 2d. IC Decay Detection

Track rolling IC over time. If rolling IC drops below zero for 10 consecutive
days, the signal has lost predictive power.

### Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Rolling IC (20d) | < 0.01 | < 0 (negative) |
| ICIR (annualized) | < 0.3 | < 0 |
| Consecutive negative IC | 5 days | 10 days |
| IC t-stat (HAC-adjusted) | < 1.5 | < 0 |

HAC-adjusted t-stat is critical because IC series has autocorrelation from
overlapping 5-day return windows. A naive t-stat of 2.0 may drop to 1.0
after HAC correction (Newey-West, lag=4).

### Implementation

File: `ashare_lab/research/lagged_ic.py` (new module)

```python
"""Lagged Information Coefficient computation.

Computes Spearman rank correlation between T-day predictions and
T+5 realized returns.  Writes results to data/ic_lagged.tsv.

Run daily from pipeline after T+5 returns are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ashare_lab.config import PREDICTIONS_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

LAG_DAYS = 5           # forward return horizon
ROLLING_WINDOW = 20    # rolling IC window
CONSECUTIVE_NEG = 10   # days of negative IC before alarm

IC_LAGGED_PATH = PROJECT_ROOT / "data" / "ic_lagged.tsv"


@dataclass
class LaggedICResult:
    """Result of lagged IC computation for one date."""
    trade_date: str
    lag_days: int
    ic: float | None        # Spearman IC for this date
    n_instruments: int      # instruments with valid data
    rolling_ic_20d: float   # 20-day rolling mean IC
    icir: float             # IC Information Ratio
    consecutive_neg: int    # consecutive days with IC < 0
    status: str             # "ok" | "warning" | "critical"


def compute_lagged_ic_for_date(
    trade_date: str,
    lag_days: int = LAG_DAYS,
) -> LaggedICResult:
    """Compute IC for predictions made lag_days ago vs realized returns.

    Args:
        trade_date: Today's date. Looks up predictions from
                    (trade_date - lag_days) trading days ago.
        lag_days: Forward return horizon (default 5).

    Returns:
        LaggedICResult with IC and diagnostics.
    """
    # Find prediction file from lag_days ago
    from ashare_lab.data.calendar import previous_trading_day
    import datetime as dt

    pred_date = dt.date.fromisoformat(trade_date)
    for _ in range(lag_days):
        pred_date = previous_trading_day(pred_date)
    pred_date_str = pred_date.isoformat()

    pred_path = PREDICTIONS_DIR / f"{pred_date_str}.parquet"
    if not pred_path.exists():
        return LaggedICResult(
            trade_date=trade_date, lag_days=lag_days,
            ic=None, n_instruments=0, rolling_ic_20d=0.0,
            icir=0.0, consecutive_neg=0, status="no_data",
        )

    # Load predictions
    pred_df = pd.read_parquet(pred_path)
    scores = pred_df.set_index("instrument")["score"]

    # Fetch realized returns from qlib
    instruments = scores.index.tolist()
    try:
        from qlib.data import D
        returns_df = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=pred_date_str,
            end_time=trade_date,
        )
        if returns_df is None or returns_df.empty:
            return LaggedICResult(
                trade_date=trade_date, lag_days=lag_days,
                ic=None, n_instruments=0, rolling_ic_20d=0.0,
                icir=0.0, consecutive_neg=0, status="no_data",
            )

        # Compute 5-day returns per instrument
        returns = {}
        for inst in returns_df.index.get_level_values(0).unique():
            inst_df = returns_df.loc[inst]
            if len(inst_df) >= 2:
                close_start = float(inst_df["$close"].iloc[0])
                close_end = float(inst_df["$close"].iloc[-1])
                if close_start > 0:
                    returns[str(inst)] = (close_end - close_start) / close_start

        returns_series = pd.Series(returns)

        # Align on common instruments
        common = scores.index.intersection(returns_series.index)
        if len(common) < 30:
            return LaggedICResult(
                trade_date=trade_date, lag_days=lag_days,
                ic=None, n_instruments=len(common),
                rolling_ic_20d=0.0, icir=0.0,
                consecutive_neg=0, status="insufficient_data",
            )

        # Spearman rank correlation
        ic, _ = stats.spearmanr(
            scores.loc[common].values,
            returns_series.loc[common].values,
        )

    except Exception as e:
        log.warning("Lagged IC computation failed: %s", e)
        return LaggedICResult(
            trade_date=trade_date, lag_days=lag_days,
            ic=None, n_instruments=0, rolling_ic_20d=0.0,
            icir=0.0, consecutive_neg=0, status="error",
        )

    # Write to history
    _append_lagged_ic(trade_date, ic, len(common))

    # Compute rolling stats from history
    history = _load_ic_history()
    recent = [h[1] for h in history[-ROLLING_WINDOW:]]
    rolling_ic = float(np.mean(recent)) if recent else 0.0
    icir = float(np.mean(recent) / np.std(recent)) if len(recent) > 1 and np.std(recent) > 0 else 0.0

    # Consecutive negative IC
    consec_neg = 0
    for _, val in reversed(history):
        if val is not None and val < 0:
            consec_neg += 1
        else:
            break

    # Status
    if consec_neg >= CONSECUTIVE_NEG or rolling_ic < 0:
        status = "critical"
    elif rolling_ic < 0.01 or icir < 0.3:
        status = "warning"
    else:
        status = "ok"

    return LaggedICResult(
        trade_date=trade_date, lag_days=lag_days,
        ic=round(ic, 4), n_instruments=len(common),
        rolling_ic_20d=round(rolling_ic, 4),
        icir=round(icir, 4),
        consecutive_neg=consec_neg,
        status=status,
    )


def _append_lagged_ic(trade_date: str, ic: float, n: int) -> None:
    """Append to data/ic_lagged.tsv (atomic write)."""
    IC_LAGGED_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if IC_LAGGED_PATH.exists():
        with IC_LAGGED_PATH.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    entries.append((parts[0], parts[1]))

    # Skip duplicate
    if entries and entries[-1][0] == trade_date:
        return

    entries.append((trade_date, f"{ic:.4f}" if ic is not None else ""))
    entries = entries[-60:]  # keep 60 days

    tmp = IC_LAGGED_PATH.with_suffix(".tsv.tmp")
    with tmp.open("w") as f:
        for d, v in entries:
            f.write(f"{d}\t{v}\n")
    tmp.rename(IC_LAGGED_PATH)


def _load_ic_history() -> list[tuple[str, float | None]]:
    """Load ic_lagged.tsv as list of (date, ic_value)."""
    if not IC_LAGGED_PATH.exists():
        return []
    entries = []
    with IC_LAGGED_PATH.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                d = parts[0]
                v = float(parts[1]) if parts[1] else None
                entries.append((d, v))
    return entries
```

### Pipeline Integration

Add a new step in `pipeline.py` after `_step10_signal_generation()`:

```python
def _step10b_lagged_ic_check(ctx: DailyRunContext) -> None:
    """Compute lagged IC (T+5) and check signal quality."""
    from ashare_lab.research.lagged_ic import compute_lagged_ic_for_date
    result = compute_lagged_ic_for_date(ctx.trade_date)

    if result.status == "critical":
        logger.critical(
            "Signal quality CRITICAL: rolling IC=%.4f (20d), "
            "ICIR=%.2f, consecutive negative=%d days -- "
            "TRADING HALTED, model retrain required",
            result.rolling_ic_20d, result.icir, result.consecutive_neg,
        )
        ctx.risk_result = replace(ctx.risk_result, buying_halted=True)
    elif result.status == "warning":
        logger.warning(
            "Signal quality warning: rolling IC=%.4f, ICIR=%.2f",
            result.rolling_ic_20d, result.icir,
        )

    # Store for report
    ctx.config["_lagged_ic"] = {
        "ic": result.ic,
        "rolling_ic": result.rolling_ic_20d,
        "icir": result.icir,
        "status": result.status,
    }
```

Call in `run_daily()` between step 10 and step 11:
```python
rc = _step10_signal_generation(ctx)
if rc == 2:
    return 2
_step10b_lagged_ic_check(ctx)  # NEW
_step11_ipo_processing(ctx)
```

---

## Layer 3: Meta-Labeling (Future Enhancement)

### Design

A secondary binary classifier that sits on top of the TRA primary model.
The meta-label model predicts whether a given TRA signal will be profitable
(triple-barrier method). It gates and sizes positions.

### Concept (Lopez de Prado, AFML 2018)

- Primary model (TRA): high recall, low precision -- fires broadly
- Meta-label model: binary classifier, learns which signals are true positives
- Output: P(meta-label=1) used as both gate and position size multiplier

### Features for Meta-Model

| Category | Features |
|----------|----------|
| Signal characteristics | score magnitude, score rank percentile, score z-score |
| Market regime | CSI1000 20-day volatility, 60-day trend, breadth (advance/decline) |
| Sector | sector momentum, sector concentration of top-K signals |
| Timing | day-of-week, days-since-model-train, market session (morning/afternoon) |

### Labeling (Triple-Barrier)

For each top-K signal on day T:
- Upper barrier: +5% from entry (take profit)
- Lower barrier: -3% from entry (stop loss)
- Vertical barrier: T+5 days (time expiry)

Meta-label = 1 if upper barrier hit first, 0 otherwise.

### Expected Impact

| Metric | Without Meta | With Meta |
|--------|-------------|-----------|
| Precision | ~52% (random top-K) | 60-70% |
| Sharpe | baseline | +30-50% |
| Max drawdown | baseline | -20-30% improvement |

### Effort

- Training data: needs 60+ days of live predictions with realized returns
- Model: LightGBM binary classifier (fast, interpretable)
- Validation: purged k-fold cross-validation (no leakage)
- Timeline: Phase 4+ (after live trading starts generating labeled data)

---

## Integration Plan Summary

### Phase 1: Layer 1 (1-2 days effort)

1. Create `ashare_lab/research/monitor.py` with PSI/entropy/HHI
2. Integrate into `predict.py` after score extraction
3. Add diagnostics to `meta.json`
4. Add critical-alert gate in `pipeline.py` step 10
5. Unit tests with known distributions

### Phase 2: Layer 2 (2-3 days effort)

1. Create `ashare_lab/research/lagged_ic.py`
2. Add `_step10b_lagged_ic_check()` to pipeline
3. Create `data/ic_lagged.tsv` tracking
4. Add HAC-adjusted t-stat for IC significance
5. Integration test with backfill data

### Phase 3: Layer 3 (1-2 weeks, after live data)

1. Collect 60+ days of predictions with realized returns
2. Label using triple-barrier method
3. Train LightGBM meta-classifier
4. Add meta-gate to signal generation
5. Backtest meta-labeling impact

---

## Risk Reduction Estimate

| Layer | Risk Reduction | Effort |
|-------|---------------|--------|
| Layer 1 (PSI/entropy) | 60% -- catches distribution drift, score collapse | 1-2 days |
| Layer 2 (Lagged IC) | 30% -- catches alpha decay, signal degradation | 2-3 days |
| Layer 3 (Meta-labeling) | 10% -- optimizes position sizing | 1-2 weeks |
| Combined | ~90% -- system can no longer trade on pure noise silently | 1 week + ongoing |

---

## Key Design Decisions

1. **Quantile-based PSI bins** (not equal-width): financial score distributions are
   heavy-tailed; equal-width bins concentrate all mass in one bin.

2. **5-day lag for IC** matches TRA's natural holding period (configurable).
   1-day IC would be noisier; 21-day would be too slow to detect decay.

3. **HAC-adjusted significance** is mandatory: IC series from overlapping 5-day
   windows has autocorrelation. Naive t-stats overstate significance by 1.5-2x.

4. **Critical threshold halts buying** (not selling): the pipeline already has
   `buying_halted` infrastructure from risk checks. Reuse it.

5. **No NannyML CBPE**: CBPE requires calibrated probabilities. TRA outputs
   raw scores, not probabilities. Calibrating TRA scores is non-trivial and
   adds complexity. PSI + lagged IC is simpler and more directly interpretable
   for this use case.

6. **No feature drift monitoring yet**: requires storing Alpha158 feature
   distributions (high-dimensional). PSI on prediction scores is a sufficient
   proxy -- if features drift, predictions drift. Feature-level monitoring is
   a Phase 5 enhancement.
