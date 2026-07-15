# ashare-lab Model/Algorithm/Data Attack Analysis

**Date**: 2026-07-15
**Codebase**: ashare-lab @ /home/houminxi/code/ashare-lab
**Models analyzed**: TRA (Temporal Routing Adaptor, KDD 2021) on CSI1000, Alpha158 features

---

## Attack 1: Factor Crowding

**Scenario**: 1000 quant funds use the same Alpha158 factors. Alpha decays to zero overnight.

### Code Path Trace

```
train.py:204  cfg_handler = "alpha158"
train.py:205  HandlerClass = Alpha158
train.py:253  handler = HandlerClass(instruments=universe, ...)
train.py:305  dataset = MTSDatasetH(handler=handler, ...)
train.py:412  model = TRAModel(...)
train.py:451  model.fit(dataset)
```

**Finding**: The system uses qlib's standard Alpha158 handler -- 158 pre-engineered technical features (momentum, volatility, volume, price patterns). These are the same features published in the qlib documentation and used by every Chinese quant tutorial. [KNOWN]

**Failure mechanism**:
- Alpha158 features are publicly known. Any fund using qlib gets identical features.
- TRA's routing head learns 3 regime patterns (config: `routing.num_states: 3`). When 1000 funds trade the same signals, the patterns self-reinforce then collapse.
- The system has **zero detection** for factor crowding. No signal-to-noise ratio monitoring, no IC decay tracking over time, no turnover-adjusted IC.

**Corrupted output**:
```python
# predict.py:197-198
pred_df = model.predict(dataset, segment="test")
pred = pred_df["score"]
# Score is still "valid" (finite, ranked) -- but the rank has no predictive power.
# blend.py:94  blended = 0.60 * pred + 0.40 * pred_ntra
# Both components are noise; blend produces noise with a fancy label.
```

**Severity**: **HIGH**. No defense exists. The system trades on noise with confidence. No IC monitoring triggers an alert; `predict.py:235-258` tries to read IC from model internals but returns `None` for live predictions (no labels exist out-of-sample). The meta.json writes `ic: null` and nobody acts on it.

---

## Attack 2: Regime Inversion

**Scenario**: Value factor inverts after 3 years. Model learned wrong relationship.

### Code Path Trace

```
train.py:253  handler = HandlerClass(start_time="2017-01-01",
              fit_start_time=window["train_start"],   # e.g. "2021-01-01"
              fit_end_time=window["train_end"])        # e.g. "2023-12-31"
train.py:305  dataset = MTSDatasetH(segments=segs)
train.py:412  model = TRAModel(n_epochs=200, early_stop=30)
train.py:451  model.fit(dataset)
```

**Finding**: The walk-forward window is 3 years (`config: train_window_years: 3`), stepping 6 months. TRA learns 3 routing states (`routing.num_states: 3`) to capture regime patterns. [KNOWN]

**Failure mechanism**:
- If the market regime inverts (e.g., growth-to-value rotation), the 3-year training window contains the OLD regime as the dominant pattern.
- TRA's `early_stop: 30` patience on valid IC means it locks in the old regime's patterns before seeing the inversion.
- The blend at `blend.py:94` (`0.60 * raw + 0.40 * neutralized`) amplifies the problem: neutralization regresses against Size/Volatility/Momentum (`neutralize.py:59-63`), but if the STYLE FACTORS themselves invert (e.g., small-cap premium reverses), neutralization makes predictions WORSE.

**Corrupted output**:
```python
# neutralize.py:183-190
x_mat = np.column_stack([
    np.ones(len(pred_valid)),           # intercept
    factors_valid["size"].values,       # inverted factor!
    factors_valid["volatility"].values, # inverted factor!
    factors_valid["momentum"].values,   # inverted factor!
])
beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
residuals = y - fitted  # residuals now CORRELATED with the inverted factor
```

**Severity**: **HIGH**. The 60/40 blend was locked at Phase 2.1 (`blend.py:7-9`). There is no mechanism to detect regime inversion, no PSI (Population Stability Index) monitoring on feature distributions, and no adaptive blend weight. The system will confidently lose money for months before the walk-forward window catches up.

---

## Attack 3: Data Poisoning (Bad Ticks)

**Scenario**: qlib data has bad tick (price = 0 or 99999). Does the system catch it?

### Code Path Trace

```
train.py:253  handler = HandlerClass(instruments=universe,
              start_time="2017-01-01", end_time=window["test_end"])
              # Alpha158 computes features from raw OHLCV via D.features
train.py:305  dataset = MTSDatasetH(handler=handler, ...)
train.py:451  model.fit(dataset)  # trains on poisoned features
```

**Finding**: The data validation layer (`data/validator.py`) runs on FETCHED tushare data BEFORE qlib dump, not on the qlib binary files that train.py reads. [KNOWN]

**Defense analysis**:
- `validator.py:103-143`: checks change limits, volume=0, close outside [low, high]. These catch SOME bad ticks but only on the tushare DataFrame, NOT on the qlib binary `.day.bin` files.
- `validate.py:95-158`: cross-validates qlib vs baostock for 3 symbols (`DEFAULT_SYMBOLS = ["SH600000", "SZ000001", "SH601318"]`). This covers 3 out of ~2597 CSI1000 stocks.
- `train.py:230`: `RobustZScoreNorm` with `clip_outlier=True` clips extreme features during training. But this clips to the TRAINING distribution -- a poisoned tick that's within the training clip range passes silently.
- `predict.py:214`: `day = day[np.isfinite(day.to_numpy(dtype=float))]` catches `inf`/`NaN` predictions but NOT finite-but-wrong predictions from poisoned inputs.

**Corrupted output**:
```
# Poisoned tick: SH600xxx close = 0.01 (data corruption)
# Alpha158 computes: $close/Ref($close,5) = 0.01/12.50 = 0.0008
# RobustZScoreNorm clips this to ~-3.5 (within clip range)
# TRA learns: "this stock crashed 99.9%" -> strong sell signal
# Result: model shorts a stock that didn't actually move
```

**Severity**: **MEDIUM**. The qlib data pipeline has a validation gap: tushare data is validated, but the qlib binary files (which train.py actually reads) are never re-validated. A corrupted `.day.bin` file passes all checks. The RobustZScoreNorm clip_outlier provides partial defense for extreme values but not for subtle corruption (e.g., close off by 10%).

---

## Attack 4: Overfitting

**Scenario**: Model memorized training data, out-of-sample IC = 0.

### Code Path Trace

```
train.py:253  handler = HandlerClass(
              fit_start_time=window["train_start"],  # 3 years
              fit_end_time=window["train_end"])
train.py:305  dataset = MTSDatasetH(segs, seq_len=20, num_states=3)
train.py:412  model = TRAModel(n_epochs=200, early_stop=30)
train.py:451  model.fit(dataset)
train.py:537  pred_df = model.predict(dataset, segment="test")
train.py:545  pred = apply_price_filter(pred, ...)  # only filter, no IC check
```

**Finding**: TRA has 158 input features, 64 hidden units, 2 GRU layers, 3 routing states. The model has ~50K+ parameters. Training window is ~750 trading days x ~2597 stocks = ~1.9M samples. This ratio is reasonable, BUT: [COMPUTED]

**Failure mechanism**:
- `early_stop: 30` patience on validation IC. If the validation set is correlated with training (adjacent 6-month window), early stopping doesn't prevent overfitting -- it just stops at the point of maximum train-valid correlation.
- `train.py:545`: `apply_price_filter` is the ONLY post-prediction check. It filters by close price threshold (`exclude_close_above_cny: 300`), NOT by prediction quality.
- `train.py:537`: TRA predictions on the test set are used directly. No out-of-sample IC computation, no comparison against a null model, no bootstrap confidence interval.

**Corrupted output**:
```
# Backtest shows IC=0.040 (good on test set, which is the NEXT 6-month window)
# But live prediction IC = 0.00 (model memorized training patterns)
# predict.py:235-258 tries to read IC from model._ic_list
# But _ic_list contains TRAINING IC, not live IC
# meta.json: {"ic": 0.040} -- MISLEADING: this is stale training IC
```

**Severity**: **HIGH**. The system has no mechanism to distinguish "model learned real patterns" from "model memorized training data." The IC reported in meta.json comes from training internals, not live performance. The gate system (`config: min_mean_rank_ic: 0.02`) only runs during backtesting, not during live prediction.

---

## Attack 5: Neutralization Failure

**Scenario**: Style factors break down during crisis.

### Code Path Trace

```
blend.py:89   factors = compute_style_factors(instruments, start, end)
neutralize.py:66  raw = D.features(instruments, fields=[
                  "$close", "$volume",
                  "Std($close, 20)/Mean($close, 20)",
                  "Ref($close, -20)/$close - 1"])
neutralize.py:87  size_raw = np.log(close * volume)
neutralize.py:101-107  grouped = factors.groupby(level="datetime")
                      means = grouped.transform("mean")
                      stds = grouped.transform("std")
                      stds = stds.replace(0.0, np.nan)
                      factors = (factors - means) / stds
neutralize.py:190  beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
```

**Finding**: The neutralization uses 3 factors: size (`log(close*volume)`), volatility (`Std/Mean over 20d`), and momentum (`20d return`). Cross-sectional OLS per date. [KNOWN]

**Failure mechanism**:
- **Crisis day**: All stocks drop 8-10%. Volume spikes 3-5x. The `size` factor (`log(close*volume)`) shifts massively because volume explodes. The z-score normalization (`neutralize.py:101-107`) uses that day's cross-section, so the mean and std are computed from crisis-day data. This is correct in principle.
- **But**: `np.linalg.lstsq` with 3 factors + intercept needs >= 30 observations (`_MIN_INSTRUMENTS = 30`). On crisis days with trading halts, many stocks are suspended. If fewer than 30 stocks have valid data, neutralization SKIPS and returns raw predictions (`neutralize.py:172-179`).
- **Worse**: The OLS regression assumes homoscedasticity. During crisis, volatility clusters -- the residuals are NOT normally distributed. The neutralization removes the WRONG component of variance.

**Corrupted output**:
```python
# Crisis day: 200 stocks halted, 2397 active
# neutralize.py:172:  len(pred_valid) >= 30 -> runs OLS
# But the 2397 active stocks have correlated residuals (crisis = systematic)
# OLS beta_size is estimated from a biased sample (surviving stocks are large-caps)
# Residuals = y - X @ beta  -- but beta is wrong for small-caps
# blend.py:94: 0.60 * raw + 0.40 * wrong_residual
# Result: small-cap predictions are over-neutralized, large-cap under-neutralized
```

**Severity**: **MEDIUM-HIGH**. The `_MIN_INSTRUMENTS = 30` threshold is too low for reliable OLS during crisis. The system has no heteroscedasticity check, no factor stability monitoring, and no fallback to raw predictions when neutralization quality degrades (it either runs or skips, never "runs badly").

---

## Attack 6: Model Staleness

**Scenario**: 23.9 days old model, IC decayed to noise. System trades on noise.

### Code Path Trace

```
predict.py:26   _MODEL_STALE_DAYS = 7
predict.py:118  model_age_days = (time.time() - model_path.stat().st_mtime) / 86400
predict.py:119  if model_age_days > _MODEL_STALE_DAYS:
predict.py:120      log.warning("MODEL STALE: ...")
# ... continues to predict anyway ...
predict.py:189  model = torch.load(str(model_path), weights_only=False)
predict.py:197  pred_df = model.predict(dataset, segment="test")
predict.py:203  blended = blend_tra_ntra(pred, window)
predict.py:263  df.to_parquet(out, index=False)  # WRITES PREDICTION FILE
```

**Finding**: The staleness check at `predict.py:119` is WARNING-ONLY. It logs a warning but does NOT stop the prediction pipeline. [KNOWN]

**Failure mechanism**:
- Model is 23.9 days old (w10.pt trained 2026-07-01). The `_MODEL_STALE_DAYS = 7` threshold is exceeded, but the system continues.
- The model was trained on data up to `train_end` (e.g., 2023-12-31 for the last walk-forward window). Live predictions use `latest.pt` (`predict.py:90`), which points to the most recently trained window.
- The Alpha158 features are computed with `fit_start_time` and `fit_end_time` from the training window (`predict.py:148-149`). This means the z-score normalization parameters are FROZEN at training time. As the live market drifts, the normalized features shift out of the training distribution.
- No PSI monitoring exists. No feature drift detection. No prediction distribution monitoring.

**Corrupted output**:
```
# predict.py:265-275 meta.json:
{
  "model": "w10.pt",
  "model_age_days": 23.9,   # triggers warning at line 120
  "ic": null,                # always null for live predictions (no labels)
  "n_instruments": 2487      # looks normal!
}
# The paper engine reads this parquet and trades on it.
# No circuit breaker. No "model too old, skip today" logic.
```

**Severity**: **CRITICAL**. The system will trade on a 23.9-day-old model with no IC feedback, no feature drift detection, and no circuit breaker. The staleness warning is logged but not acted upon. The paper engine (`paper/pipeline.py`) reads the prediction file and executes trades regardless.

---

## Attack 7: Blend Ratio Failure

**Scenario**: 60/40 fixed ratio. What if TRA is completely wrong?

### Code Path Trace

```
blend.py:33   tra_weight: float = 0.60
blend.py:94   blended = tra_weight * pred.loc[idx] + (1 - tra_weight) * pred_ntra.loc[idx]
predict.py:203  blended = blend_tra_ntra(pred, window)
predict.py:214  day = day[np.isfinite(day.to_numpy(dtype=float))]
```

**Finding**: The blend ratio is hardcoded at 0.60/0.40. The docstring says "Phase 2.1 lock" and "never exposed to config or CLI" (`blend.py:7-9`). [KNOWN]

**Failure mechanism**:
- If TRA produces garbage predictions (all random, or all identical), the blend is 60% garbage + 40% neutralized-garbage.
- `neutralize.py:190`: OLS residuals from garbage predictions are also garbage (the regression fits noise).
- `predict.py:214`: `np.isfinite` check catches `NaN`/`inf` but NOT "all scores are 0.001" or "all scores are identical."
- No prediction diversity check. No "are these predictions actually different from random?" test.

**Corrupted output**:
```python
# TRA model collapsed: all predictions = 0.5 (constant)
# neutralize.py: y = [0.5, 0.5, ..., 0.5] (2487 identical values)
# OLS: beta = [0.5, 0, 0, 0] (intercept only, all factor betas = 0)
# residuals = y - X @ beta = [0, 0, ..., 0]
# blend.py:94: 0.60 * 0.5 + 0.40 * 0.0 = 0.30 for all stocks
# predict.py:214: all finite -> passes filter
# Result: paper engine gets 2487 stocks all with score=0.30
# Top-K selection is RANDOM (tie-breaking by instrument code)
```

**Severity**: **HIGH**. A model collapse (all predictions identical) passes every check in the pipeline. The only defense is the `min_signal_coverage: 100` check in `paper/signal.py:159`, which only verifies COUNT, not QUALITY. There is no prediction entropy check, no "scores must have non-zero variance" assertion, and no comparison against a random baseline.

---

## Summary Matrix

| # | Attack | Severity | Defense Exists? | Detection Latency |
|---|--------|----------|-----------------|-------------------|
| 1 | Factor Crowding | HIGH | No | Months (walk-forward lag) |
| 2 | Regime Inversion | HIGH | No | Months (walk-forward lag) |
| 3 | Data Poisoning | MEDIUM | Partial (tushare only) | Never (qlib binary gap) |
| 4 | Overfitting | HIGH | No | Never (IC=stale training) |
| 5 | Neutralization Failure | MEDIUM-HIGH | Partial (min 30 inst) | Days (crisis-driven) |
| 6 | Model Staleness | CRITICAL | Warning only | Never (no circuit breaker) |
| 7 | Blend Ratio Failure | HIGH | No | Never (no quality check) |

---

## Critical Gaps (Ordered by Impact)

1. **No live IC monitoring**. `predict.py:235-258` tries to read IC from model internals but returns `None` for live predictions. The system has NO way to know if predictions are useful. [KNOWN from code trace]

2. **No circuit breaker on stale models**. `predict.py:119-124` warns but continues. The paper engine trades regardless. [KNOWN from code trace]

3. **No prediction quality check**. Identical scores, zero variance, random rankings -- all pass the pipeline. `min_signal_coverage` checks count, not quality. [KNOWN from code trace]

4. **No feature drift detection**. Alpha158 z-scores are frozen at training time. No PSI, no KL-divergence, no distribution monitoring. [INFERRED from absence of code]

5. **No data re-validation on qlib binaries**. `validator.py` runs on tushare DataFrames, not on the `.day.bin` files that `train.py` reads. [KNOWN from code trace]

6. **No adaptive blend weight**. 60/40 is hardcoded. If one component degrades, the blend degrades proportionally with no feedback loop. [KNOWN from code trace]

7. **Neutralization assumes homoscedasticity**. Crisis-day OLS produces biased residuals. No robust regression, no heteroscedasticity test. [KNOWN from code trace]

---

## Recommended Mitigations (Priority Order)

1. **Live IC proxy**: Compute daily rank correlation between today's predictions and next-day returns (delayed by 1 day). Alert if 5-day rolling IC < 0.01.

2. **Staleness circuit breaker**: If model_age_days > threshold AND live IC proxy < noise floor, skip prediction and log CRITICAL.

3. **Prediction sanity check**: Assert `pred.std() > epsilon` and `pred.nunique() > min_coverage * 0.5` before writing parquet.

4. **PSI monitoring**: Compute PSI of Alpha158 feature distributions weekly vs training baseline. Alert if PSI > 0.25.

5. **Adaptive blend**: Replace fixed 0.60/0.40 with rolling IC-weighted blend. If TRA IC < neutral IC over trailing 20 days, shift weight toward neutral.

6. **Robust neutralization**: Replace OLS with WLS or Huber regression during crisis detection (e.g., when market return < -3%).

7. **Data pipeline hardening**: Re-validate qlib binary files against baostock before training, not just at fetch time.
