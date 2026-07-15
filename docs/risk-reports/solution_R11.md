# R11 Solution: GPU Down, Stale Predictions Cutoff

## Problem Statement

`ashare-pipeline.sh` lines 208-221: when GPU inference fails after all retries, the
stale fallback picks the most recent `.parquet` file with **zero age check**:

```bash
# Current code (line 211) -- no cutoff
LATEST_PRED=$(find "$PREDICTIONS_DIR" -name '*.parquet' 2>/dev/null | sort | tail -1)
```

A 90-day-old prediction file would be used as if it were fresh. The pipeline sets
`ASHARE_USE_STALE=1` and sends an alert, but the downstream risk system has no
mechanism to scale or halt positions based on prediction age.

## Research Findings

### Industry Practice (PMTS, StockAlpha, AlgoKing)

1. **PMTS retraining pipeline** (2026-06): "no model goes stale past a fixed
   horizon" -- uses dual-clock: scheduled cadence + event-driven drift detection.
   Retraining triggered when predicted vs realized divergence crosses threshold.

2. **StockAlpha concept drift alarms** (2026-02): Staged operational responses:
   (1) alert, (2) reduce position size, (3) stop new entries, (4) quarantine,
   (5) rollback. Common rolling windows: 30/60/90 trading days for performance
   metrics, 20-60 days for distributional checks.

3. **AlgoKing signal decay** (2026-03): Measured 76% alpha capture at 78ms latency.
   Decay rate depends on market type (equities slower than crypto) and strategy
   type (momentum decays faster than mean reversion). Key insight: decay is
   measurable and should factor into position sizing.

### Synthesis for A-Share Daily Predictions

- **Daily predictions have a 1-day shelf life.** A prediction for day T is
  calibrated on features observed through day T-1 close. By day T+2, the market
  has absorbed 2 sessions of information the model never saw.
- **Weekend/holiday compounding.** A Friday prediction used on Monday is already
  3 calendar days stale. A holiday-week prediction could be 5-7 days stale.
- **Regime sensitivity.** A-share markets are retail-heavy and momentum-driven;
  concept drift happens faster than in institutional US equities.

## Solution Design

### 1. Staleness Threshold (hard cutoff + soft tiers)

```
MAX_STALE_DAYS=3   # configurable, default 3 trading days
```

| Age (trading days) | Tier          | Action                          |
|--------------------|---------------|---------------------------------|
| 0                  | FRESH         | Full position sizing            |
| 1-2                | STALE_WARN    | 50% position sizing, alert      |
| 3+                 | STALE_REJECT  | Hard halt: no new buys, exit    |

**Rationale:** 3 trading days is the tightest reasonable cutoff. Beyond that, the
prediction is calibrated on a market state that has had 3+ sessions to drift.
Industry consensus (StockAlpha): "reduce new allocations within 24 hours" for
confirmed signal degradation.

### 2. Implementation in ashare-pipeline.sh

Replace lines 208-221 with:

```bash
# Stale fallback: ONLY after all GPU retries exhausted
if [ $GPU_RC -ne 0 ]; then
    USE_STALE=1
    MAX_STALE_SECONDS=$((MAX_STALE_DAYS * 86400))  # conservative: calendar days

    # Find newest parquet AND check its age
    LATEST_PRED=""
    LATEST_AGE=999999
    while IFS= read -r f; do
        file_ts=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        age=$(( $(date +%s) - file_ts ))
        if [ "$age" -lt "$LATEST_AGE" ]; then
            LATEST_AGE=$age
            LATEST_PRED="$f"
        fi
    done < <(find "$PREDICTIONS_DIR" -name '*.parquet' 2>/dev/null)

    if [ -z "$LATEST_PRED" ]; then
        echo "ERROR: GPU failed and no stale predictions available"
        python3 "$REPO/scripts/alert.py" "1" "gpu_inference" \
            "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
        exit 1
    fi

    # Compute staleness tier
    STALE_DAYS=$(( LATEST_AGE / 86400 ))
    if [ "$LATEST_AGE" -gt "$MAX_STALE_SECONDS" ]; then
        echo "ERROR: newest prediction is ${STALE_DAYS}d old, exceeds ${MAX_STALE_DAYS}d cutoff"
        python3 "$REPO/scripts/alert.py" "1" "stale_rejected" \
            "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
        exit 1
    fi

    PRED_ARG="--pred-path $LATEST_PRED"
    echo "WARNING: using ${STALE_DAYS}d-old prediction ($LATEST_PRED)"

    # RH1: alert on GPU failure BEFORE running pipeline (D-D19: stale never silent)
    python3 "$REPO/scripts/alert.py" "1" "gpu_inference" \
        "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
fi
```

### 3. Graceful Degradation in risk.py

Add an 8th risk dimension to `ashare_lab/paper/risk.py`:

```python
# New dimension 8: Prediction staleness scaling
def check_prediction_staleness(
    pred_age_days: int,
    stale_warn_threshold: int = 2,
    stale_reject_threshold: int = 3,
) -> tuple[float, bool]:
    """Scale position size by prediction freshness.

    Returns:
        (size_multiplier, halt_new_buys):
        - size_multiplier: 1.0 for fresh, 0.5 for warn, 0.0 for reject
        - halt_new_buys: True if age >= reject_threshold
    """
    if pred_age_days >= stale_reject_threshold:
        return 0.0, True
    elif pred_age_days > stale_warn_threshold:
        return 0.5, False
    return 1.0, False
```

Integration points in the existing `RiskCheckResult`:
- `buying_halted` already exists -- stale_reject sets it True
- New field `position_scale_factor: float` (default 1.0) -- pipeline multiplies
  all target quantities by this factor before order generation

### 4. Communicating Staleness Downstream

Pass staleness metadata through the pipeline:

```bash
# In ashare-pipeline.sh, export for pipeline.py
export ASHARE_PRED_AGE_DAYS=$STALE_DAYS
export ASHARE_STALE_TIER="FRESH|STALE_WARN|STALE_REJECT"
```

In `pipeline.py` / `cli/paper.py`:
- Read `ASHARE_PRED_AGE_DAYS` from environment
- Pass to `run_all_risk_checks()` which includes staleness check
- Log the tier in the daily summary for audit trail

### 5. Meta.json Enhancement

Each prediction's `.meta.json` should record generation timestamp:

```json
{
    "trade_date": "2026-07-15",
    "generated_at": "2026-07-15T18:30:00+08:00",
    "model_version": "v4",
    "gpu_host": "192.168.100.11"
}
```

The pipeline reads `generated_at` from meta.json instead of relying on file
mtime (which can be wrong after scp or filesystem operations).

## Files to Modify

| File | Change | Type |
|------|--------|------|
| `scripts/ashare-pipeline.sh` | Add staleness cutoff in stale fallback block | Logic |
| `ashare_lab/paper/risk.py` | Add `check_prediction_staleness()` + `position_scale_factor` | Logic |
| `tests/paper/test_risk.py` | Tests for staleness dimension | Test |
| `ashare_lab/cli/paper.py` | Read ASHARE_PRED_AGE_DAYS, pass to risk checks | Logic |

## Estimated Effort

- **Implementation:** 2-3 hours (pipeline.sh change + risk.py dimension + tests)
- **Review:** 3-cycle standard (logic-bearing code)
- **Testing:** Bug-inject verify: set MAX_STALE_DAYS=0, confirm pipeline rejects
  any stale fallback; set to 99, confirm it accepts
- **Total:** ~4 hours including review

## Priority Justification

This is a **correctness** fix, not a feature. Without it, a GPU outage that lasts
longer than expected silently degrades prediction quality with no signal to the
risk layer. The fix is small, well-scoped, and eliminates a class of silent
failures.
