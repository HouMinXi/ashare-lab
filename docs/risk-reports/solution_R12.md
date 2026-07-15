# R12: Blend Collapse -- Detection and Mitigation

## Problem

If the TRA model produces identical or near-identical scores for all stocks,
the pipeline passes every check because `min_signal_coverage` only counts
stocks with scores, not score variance or entropy.  Model collapse can happen
silently -- the neural network degenerates to a constant-output mode while
the inference path remains error-free.

**Current gap in predict.py (line 214-220):**  After blending and dropping
non-finite scores, the code checks `day.empty` but never inspects the
distribution of the surviving scores.

---

## Research Summary

### Key Sources

| Source | Year | Relevance |
|--------|------|-----------|
| Shumailov et al., "AI models collapse..." (Nature) | 2024 | Foundational: collapse = convergence to point estimate with near-zero variance |
| "When Alpha Breaks" (arXiv 2603.13252) | 2026-02 | Cross-sectional stock ranker regime failure; regime-trust gate G(t); per-stock uncertainty cannot detect uniform collapse (AUROC ~0.50) |
| ORCA (arXiv 2604.17251) | 2026-04 | Uses eigenvalue entropy + effective rank from spectral graph theory on correlation matrices |
| SIGMA (arXiv 2601.03385) | 2026-01 | Spectral framework for detecting model collapse via Gram matrix contraction |
| "Output Diversity Collapse in Post-Training" (arXiv 2604.16027) | 2026-04 | Diversity collapse is embedded in weights by training data composition |
| ODI Monitoring (IJHSR 2026) | 2026 | Output Diversity Index: type-token ratio + unique n-grams for behavioral collapse detection |
| MASSIF (Zenodo 2026) | 2026-06 | Hidden-state trajectory monitoring; persistence flip as collapse precursor |
| Spectral Alignment (arXiv 2510.04202) | 2025 | Sign-diversity collapse in weight matrices as early warning for training divergence |

### Key Insight from "When Alpha Breaks"

> Per-stock uncertainty cannot detect regime-level model failure: aggregated
> per-stock uncertainty achieves AUROC ~0.50 for predicting whether the model
> will have a positive-RankIC day -- no better than random. This is by design:
> the failure is a latent variable that affects all stocks uniformly.

This directly describes R12: if TRA collapses to identical scores, no
per-stock metric catches it.  We need a **cross-sectional diversity metric**
computed on the entire score vector for each date.

---

## Detection Metrics

### Tier 1: Always-On Gate (O(n), fail-fast)

These run on every prediction batch.  If any threshold is breached, the
pipeline raises `ValueError` and refuses to write the parquet file.

| Metric | Formula | Collapse Value | Threshold (FAIL) | Rationale |
|--------|---------|----------------|-------------------|-----------|
| `score_std` | `np.std(scores)` | 0.0 | < 0.001 | Zero variance = all identical.  Cheapest check. |
| `unique_ratio` | `len(set(round(s, 4))) / n` | 1/n | < 0.10 | At 4dp precision, <10% unique bins means near-identical outputs. |
| `iqr_range` | `Q3 - Q1` | 0.0 | < 0.005 | Robust to outliers; catches narrow-but-not-zero spread. |

### Tier 2: Diagnostic Logging (O(n log n), warn-only)

These are logged to `meta.json` for historical monitoring and alerting,
but do not block the pipeline by default (configurable).

| Metric | Formula | Collapse Value | Threshold (WARN) | Rationale |
|--------|---------|----------------|-------------------|-----------|
| `effective_rank` | `exp(H(p))` where `p_i = \|s_i\| / sum(\|s_j\|)` | 1.0 | < 1.5 | Measures intrinsic dimensionality.  1 = delta function.  From SIGMA/ORCA spectral methods. |
| `gini_coefficient` | Classic Gini on `abs(scores)` | 0.0 | < 0.05 | Inequality measure.  0 = perfect equality (all same).  Well-understood economics metric. |
| `entropy_normalized` | `H(p) / log(n)` | 0.0 | < 0.05 | Shannon entropy normalized to [0,1].  0 = degenerate, 1 = uniform. |

### Tier 3: Historical Drift (rolling window, alert-only)

Compare current batch metrics against a rolling baseline (e.g., 30-day
trailing median).  Alert if metrics drop by >50% from baseline.

| Signal | Implementation |
|--------|---------------|
| `std_ratio` | `current_std / rolling_median_std` < 0.5 |
| `rank_ratio` | `current_effective_rank / rolling_median_er` < 0.5 |

---

## Integration Point in predict.py

**Location:** Between step 8 (drop non-finite, line 214) and step 9 (build
output DataFrame, line 222).  This is after blending and filtering, before
writing.

```python
# -- 8b. Diversity gate (R12: blend collapse detection) ------------------
scores = day.to_numpy(dtype=float)
n = len(scores)

if n >= 10:  # skip check for tiny universes
    score_std = float(np.std(scores))
    unique_ratio = len(set(np.round(scores, 4))) / n
    q25, q75 = np.percentile(scores, [25, 75])
    iqr_range = float(q75 - q25)

    # Tier 1: fail-fast
    if score_std < 0.001:
        raise ValueError(
            "BLEND COLLAPSE DETECTED for %s: score_std=%.6f "
            "(threshold: 0.001). All scores are near-identical."
            % (trade_date, score_std)
        )
    if unique_ratio < 0.10:
        raise ValueError(
            "BLEND COLLAPSE DETECTED for %s: unique_ratio=%.3f "
            "(threshold: 0.10). Only %d unique bins out of %d."
            % (trade_date, unique_ratio,
               len(set(np.round(scores, 4))), n)
        )
    if iqr_range < 0.005:
        raise ValueError(
            "BLEND COLLAPSE DETECTED for %s: iqr_range=%.6f "
            "(threshold: 0.005). Score distribution is degenerate."
            % (trade_date, iqr_range)
        )

    # Tier 2: diagnostic (logged to meta.json, warn-only)
    abs_scores = np.abs(scores)
    p = abs_scores / abs_scores.sum()
    p = p[p > 0]  # avoid log(0)
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))
    entropy_norm = float(-np.sum(p * np.log(p)) / np.log(n))

    # Gini coefficient
    sorted_scores = np.sort(abs_scores)
    index = np.arange(1, n + 1)
    gini = float(
        (2 * np.sum(index * sorted_scores) / (n * np.sum(sorted_scores)))
        - (n + 1) / n
    )

    if effective_rank < 1.5:
        log.warning(
            "DIVERSITY WARN %s: effective_rank=%.2f (threshold: 1.5)",
            trade_date, effective_rank,
        )
else:
    score_std = unique_ratio = iqr_range = None
    effective_rank = entropy_norm = gini = None
```

**Meta.json additions (line 265-274):**

```python
meta = {
    # ... existing fields ...
    "diversity": {
        "score_std": score_std,
        "unique_ratio": unique_ratio,
        "iqr_range": iqr_range,
        "effective_rank": effective_rank,
        "entropy_normalized": entropy_norm,
        "gini": gini,
    },
}
```

---

## Alert Thresholds Summary

| Metric | FAIL (block pipeline) | WARN (log + alert) | Healthy Range |
|--------|-----------------------|--------------------| --------------|
| `score_std` | < 0.001 | < 0.01 | 0.02 - 0.15 |
| `unique_ratio` | < 0.10 | < 0.30 | 0.50 - 1.00 |
| `iqr_range` | < 0.005 | < 0.02 | 0.03 - 0.20 |
| `effective_rank` | -- | < 1.5 | 2.0 - 10.0 |
| `entropy_normalized` | -- | < 0.05 | 0.30 - 0.90 |
| `gini` | -- | < 0.05 | 0.15 - 0.60 |

**Notes on thresholds:**
- Tier 1 thresholds are deliberately conservative (only catch true collapse).
- Healthy ranges are estimates; tune on historical prediction data after
  running the pipeline for a few weeks.
- For a universe of ~4000 stocks, `effective_rank` of 2-3 is normal (scores
  are concentrated around the mean with long tails).  A value near 1.0 means
  near-identical scores.

---

## Configuration

Add to `ashare_lab/config.yaml`:

```yaml
diversity_gate:
  enabled: true
  fail_on_collapse: true       # Tier 1: raise ValueError
  warn_threshold: true         # Tier 2: log warning
  min_universe_size: 10        # skip check for tiny universes
  thresholds:
    score_std_min: 0.001
    unique_ratio_min: 0.10
    iqr_range_min: 0.005
    effective_rank_warn: 1.5
    entropy_warn: 0.05
    gini_warn: 0.05
```

---

## Effort Estimate

| Task | Time | Notes |
|------|------|-------|
| Implement `check_score_diversity()` in a new module `ashare_lab/research/diversity.py` | 30 min | Standalone function, testable in isolation |
| Integrate into `predict.py` between steps 8 and 9 | 20 min | ~20 lines of gate code + meta.json additions |
| Unit tests with synthetic collapse scenarios | 40 min | Test: all-identical, near-identical, normal, edge cases (n<10, single value) |
| Tune thresholds on historical predictions | 30 min | Run against existing `predictions/*.parquet` files |
| Add config entries | 10 min | Config schema + defaults |
| **Total** | **~2.5 hours** | |

---

## Testing Strategy

### Synthetic Collapse Scenarios

```python
def test_collapse_all_identical():
    """All scores the same -> FAIL."""
    scores = pd.Series([0.5] * 100)
    with pytest.raises(ValueError, match="BLEND COLLAPSE"):
        check_score_diversity(scores, "2026-01-01")

def test_collapse_near_identical():
    """Scores differ by < 0.001 -> FAIL on std."""
    scores = pd.Series([0.5 + i * 1e-6 for i in range(100)])
    with pytest.raises(ValueError, match="BLEND COLLAPSE"):
        check_score_diversity(scores, "2026-01-01")

def test_normal_diversity():
    """Normal score distribution -> passes."""
    np.random.seed(42)
    scores = pd.Series(np.random.randn(500) * 0.05 + 0.5)
    # Should not raise
    check_score_diversity(scores, "2026-01-01")

def test_small_universe_skipped():
    """Universe < min_universe_size -> skip check."""
    scores = pd.Series([0.5] * 5)
    # Should not raise (n < 10)
    check_score_diversity(scores, "2026-01-01")
```

### Bug-Injection Validation

Per CLAUDE.md rule: inject the bug (force identical scores), verify the gate
catches it (FAIL), revert, verify normal scores pass (PASS).

---

## Relationship to Existing Pipeline

```
predict_for_date()
  |
  +-- step 7: blend_tra_ntra()     <- produces blended scores
  +-- step 8: drop non-finite       <- filters NaN/inf
  +-- step 8b: diversity gate       <- NEW: R12 detection
  +-- step 9: build DataFrame
  +-- step 10: write parquet + meta
```

The diversity gate is **additive** -- it does not modify any existing logic.
If the gate fails, the parquet is never written, so downstream consumers
(BacktestEngine, report) never see a collapsed prediction file.
