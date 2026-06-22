"""Walk-forward validation: permutation test and CPCV PBO.

Provides run_permutation_test() and compute_cpcv_pbo() for assessing
whether walk-forward excess returns are statistically significant or
likely the result of overfitting.

No qlib dependency. Operates on lists of per-window excess returns
produced by the walk-forward pipeline.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

log = logging.getLogger(__name__)

# Minimum number of windows required for meaningful statistical tests.
_MIN_WINDOWS = 4


def run_permutation_test(
    excesses: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for walk-forward excess return significance.

    Shuffles the window-ordering of excess returns n_permutations times
    and computes the mean of the running cumulative sum for each shuffle.
    The p-value is the fraction of permuted statistics that equal or
    exceed the observed statistic.

    The test answers: "is the observed walk-forward performance path
    unusually favorable compared to random window orderings?"

    The statistic (mean of cumulative sum) is path-dependent: windows
    with positive returns appearing early in the sequence produce a
    higher mean than the same returns appearing late. This avoids the
    degeneracy of compounded product (which is commutative and always
    yields p=1.0 under permutation).

    Args:
        excesses: Per-window cumulative excess returns (floats).
            Order matters for the observed statistic; permutations
            destroy ordering.
        n_permutations: Number of random shuffles. Default 1000 gives
            p-value resolution of 0.001.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys:
            observed: float, mean of the running cumulative sum.
            p_value: float in [0, 1], fraction of permutations >= observed.
            n_beats: int, count of permutations >= observed.
            perm_mean: float, mean of permuted statistics.
            perm_std: float, std of permuted statistics.
            n_permutations: int, as passed.
            verdict: str, "SIGNIFICANT" if p < 0.05, else "NOT_SIGNIFICANT".

    Raises:
        ValueError: When len(excesses) < _MIN_WINDOWS or contains NaN/Inf.
    """
    n = len(excesses)
    if n < _MIN_WINDOWS:
        raise ValueError(
            f"permutation test requires >= {_MIN_WINDOWS} windows, got {n}"
        )

    arr = np.array(excesses, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("excesses must not contain NaN or Inf values")

    # Observed statistic: mean of the running cumulative sum of excess
    # returns. Unlike compounded total (which is commutative under
    # permutation and always yields p=1.0), the running cumulative sum
    # is path-dependent -- favorable windows early in the sequence
    # produce a higher mean than favorable windows late. This tests
    # whether the walk-forward ordering matters for overall performance.
    cumsum = np.cumsum(arr)
    observed = float(np.mean(cumsum))

    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        shuffled = rng.permutation(arr)
        perm_stats[i] = np.mean(np.cumsum(shuffled))

    n_beats = int(np.sum(perm_stats >= observed))
    p_value = n_beats / n_permutations

    verdict = "SIGNIFICANT" if p_value < 0.05 else "NOT_SIGNIFICANT"

    log.info(
        "permutation: observed=%.4f, p=%.4f (%d/%d beats), verdict=%s",
        observed,
        p_value,
        n_beats,
        n_permutations,
        verdict,
    )

    return {
        "observed": observed,
        "p_value": p_value,
        "n_beats": n_beats,
        "perm_mean": float(np.mean(perm_stats)),
        "perm_std": float(np.std(perm_stats)),
        "n_permutations": n_permutations,
        "verdict": verdict,
    }


def compute_cpcv_pbo(
    excesses: list[float],
) -> dict:
    """Combinatorial purged cross-validation probability of backtest overfitting.

    Splits N windows into all possible (N choose N//2) train/test
    combinations. For each split, the "train" performance (mean excess)
    is compared to "test" performance. A split is counted as "overfit"
    when train mean > 0 but test mean <= 0 (in-sample looks good,
    out-of-sample does not).

    PBO = n_overfit / n_combinations. Higher PBO indicates greater
    overfitting risk.

    This is a simplified CPCV following Bailey et al. (2014) "The
    Probability of Backtest Overfitting" (Journal of Computational
    Finance), adapted for walk-forward windows instead of individual
    trials.

    Args:
        excesses: Per-window cumulative excess returns (floats).

    Returns:
        Dict with keys:
            pbo: float in [0, 1], probability of backtest overfitting.
            n_overfit: int, number of overfit combinations.
            n_combinations: int, total (N choose N//2) combinations.
            n_train_positive: int, combinations where train mean > 0.
            verdict: str, "LOW_RISK" if pbo < 0.50, else "HIGH_RISK".

    Raises:
        ValueError: When len(excesses) < _MIN_WINDOWS or contains NaN/Inf.
    """
    n = len(excesses)
    if n < _MIN_WINDOWS:
        raise ValueError(
            f"CPCV PBO requires >= {_MIN_WINDOWS} windows, got {n}"
        )

    arr = np.array(excesses, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("excesses must not contain NaN or Inf values")
    half = n // 2
    indices = list(range(n))

    n_overfit = 0
    n_train_positive = 0
    n_combinations = 0

    for train_idx in combinations(indices, half):
        train_set = set(train_idx)
        test_idx = [i for i in indices if i not in train_set]

        train_mean = float(np.mean(arr[list(train_idx)]))
        test_mean = float(np.mean(arr[test_idx]))

        n_combinations += 1

        if train_mean > 0:
            n_train_positive += 1
            if test_mean <= 0:
                n_overfit += 1

    pbo = n_overfit / n_combinations if n_combinations > 0 else 0.0
    verdict = "LOW_RISK" if pbo < 0.50 else "HIGH_RISK"

    log.info(
        "cpcv_pbo: pbo=%.4f (%d/%d overfit, %d train_positive), verdict=%s",
        pbo,
        n_overfit,
        n_combinations,
        n_train_positive,
        verdict,
    )

    return {
        "pbo": pbo,
        "n_overfit": n_overfit,
        "n_combinations": n_combinations,
        "n_train_positive": n_train_positive,
        "verdict": verdict,
    }
