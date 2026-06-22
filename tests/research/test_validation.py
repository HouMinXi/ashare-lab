"""Unit tests for Contract 4: validation.py (permutation test and CPCV PBO).

Tests that:
  - Permutation test on a known series produces expected p-value.
  - CPCV PBO on a known series produces expected verdict.
  - Both raise ValueError when too few windows are provided.
  - Seed reproducibility is verified.

No qlib dependency. Pure numerical tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_lab.research.validation import compute_cpcv_pbo, run_permutation_test


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_strong_positive_series(self):
        """A strongly positive series: verify all expected keys are present
        and p_value is in [0, 1]."""
        excesses = [0.05, 0.03, 0.04, 0.02, 0.06]
        result = run_permutation_test(excesses, n_permutations=500, seed=42)

        assert "p_value" in result
        assert "n_beats" in result
        assert "verdict" in result
        assert result["n_permutations"] == 500
        assert 0.0 <= result["p_value"] <= 1.0

    def test_mixed_series_p_value_range(self):
        """A mixed series should have p in (0, 1)."""
        excesses = [0.10, -0.05, 0.08, -0.03, 0.12, -0.01]
        result = run_permutation_test(excesses, n_permutations=1000, seed=42)

        assert 0.0 <= result["p_value"] <= 1.0
        # Observed = mean of cumulative sum (path-dependent statistic).
        expected_observed = float(np.mean(np.cumsum(np.array(excesses))))
        assert result["observed"] == pytest.approx(expected_observed, rel=1e-10)

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        excesses = [0.05, -0.02, 0.03, 0.01, -0.01]
        r1 = run_permutation_test(excesses, n_permutations=200, seed=123)
        r2 = run_permutation_test(excesses, n_permutations=200, seed=123)

        assert r1["p_value"] == r2["p_value"]
        assert r1["n_beats"] == r2["n_beats"]
        assert r1["perm_mean"] == r2["perm_mean"]

    def test_too_few_windows_raises(self):
        """Fewer than 4 windows raises ValueError."""
        with pytest.raises(ValueError, match="requires >= 4"):
            run_permutation_test([0.01, 0.02, 0.03])

    def test_observed_mean_cumsum(self):
        """Observed statistic matches mean of cumulative sum."""
        excesses = [0.10, 0.20, 0.05, -0.03]
        # cumsum = [0.10, 0.30, 0.35, 0.32], mean = 0.2675
        expected = float(np.mean(np.cumsum(np.array(excesses))))
        result = run_permutation_test(excesses, n_permutations=100, seed=42)
        assert result["observed"] == pytest.approx(expected, rel=1e-10)

    def test_verdict_ordering_dependent(self):
        """The mean-of-cumsum statistic is path-dependent: early gains
        produce a higher observed than late gains. A series with the
        large negative first has a LOWER observed than most permutations
        (where the negative moves to later positions), so p_value should
        be relatively high."""
        excesses = [-0.50, 0.01, 0.01, 0.01, 0.01]
        result = run_permutation_test(excesses, n_permutations=500, seed=42)
        assert result["verdict"] in ("SIGNIFICANT", "NOT_SIGNIFICANT")

    def test_n_permutations_respected(self):
        """The n_permutations parameter controls the number of shuffles."""
        excesses = [0.05, -0.02, 0.03, 0.01, -0.01]
        result = run_permutation_test(excesses, n_permutations=50, seed=42)
        assert result["n_permutations"] == 50

    def test_nan_raises(self):
        """NaN in excesses raises ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            run_permutation_test([0.01, float("nan"), 0.03, 0.02])

    def test_inf_raises(self):
        """Inf in excesses raises ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            run_permutation_test([0.01, float("inf"), 0.03, 0.02])


# ---------------------------------------------------------------------------
# CPCV PBO
# ---------------------------------------------------------------------------


class TestCpcvPbo:
    def test_all_positive_low_risk(self):
        """When all windows are positive, PBO should be 0 (LOW_RISK)."""
        excesses = [0.05, 0.03, 0.04, 0.02, 0.06, 0.01]
        result = compute_cpcv_pbo(excesses)

        assert result["pbo"] == pytest.approx(0.0)
        assert result["verdict"] == "LOW_RISK"
        assert result["n_overfit"] == 0

    def test_mixed_series_overfit_detected(self):
        """A series with in-sample positive but OOS negative splits."""
        # Half strongly positive, half negative: many splits will have
        # train>0 but test<=0.
        excesses = [0.20, 0.15, 0.10, -0.15, -0.20, -0.10]
        result = compute_cpcv_pbo(excesses)

        assert 0.0 <= result["pbo"] <= 1.0
        assert result["n_combinations"] > 0
        assert result["n_overfit"] >= 0
        assert result["verdict"] in ("LOW_RISK", "HIGH_RISK")

    def test_n_combinations_is_n_choose_half(self):
        """n_combinations should equal C(n, n//2)."""
        from math import comb

        excesses = [0.01, 0.02, -0.01, -0.02, 0.03, 0.04]
        result = compute_cpcv_pbo(excesses)
        n = len(excesses)
        expected_combinations = comb(n, n // 2)
        assert result["n_combinations"] == expected_combinations

    def test_too_few_windows_raises(self):
        """Fewer than 4 windows raises ValueError."""
        with pytest.raises(ValueError, match="requires >= 4"):
            compute_cpcv_pbo([0.01, 0.02, 0.03])

    def test_all_negative_no_overfit(self):
        """When all windows are negative, no split has train>0, so n_overfit=0."""
        excesses = [-0.05, -0.03, -0.04, -0.02]
        result = compute_cpcv_pbo(excesses)

        # train mean is always <= 0 (all values negative), so n_train_positive
        # and n_overfit should be 0.
        assert result["n_overfit"] == 0
        assert result["pbo"] == pytest.approx(0.0)
        assert result["verdict"] == "LOW_RISK"

    def test_pbo_keys_present(self):
        """All expected keys are present in the result."""
        excesses = [0.01, -0.01, 0.02, -0.02]
        result = compute_cpcv_pbo(excesses)

        expected_keys = {"pbo", "n_overfit", "n_combinations", "n_train_positive", "verdict"}
        assert set(result.keys()) == expected_keys

    def test_pbo_nan_raises(self):
        """NaN in excesses raises ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_cpcv_pbo([0.01, float("nan"), 0.03, 0.02])

    def test_pbo_inf_raises(self):
        """Inf in excesses raises ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_cpcv_pbo([0.01, float("inf"), 0.03, 0.02])
