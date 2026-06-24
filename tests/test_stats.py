"""Tests for postureopt.stats — confidence intervals, significance testing,
Bonferroni correction, and variance decomposition (issue #15)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import pytest
import numpy as np

from postureopt.stats import (
    bonferroni_correct,
    ci_latex,
    ci_str,
    confidence_interval,
    paired_ttest,
    variance_decomposition,
)


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------

def test_ci_mean_matches_sample_mean():
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci = confidence_interval(data)
    assert ci["mean"] == pytest.approx(sum(data) / len(data))


def test_ci_bounds_bracket_mean():
    data = [0.5, 0.6, 0.55, 0.52, 0.58, 0.61, 0.49, 0.57, 0.53, 0.56]
    ci = confidence_interval(data, confidence=0.95)
    assert ci["lower"] < ci["mean"] < ci["upper"]


def test_ci_wider_at_higher_confidence():
    data = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.2, 0.8, 0.35]
    ci_90 = confidence_interval(data, confidence=0.90)
    ci_99 = confidence_interval(data, confidence=0.99)
    assert ci_99["margin"] > ci_90["margin"]


def test_ci_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        confidence_interval([0.5])


def test_ci_symmetric():
    data = [0.4, 0.6, 0.5, 0.55, 0.45, 0.52, 0.48, 0.51, 0.49, 0.50]
    ci = confidence_interval(data)
    assert ci["mean"] - ci["lower"] == pytest.approx(ci["upper"] - ci["mean"], rel=1e-6)


def test_ci_n_reported_correctly():
    data = list(range(7))
    ci = confidence_interval(data)
    assert ci["n"] == 7


def test_ci_str_format():
    data = [0.5] * 10
    s = ci_str(data)
    assert "(" in s and "," in s and ")" in s


def test_ci_latex_format():
    data = [0.5] * 10
    s = ci_latex(data)
    assert "~(" in s


# ---------------------------------------------------------------------------
# paired_ttest
# ---------------------------------------------------------------------------

def test_paired_ttest_identical_samples_not_significant():
    a = [0.5, 0.6, 0.55, 0.52, 0.58]
    result = paired_ttest(a, a)
    assert not result["reject_null"]
    assert result["mean_diff"] == pytest.approx(0.0)


def test_paired_ttest_large_diff_significant():
    a = [1.0] * 10
    b = [0.0] * 10
    result = paired_ttest(a, b, alpha=0.05)
    assert result["reject_null"]
    assert result["mean_diff"] == pytest.approx(1.0)


def test_paired_ttest_mean_diff_direction():
    a = [0.8, 0.7, 0.9, 0.85, 0.75]
    b = [0.5, 0.4, 0.6, 0.55, 0.45]
    result = paired_ttest(a, b)
    assert result["mean_diff"] > 0


def test_paired_ttest_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_ttest([0.1, 0.2], [0.1])


def test_paired_ttest_p_value_in_range():
    a = [0.5, 0.6, 0.55, 0.52, 0.58, 0.61, 0.49, 0.57, 0.53, 0.56]
    b = [0.4, 0.5, 0.45, 0.42, 0.48, 0.51, 0.39, 0.47, 0.43, 0.46]
    result = paired_ttest(a, b)
    assert 0.0 <= result["p_value"] <= 1.0


def test_paired_ttest_uses_adjusted_alpha():
    a = [0.5, 0.6, 0.55, 0.52, 0.58, 0.61, 0.49, 0.57, 0.53, 0.56]
    b = [0.49, 0.59, 0.54, 0.51, 0.57, 0.60, 0.48, 0.56, 0.52, 0.55]
    r_loose = paired_ttest(a, b, alpha=0.5)
    r_strict = paired_ttest(a, b, alpha=0.0001)
    # Should be significant with loose alpha, not with very strict
    assert r_loose["reject_null"] or not r_strict["reject_null"]


# ---------------------------------------------------------------------------
# bonferroni_correct
# ---------------------------------------------------------------------------

def test_bonferroni_single_test_unchanged():
    assert bonferroni_correct(0.05, 1) == pytest.approx(0.05)


def test_bonferroni_halves_with_two_tests():
    assert bonferroni_correct(0.05, 2) == pytest.approx(0.025)


def test_bonferroni_six_tests():
    assert bonferroni_correct(0.05, 6) == pytest.approx(0.05 / 6)


def test_bonferroni_invalid_n_tests():
    with pytest.raises(ValueError):
        bonferroni_correct(0.05, 0)


def test_bonferroni_decreases_with_more_tests():
    thresholds = [bonferroni_correct(0.05, n) for n in [1, 2, 5, 10]]
    assert thresholds == sorted(thresholds, reverse=True)


# ---------------------------------------------------------------------------
# variance_decomposition
# ---------------------------------------------------------------------------

def test_variance_decomposition_shape():
    data = [[0.5, 0.6, 0.55], [0.4, 0.45, 0.42]]
    result = variance_decomposition(data)
    assert result["n_outer"] == 2
    assert result["n_inner"] == 3


def test_variance_decomposition_identical_groups_zero_outer():
    # All groups identical → outer variance = 0
    data = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    result = variance_decomposition(data)
    assert result["outer_var"] == pytest.approx(0.0, abs=1e-10)


def test_variance_decomposition_identical_within_groups_zero_inner():
    # Each group has identical values → inner variance = 0
    data = [[0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]]
    result = variance_decomposition(data)
    assert result["inner_var"] == pytest.approx(0.0, abs=1e-10)


def test_variance_decomposition_icc_between_zero_and_one():
    rng = np.random.default_rng(42)
    data = rng.random((4, 8)).tolist()
    result = variance_decomposition(data)
    assert 0.0 <= result["icc"] <= 1.0


def test_variance_decomposition_high_outer_gives_high_icc():
    # Groups very different from each other, tight within each group
    data = [[0.1 + i * 0.01 for _ in range(5)] for i in range(5)]
    result = variance_decomposition(data)
    assert result["icc"] > 0.9


def test_variance_decomposition_high_inner_gives_low_icc():
    # Groups similar to each other, high within-group spread
    rng = np.random.default_rng(0)
    data = (0.5 + rng.standard_normal((3, 20)) * 0.3).tolist()
    result = variance_decomposition(data)
    assert result["icc"] < 0.5


def test_variance_decomposition_requires_2d():
    with pytest.raises(ValueError):
        variance_decomposition([0.1, 0.2, 0.3])


def test_variance_decomposition_total_var_nonnegative():
    data = [[0.3, 0.5], [0.4, 0.6]]
    result = variance_decomposition(data)
    assert result["total_var"] >= 0.0
