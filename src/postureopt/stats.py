"""Statistical rigor utilities for postureopt experiments.

Provides confidence intervals, significance testing, Bonferroni correction,
and two-level variance decomposition — addressing issue #15 requirements
for OR-journal-quality statistical validation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def confidence_interval(
    samples: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute a confidence interval using the t-distribution.

    Uses the t-distribution rather than the normal approximation, which is
    appropriate when sample sizes are small (n < 30).

    Returns a dict with keys:
        mean, lower, upper, margin, n, confidence
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError("Need at least 2 samples to compute a confidence interval.")
    mean = float(arr.mean())
    se = float(_scipy_stats.sem(arr))
    t_crit = float(_scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * se
    return {
        "mean": mean,
        "lower": mean - margin,
        "upper": mean + margin,
        "margin": margin,
        "n": n,
        "confidence": confidence,
    }


def ci_str(samples: Sequence[float], confidence: float = 0.95, decimals: int = 3) -> str:
    """Return a compact CI string: 'mean (lower, upper)'."""
    ci = confidence_interval(samples, confidence)
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(ci['mean'])} ({fmt.format(ci['lower'])}, {fmt.format(ci['upper'])})"


def ci_latex(samples: Sequence[float], confidence: float = 0.95, decimals: int = 3) -> str:
    """Return a LaTeX-formatted CI string: 'mean~(lower, upper)'."""
    ci = confidence_interval(samples, confidence)
    fmt = f"{{:.{decimals}f}}"
    return (
        f"{fmt.format(ci['mean'])}~"
        f"({fmt.format(ci['lower'])}, {fmt.format(ci['upper'])})"
    )


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------


def paired_ttest(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired two-tailed t-test comparing samples_a vs. samples_b.

    Appropriate when each pair (a_i, b_i) comes from the same random seed,
    as is the case when comparing greedy vs. CEV under matched conditions.

    Returns a dict with keys:
        t_stat, p_value, mean_diff, reject_null, alpha
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("samples_a and samples_b must have the same length.")
    t_stat, p_value = _scipy_stats.ttest_rel(a, b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float((a - b).mean()),
        "reject_null": bool(p_value < alpha),
        "alpha": alpha,
    }


def bonferroni_correct(alpha: float, n_tests: int) -> float:
    """Return the Bonferroni-corrected significance threshold.

    With n_tests simultaneous comparisons, the per-test alpha is divided
    by n_tests to control the family-wise error rate at the nominal level.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1.")
    return alpha / n_tests


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


def variance_decomposition(data: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Two-level variance decomposition for nested Monte Carlo designs.

    Decomposes total variance into:
      - Outer variance: variance of group means across scenario seeds
        (captures sensitivity to the choice of scenario set)
      - Inner variance: mean within-group variance across simulation seeds
        (captures sensitivity to initial-state randomness)

    Parameters
    ----------
    data : array-like of shape (n_outer, n_inner)
        E.g., rows = different scenario seeds, columns = simulation seeds.

    Returns a dict with keys:
        outer_var, inner_var, total_var, icc, n_outer, n_inner
    where icc (intraclass correlation) = outer_var / (outer_var + inner_var).
    An ICC near 1 means scenario-set choice dominates; near 0 means
    initial-state randomness dominates.
    """
    arr = np.array(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be 2-dimensional: (n_outer, n_inner).")
    n_outer, n_inner = arr.shape

    group_means = arr.mean(axis=1)
    outer_var = float(np.var(group_means, ddof=1)) if n_outer > 1 else 0.0

    within_vars = np.var(arr, axis=1, ddof=1) if n_inner > 1 else np.zeros(n_outer)
    inner_var = float(within_vars.mean())

    total_var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0

    denom = outer_var + inner_var
    icc = outer_var / denom if denom > 0 else 0.0

    return {
        "outer_var": outer_var,
        "inner_var": inner_var,
        "total_var": total_var,
        "icc": icc,
        "n_outer": n_outer,
        "n_inner": n_inner,
    }
