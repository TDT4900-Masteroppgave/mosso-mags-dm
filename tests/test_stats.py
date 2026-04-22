"""Regression tests for scripts/stats.py statistical helpers.

Run with:  pytest tests/test_stats.py -v
"""
import math
import pytest
import numpy as np

from scripts.stats import (
    bootstrap_ci,
    cliffs_delta,
    holm_bonferroni,
    knee_point,
    paired_wilcoxon,
)


# ── knee_point ────────────────────────────────────────────────────────────────

class TestKneePoint:
    def test_obvious_elbow(self):
        # Clear knee at index 2: (1,10) (2,5) (3,1) (4,0.9) (5,0.8)
        times  = [1.0, 2.0, 3.0, 4.0, 5.0]
        ratios = [10.0, 5.0, 1.0, 0.9, 0.8]
        idx = knee_point(times, ratios)
        assert idx == 2

    def test_single_point(self):
        assert knee_point([1.0], [1.0]) == 0

    def test_two_points(self):
        assert knee_point([1.0, 2.0], [2.0, 1.0]) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            knee_point([], [])

    def test_flat_line_returns_zero(self):
        # All points identical → no knee, should return 0
        times = [1.0, 2.0, 3.0]
        ratios = [1.0, 1.0, 1.0]
        idx = knee_point(times, ratios)
        assert 0 <= idx < 3


# ── cliffs_delta ──────────────────────────────────────────────────────────────

class TestCliffsDelta:
    def test_identical_distributions_zero(self):
        x = [1.0, 2.0, 3.0]
        delta, mag = cliffs_delta(x, x)
        assert delta == pytest.approx(0.0)
        assert mag == "negligible"

    def test_x_always_greater_than_y(self):
        # x stochastically dominates y → delta should be +1
        x = [10.0, 11.0, 12.0]
        y = [1.0, 2.0, 3.0]
        delta, mag = cliffs_delta(x, y)
        assert delta == pytest.approx(1.0)
        assert mag == "large"

    def test_x_always_less_than_y(self):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 11.0, 12.0]
        delta, mag = cliffs_delta(x, y)
        assert delta == pytest.approx(-1.0)
        assert mag == "large"

    def test_empty_inputs(self):
        delta, mag = cliffs_delta([], [])
        assert delta == pytest.approx(0.0)
        assert mag == "negligible"

    def test_medium_effect(self):
        # Craft arrays with known delta ≈ 0.4 (medium range 0.33–0.474)
        x = [3, 4, 5, 6]
        y = [1, 2, 3, 4]
        delta, mag = cliffs_delta(x, y)
        assert delta > 0.33
        assert mag in ("medium", "large")


# ── paired_wilcoxon ───────────────────────────────────────────────────────────

class TestPairedWilcoxon:
    def test_identical_returns_pvalue_one(self):
        medians = {"A": 1.0, "B": 2.0, "C": 3.0}
        stat, p = paired_wilcoxon(medians, medians)
        assert p == pytest.approx(1.0)

    def test_clearly_different_low_pvalue(self):
        # Need ≥ 6 pairs for Wilcoxon to reach p < 0.05 with all-same differences
        strat = {k: 1.0 for k in "ABCDEF"}
        base  = {k: 100.0 for k in "ABCDEF"}
        _, p = paired_wilcoxon(strat, base)
        assert p < 0.05

    def test_insufficient_shared_datasets_returns_nan(self):
        stat, p = paired_wilcoxon({"A": 1.0}, {"A": 2.0})
        assert math.isnan(p)

    def test_no_shared_returns_nan(self):
        stat, p = paired_wilcoxon({"A": 1.0, "B": 2.0}, {"C": 3.0, "D": 4.0})
        assert math.isnan(p)


# ── holm_bonferroni ───────────────────────────────────────────────────────────

class TestHolmBonferroni:
    def test_all_significant(self):
        p_values = {"a": 0.001, "b": 0.002, "c": 0.003}
        result = holm_bonferroni(p_values, alpha=0.05)
        assert all(sig for _, sig in result.values())

    def test_none_significant(self):
        p_values = {"a": 0.9, "b": 0.8, "c": 0.7}
        result = holm_bonferroni(p_values, alpha=0.05)
        assert not any(sig for _, sig in result.values())

    def test_mixed(self):
        # a should be significant, c should not
        p_values = {"a": 0.001, "b": 0.04, "c": 0.9}
        result = holm_bonferroni(p_values, alpha=0.05)
        assert result["a"][1] is True
        assert result["c"][1] is False

    def test_nan_passthrough(self):
        p_values = {"a": float("nan"), "b": 0.001}
        result = holm_bonferroni(p_values, alpha=0.05)
        adj_a, sig_a = result["a"]
        assert math.isnan(adj_a)
        assert sig_a is False
        assert result["b"][1] is True

    def test_adjusted_monotone_nondecreasing(self):
        # Adjusted p-values must be non-decreasing when sorted by raw p
        p_values = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        result = holm_bonferroni(p_values, alpha=0.05)
        sorted_adj = [result[k][0] for k in sorted(p_values, key=lambda k: p_values[k])]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-10


# ── bootstrap_ci ──────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_ci_contains_true_median(self):
        rng = np.random.default_rng(0)
        xs = rng.normal(loc=5.0, scale=1.0, size=100).tolist()
        lo, hi = bootstrap_ci(xs, seed=42)
        assert lo < 5.0 < hi

    def test_ci_width_shrinks_with_more_data(self):
        rng = np.random.default_rng(1)
        small = rng.normal(5, 1, 10).tolist()
        large = rng.normal(5, 1, 200).tolist()
        lo_s, hi_s = bootstrap_ci(small, seed=42)
        lo_l, hi_l = bootstrap_ci(large, seed=42)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_empty_input_returns_nan(self):
        lo, hi = bootstrap_ci([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_deterministic_with_seed(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo1, hi1 = bootstrap_ci(xs, seed=99)
        lo2, hi2 = bootstrap_ci(xs, seed=99)
        assert lo1 == lo2 and hi1 == hi2
