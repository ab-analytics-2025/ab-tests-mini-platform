import numpy as np

from ab_platform.pipeline import (
    holm_adjust,
    ztest_proportions,
    bootstrap_mean_diff,
    bootstrap_diff_proportions_parametric,
    mde_for_proportions,
)


def test_holm_adjust_monotonic_and_significance():
    pvals = {"a": 0.001, "b": 0.02, "c": 0.04}
    out = holm_adjust(pvals, alpha=0.05)

    assert out["method"] == "holm-bonferroni"
    p_adj = out["p_adj"]

    # p_adj should be >= original p and within [0,1]
    for k, p in pvals.items():
        assert p_adj[k] >= p
        assert 0.0 <= p_adj[k] <= 1.0

    order = sorted(pvals, key=pvals.get)
    assert p_adj[order[0]] <= p_adj[order[1]] <= p_adj[order[2]]
    assert isinstance(out["significant"]["a"], bool)


def test_ztest_proportions_sanity():
    # treatment: 30/100, control: 20/100
    out = ztest_proportions(30, 100, 20, 100, ci_level=0.95)
    assert abs(out["diff"] - 0.10) < 1e-9
    lo, hi = out["ci"]
    assert lo < hi
    assert 0.0 <= out["p1"] <= 1.0
    assert 0.0 <= out["p2"] <= 1.0


def test_bootstrap_mean_diff_deterministic_seed():
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([1, 1, 2, 2, 3], dtype=float)
    r1 = bootstrap_mean_diff(a, b, n_boot=500, seed=123, ci_level=0.95)
    r2 = bootstrap_mean_diff(a, b, n_boot=500, seed=123, ci_level=0.95)
    assert r1["ci"] == r2["ci"]
    assert abs(r1["mean_diff"] - r2["mean_diff"]) < 1e-12


def test_bootstrap_diff_proportions_parametric_shape():
    out = bootstrap_diff_proportions_parametric(10, 100, 8, 100, n_boot=2000, seed=7, ci_level=0.95)
    lo, hi = out["ci"]
    assert lo < hi
    assert isinstance(out["mean_diff"], float)


def test_mde_for_proportions_positive():
    out = mde_for_proportions(p_baseline=0.02, n1=50000, n2=50000, alpha=0.05, power=0.8)
    assert out["mde_abs"] > 0
