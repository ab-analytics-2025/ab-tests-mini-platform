import numpy as np
import pandas as pd

from ab_platform.pipeline import (
    srm_check,
    balance_checks,
    standardized_mean_diff,
    cramers_v_from_crosstab,
)


def test_srm_skipped_without_expected_split_on_unequal_split():
    counts = {"ad": 96, "psa": 4}
    out = srm_check(counts, expected_split=None, alpha=0.05, uniform_tol=0.02)
    assert out["status"] == "skipped"
    assert out["passes"] is None
    assert out["needs_expected_split"] is True


def test_srm_pass_with_expected_split_matching():
    counts = {"ad": 96, "psa": 4}
    out = srm_check(counts, expected_split={"ad": 0.96, "psa": 0.04}, alpha=0.05, uniform_tol=0.02)
    assert out["status"] in ("pass", "fail")
    assert out["passes"] is True


def test_standardized_mean_diff_zero_for_equal_arrays():
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([1, 2, 3, 4, 5], dtype=float)
    assert standardized_mean_diff(a, b) == 0.0


def test_cramers_v_basic():
    ct = pd.DataFrame([[10, 10], [10, 10]])
    v = cramers_v_from_crosstab(ct)
    assert abs(v) < 1e-12


def test_balance_checks_pass_for_identical_distributions():
    df = pd.DataFrame(
        {
            "user id": range(200),
            "test group": ["psa"] * 100 + ["ad"] * 100,
            "converted": [0] * 200,
            "total ads": [10] * 200,
            "most ads day": ["Monday"] * 200,
            "most ads hour": [12] * 200,
        }
    )
    out = balance_checks(df, alpha=0.05, smd_threshold=0.1, cramersv_threshold=0.1)
    assert out["overall"]["passes_effect_size"] is True
