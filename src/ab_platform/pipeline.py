from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ====== Конфиг/константы ======

REPORT_VERSION = "0.2.0"

DEFAULT_HIST_BINS = 50
DEFAULT_FIGSIZE_WIDE = (10, 6)
DEFAULT_FIGSIZE = (8, 5)

# SRM: если expected_split не задан, мы НЕ имеем права "фейлить" дизайн на основании 50/50,
# если фактический сплит явно не 50/50. В таком случае SRM помечаем как skipped (нужно задать expected_split).
DEFAULT_SRM_ASSUME_UNIFORM_TOL = 0.02  # 2% отклонение от равного сплита считаем допустимым

# Balance: на больших N p-value почти всегда "значимы". Поэтому gating делаем по effect size.
DEFAULT_BALANCE_SMD_THRESHOLD = 0.10       # standardised mean difference
DEFAULT_BALANCE_CRAMERSV_THRESHOLD = 0.10  # Cramér's V

# Практическая значимость (для primary metric: conversion)
DEFAULT_MIN_UPLIFT_ABS = 0.0

# Датасет
COL_USER_ID = "user id"
COL_GROUP = "test group"
COL_CONVERTED = "converted"
COL_TOTAL_ADS = "total ads"
COL_MOST_DAY = "most ads day"
COL_MOST_HOUR = "most ads hour"

CONTROL_LABEL = "psa"
TREATMENT_LABEL = "ad"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    cleaned_path: str
    report_path: str
    figures_dir: str
    write_plots: bool
    seed: int
    n_boot: int
    alpha: float
    ci_level: float

    # Step 1: контракт эксперимента / параметры дизайна и решений
    expected_split: Optional[Dict[str, float]]
    srm_uniform_tol: float

    balance_smd_threshold: float
    balance_cramersv_threshold: float

    min_uplift_abs: float


# ====== utils ======

def convert_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [convert_numpy(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_binary_converted(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({"true": 1, "false": 0, "1": 1, "0": 0})
    return mapped.fillna(series).astype(int)


def holm_adjust(p_values: Dict[str, float], alpha: float) -> Dict[str, Any]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    significant: Dict[str, bool] = {}

    prev_adj = 0.0
    for i, (name, p) in enumerate(items, start=1):
        a = (m - i + 1) * p
        a = max(a, prev_adj)
        a = min(a, 1.0)
        adj[name] = a
        prev_adj = a

    for name, p_adj in adj.items():
        significant[name] = p_adj < alpha

    return {"method": "holm-bonferroni", "alpha": alpha, "p_adj": adj, "significant": significant}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _normalize_expected_split(expected_split: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    """Нормализуем вероятности, если сумма != 1. Возвращаем (split, normalized_flag)."""
    s = float(sum(expected_split.values()))
    if s <= 0:
        raise ValueError("expected_split must have positive sum")
    if abs(s - 1.0) < 1e-9:
        return {k: float(v) for k, v in expected_split.items()}, False
    return {k: float(v) / s for k, v in expected_split.items()}, True


def _validate_expected_split(groups: List[str], expected_split: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "errors": [], "warnings": []}

    for g in groups:
        if g not in expected_split:
            out["ok"] = False
            out["errors"].append(f"expected_split missing group '{g}'")

    extra = [g for g in expected_split.keys() if g not in groups]
    if extra:
        out["warnings"].append(f"expected_split has extra keys not present in data: {extra}")

    for k, v in expected_split.items():
        if v < 0:
            out["ok"] = False
            out["errors"].append(f"expected_split[{k}] must be >= 0")

    s = float(sum(expected_split.values()))
    if s <= 0:
        out["ok"] = False
        out["errors"].append("expected_split sum must be > 0")

    return out


# ====== checks ======

def data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    required = [COL_USER_ID, COL_GROUP, COL_CONVERTED, COL_TOTAL_ADS, COL_MOST_DAY, COL_MOST_HOUR]
    report: Dict[str, Any] = {"required_columns_present": True, "checks": {}, "errors": []}

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        report["required_columns_present"] = False
        report["errors"].append({"missing_columns": missing_cols})
        return report

    report["checks"]["nulls_total_required"] = int(df[required].isnull().sum().sum())
    report["checks"]["user_id_unique"] = bool(df[COL_USER_ID].is_unique)
    report["checks"]["group_values"] = sorted(df[COL_GROUP].dropna().unique().tolist())
    report["checks"]["converted_values"] = sorted(df[COL_CONVERTED].dropna().unique().tolist())
    report["checks"]["most_ads_hour_in_range"] = bool(df[COL_MOST_HOUR].between(0, 23).all())
    report["checks"]["total_ads_non_negative"] = bool((df[COL_TOTAL_ADS] >= 0).all())
    return report


def srm_check(
    group_counts: Dict[str, int],
    expected_split: Optional[Dict[str, float]],
    alpha: float,
    uniform_tol: float = DEFAULT_SRM_ASSUME_UNIFORM_TOL,
) -> Dict[str, Any]:
    """SRM (Sample Ratio Mismatch).

    Правило:
    - Если expected_split задан: проверяем SRM строго относительно него -> pass/fail.
    - Если expected_split НЕ задан:
        * если фактический сплит близок к 50/50 (или к равному для k групп) -> можем проверить относительно равного;
        * если не близок -> SRM = skipped (не считаем дизайн проваленным, пока не знаем плановый сплит).
    """
    groups = sorted(group_counts.keys())
    n = int(sum(group_counts.values()))

    if len(groups) < 2 or n == 0:
        return {"error": "Need at least 2 groups with non-zero size", "observed": group_counts}

    observed_frac = {g: _safe_div(group_counts[g], n) for g in groups}

    used_expected: Dict[str, float]
    expected_provided = expected_split is not None
    normalized_flag = False
    validation: Optional[Dict[str, Any]] = None

    if expected_split is not None:
        used_expected, normalized_flag = _normalize_expected_split(expected_split)
        validation = _validate_expected_split(groups, used_expected)
        if not validation["ok"]:
            return {
                "observed": {g: int(group_counts[g]) for g in groups},
                "observed_frac": observed_frac,
                "expected_provided": True,
                "expected": {g: float(used_expected.get(g, 0.0)) for g in sorted(used_expected.keys())},
                "alpha": float(alpha),
                "passes": None,
                "status": "error",
                "message": "Invalid expected_split",
                "validation": validation,
            }
    else:
        used_expected = {g: 1.0 / len(groups) for g in groups}

    obs = np.array([group_counts[g] for g in groups], dtype=float)
    exp = np.array([used_expected[g] * n for g in groups], dtype=float)

    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)

    status: str
    passes: Optional[bool]
    message: str = ""
    needs_expected_split = False

    if expected_provided:
        passes = bool(p >= alpha)
        status = "pass" if passes else "fail"
        if normalized_flag:
            message = "expected_split was normalized to sum=1"
    else:
        uniform = 1.0 / len(groups)
        max_dev = max(abs(observed_frac[g] - uniform) for g in groups)
        if max_dev <= float(uniform_tol):
            passes = bool(p >= alpha)
            status = "pass" if passes else "fail"
            message = f"expected_split not provided; assumed uniform because max deviation {max_dev:.4f} <= {uniform_tol:.4f}"
        else:
            passes = None
            status = "skipped"
            needs_expected_split = True
            message = (
                "expected_split not provided and observed split is far from uniform; "
                "SRM check skipped to avoid incorrect conclusion. Provide expected_split to validate SRM."
            )

    return {
        "observed": {g: int(group_counts[g]) for g in groups},
        "observed_frac": observed_frac,
        "expected_provided": bool(expected_provided),
        "expected": {g: float(used_expected[g]) for g in groups},
        "chi2": float(chi2),
        "p_value": float(p),
        "alpha": float(alpha),
        "passes": passes,
        "status": status,
        "needs_expected_split": bool(needs_expected_split),
        "suggested_expected_split": observed_frac,
        "message": message,
        "validation": validation,
    }


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    v = ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1))) / (len(a) + len(b) - 2)
    return float(np.sqrt(max(v, 0.0)))


def standardized_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """SMD (Cohen's d style): (mean(a)-mean(b)) / pooled_std."""
    ps = _pooled_std(a, b)
    if ps == 0:
        return 0.0
    return float((float(np.mean(a)) - float(np.mean(b))) / ps)


def cramers_v_from_crosstab(ct: pd.DataFrame) -> float:
    """Cramér's V для таблицы сопряженности."""
    if ct.size == 0:
        return 0.0
    chi2, _p, _dof, _ = stats.chi2_contingency(ct)
    n = float(ct.to_numpy().sum())
    if n <= 0:
        return 0.0
    r, k = ct.shape
    denom = n * float(min(r - 1, k - 1))
    return float(np.sqrt(_safe_div(chi2, denom))) if denom > 0 else 0.0


def balance_checks(
    df: pd.DataFrame,
    alpha: float,
    smd_threshold: float = DEFAULT_BALANCE_SMD_THRESHOLD,
    cramersv_threshold: float = DEFAULT_BALANCE_CRAMERSV_THRESHOLD,
) -> Dict[str, Any]:
    groups = sorted(df[COL_GROUP].unique().tolist())
    if len(groups) != 2:
        return {"error": f"Expected 2 groups for balance checks, got {groups}"}

    if CONTROL_LABEL in groups and TREATMENT_LABEL in groups:
        g_control, g_treat = CONTROL_LABEL, TREATMENT_LABEL
    else:
        g_control, g_treat = groups[0], groups[1]

    d_c = df[df[COL_GROUP] == g_control]
    d_t = df[df[COL_GROUP] == g_treat]

    out: Dict[str, Any] = {
        "alpha": float(alpha),
        "thresholds": {
            "smd_abs_max": float(smd_threshold),
            "cramers_v_max": float(cramersv_threshold),
        },
        "groups": {"control": g_control, "treatment": g_treat},
        "numeric": {},
        "categorical": {},
        "overall": {},
    }

    numeric_pass_effect = True
    numeric_pass_p = True

    for col in (COL_TOTAL_ADS, COL_MOST_HOUR):
        a = d_t[col].to_numpy()
        b = d_c[col].to_numpy()

        ks = stats.ks_2samp(a, b)
        smd = standardized_mean_diff(a, b)
        passes_p = bool(ks.pvalue >= alpha)
        passes_eff = bool(abs(smd) <= smd_threshold)

        out["numeric"][col] = {
            "test": "ks_2samp",
            "p_value": float(ks.pvalue),
            "passes_p_value": passes_p,
            "smd": float(smd),
            "passes_effect_size": passes_eff,
            "passes": passes_eff,
            "mean_diff": float(float(np.mean(a)) - float(np.mean(b))),
        }

        numeric_pass_effect = bool(numeric_pass_effect and passes_eff)
        numeric_pass_p = bool(numeric_pass_p and passes_p)

    ct = pd.crosstab(df[COL_GROUP], df[COL_MOST_DAY])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = cramers_v_from_crosstab(ct)
    passes_p = bool(p >= alpha)
    passes_eff = bool(v <= cramersv_threshold)

    out["categorical"][COL_MOST_DAY] = {
        "test": "chi2_contingency",
        "p_value": float(p),
        "dof": int(dof),
        "passes_p_value": passes_p,
        "cramers_v": float(v),
        "passes_effect_size": passes_eff,
        "passes": passes_eff,
    }

    out["overall"] = {
        "passes_effect_size": bool(numeric_pass_effect and passes_eff),
        "passes_p_value": bool(numeric_pass_p and passes_p),
    }

    return out


# ====== stats ======

def ztest_proportions(x1: int, n1: int, x2: int, n2: int, ci_level: float) -> Dict[str, Any]:
    if n1 <= 0 or n2 <= 0:
        return {"error": "n1 and n2 must be > 0"}

    p1 = x1 / n1
    p2 = x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    zcrit = stats.norm.ppf(0.5 + ci_level / 2)
    se_ci = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    delta = zcrit * se_ci

    return {
        "z": float(z),
        "p_value": float(p),
        "diff": float(p1 - p2),
        "ci": [float((p1 - p2) - delta), float((p1 - p2) + delta)],
        "p1": float(p1),
        "p2": float(p2),
    }


def bootstrap_mean_diff(
    data1: np.ndarray,
    data2: np.ndarray,
    n_boot: int,
    seed: int,
    ci_level: float,
    max_n: int = 50000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    a = np.asarray(data1)
    b = np.asarray(data2)

    if len(a) > max_n:
        a = rng.choice(a, size=max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, size=max_n, replace=False)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = float(np.mean(sa) - np.mean(sb))

    lo = (1 - ci_level) / 2 * 100
    hi = (1 + ci_level) / 2 * 100
    ci = np.percentile(diffs, [lo, hi])
    return {"ci": [float(ci[0]), float(ci[1])], "mean_diff": float(np.mean(diffs)), "used_n": [int(len(a)), int(len(b))]}


def bootstrap_diff_proportions_parametric(
    x1: int, n1: int, x2: int, n2: int, n_boot: int, seed: int, ci_level: float
) -> Dict[str, Any]:
    """Параметрический bootstrap для разницы долей (быстро и без построения массивов длиной n)."""
    rng = np.random.default_rng(seed)
    p1 = x1 / n1
    p2 = x2 / n2

    s1 = rng.binomial(n1, p1, size=n_boot) / n1
    s2 = rng.binomial(n2, p2, size=n_boot) / n2
    diffs = s1 - s2

    lo = (1 - ci_level) / 2
    hi = (1 + ci_level) / 2
    ci = np.quantile(diffs, [lo, hi])

    return {"ci": [float(ci[0]), float(ci[1])], "mean_diff": float(np.mean(diffs)), "note": "parametric_binomial"}


def chi_square_2x2(x1: int, n1: int, x2: int, n2: int) -> Dict[str, Any]:
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "expected": expected.tolist()}


def ttest_ind(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p_value": float(p)}


def mde_for_proportions(p_baseline: float, n1: int, n2: int, alpha: float, power: float = 0.8) -> Dict[str, Any]:
    """MDE для разницы долей (грубая оценка через нормальное приближение)."""
    if n1 <= 0 or n2 <= 0:
        return {"error": "n1 and n2 must be > 0"}

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p = float(p_baseline)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    mde = (z_alpha + z_beta) * se
    return {"power_target": float(power), "mde_abs": float(mde), "baseline": float(p), "alpha": float(alpha)}


# ====== plots ======

def _save_plot_paths(figures_dir: str, name: str) -> str:
    ensure_dir(figures_dir)
    return os.path.join(figures_dir, name)


def plot_hist_total_ads(df: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE_WIDE)
    for g, sub in df.groupby(COL_GROUP):
        plt.hist(sub[COL_TOTAL_ADS], bins=DEFAULT_HIST_BINS, alpha=0.5, label=str(g))
    plt.title("Total ads distribution")
    plt.xlabel("total ads")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_conversion_by_day(df: pd.DataFrame, path: str) -> None:
    pivot = df.groupby([COL_MOST_DAY, COL_GROUP])[COL_CONVERTED].mean().unstack()
    pivot = pivot.reindex(DAY_ORDER)
    pivot.plot(kind="bar", figsize=DEFAULT_FIGSIZE_WIDE)
    plt.title("Conversion rate by day")
    plt.ylabel("conversion rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for (i, j), v in np.ndenumerate(corr.values):
        plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_conversion_rates(conversion_rates: Dict[str, float], path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    keys = list(conversion_rates.keys())
    vals = [conversion_rates[k] for k in keys]
    plt.bar(keys, vals)
    plt.title("Conversion rates by group")
    plt.ylabel("conversion rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_box_total_ads(control: np.ndarray, test: np.ndarray, path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.boxplot([control, test], labels=[CONTROL_LABEL, TREATMENT_LABEL])
    plt.title("Total ads: control vs treatment")
    plt.ylabel("total ads")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ====== pipeline ======

def run_pipeline(
    input_path: str,
    cleaned_path: str,
    report_path: str,
    figures_dir: str,
    seed: int = 42,
    n_boot: int = 5000,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    write_plots: bool = True,
    expected_split: Optional[Dict[str, float]] = None,
    srm_uniform_tol: float = DEFAULT_SRM_ASSUME_UNIFORM_TOL,
    balance_smd_threshold: float = DEFAULT_BALANCE_SMD_THRESHOLD,
    balance_cramersv_threshold: float = DEFAULT_BALANCE_CRAMERSV_THRESHOLD,
    min_uplift_abs: float = DEFAULT_MIN_UPLIFT_ABS,
) -> Dict[str, Any]:
    cfg = RunConfig(
        input_path=input_path,
        cleaned_path=cleaned_path,
        report_path=report_path,
        figures_dir=figures_dir,
        write_plots=write_plots,
        seed=seed,
        n_boot=n_boot,
        alpha=alpha,
        ci_level=ci_level,
        expected_split=expected_split,
        srm_uniform_tol=srm_uniform_tol,
        balance_smd_threshold=balance_smd_threshold,
        balance_cramersv_threshold=balance_cramersv_threshold,
        min_uplift_abs=min_uplift_abs,
    )

    ensure_parent_dir(cfg.cleaned_path)
    ensure_parent_dir(cfg.report_path)
    ensure_dir(cfg.figures_dir)

    started = time.time()
    results: Dict[str, Any] = {}

    df = pd.read_csv(cfg.input_path)

    # нормализация
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df[COL_CONVERTED] = to_binary_converted(df[COL_CONVERTED])

    # очистка: уникальность пользователей
    dup_rows = int(df.duplicated().sum())
    dup_user_ids = int(df.duplicated(subset=[COL_USER_ID]).sum())
    df = df.drop_duplicates(subset=[COL_USER_ID], keep="first")

    # сохранить cleaned
    df.to_csv(cfg.cleaned_path, index=False)

    # meta + контракт
    results["meta"] = {
        "report_version": REPORT_VERSION,
        "seed": cfg.seed,
        "n_boot": cfg.n_boot,
        "alpha": cfg.alpha,
        "ci_level": cfg.ci_level,
        "input_path": cfg.input_path,
        "cleaned_path": cfg.cleaned_path,
        "report_path": cfg.report_path,
        "figures_dir": cfg.figures_dir,
        "write_plots": cfg.write_plots,
        "runtime_sec": None,
        # contract
        "unit_of_randomization": COL_USER_ID,
        "group_column": COL_GROUP,
        "primary_metric": COL_CONVERTED,
        "control_label": CONTROL_LABEL,
        "treatment_label": TREATMENT_LABEL,
        # decision params
        "expected_split": cfg.expected_split,
        "srm_uniform_tol": float(cfg.srm_uniform_tol),
        "balance_smd_threshold": float(cfg.balance_smd_threshold),
        "balance_cramersv_threshold": float(cfg.balance_cramersv_threshold),
        "min_uplift_abs": float(cfg.min_uplift_abs),
    }

    # базовая инфа
    results["dataset_shape"] = [int(df.shape[0]), int(df.shape[1])]
    results["columns"] = df.columns.tolist()
    results["data_types"] = df.dtypes.astype(str).to_dict()
    results["first_5_rows"] = df.head().to_dict("records")

    results["summary_stats"] = df.describe(include="all").to_dict()
    results["unique_test_groups"] = df[COL_GROUP].unique().tolist()
    results["test_group_counts"] = df[COL_GROUP].value_counts().to_dict()

    results["overall_conversion_rate"] = float(df[COL_CONVERTED].mean())
    results["conversion_by_group"] = {k: float(v) for k, v in df.groupby(COL_GROUP)[COL_CONVERTED].mean().to_dict().items()}

    # качество данных
    results["missing_values"] = df.isnull().sum().to_dict()
    results["duplicates_rows"] = dup_rows
    results["duplicates_user_id"] = dup_user_ids
    results["unique_users"] = int(df[COL_USER_ID].nunique())
    results["total_rows"] = int(len(df))
    results["data_quality"] = data_quality(df)

    # дизайн-валидация
    results["srm"] = srm_check(
        results["test_group_counts"],
        expected_split=cfg.expected_split,
        alpha=cfg.alpha,
        uniform_tol=cfg.srm_uniform_tol,
    )
    results["balance_checks"] = balance_checks(
        df,
        alpha=cfg.alpha,
        smd_threshold=cfg.balance_smd_threshold,
        cramersv_threshold=cfg.balance_cramersv_threshold,
    )

    # EDA summaries
    results["eda_total_ads_hist"] = df.groupby(COL_GROUP)[COL_TOTAL_ADS].describe().to_dict()
    results["eda_conversion_by_day"] = df.groupby([COL_MOST_DAY, COL_GROUP])[COL_CONVERTED].mean().unstack().to_dict()
    corr = df[[COL_CONVERTED, COL_TOTAL_ADS, COL_MOST_HOUR]].corr()
    results["eda_correlation"] = corr.to_dict()

    # A/B split
    if CONTROL_LABEL not in df[COL_GROUP].unique().tolist() or TREATMENT_LABEL not in df[COL_GROUP].unique().tolist():
        results["error"] = f"Expected groups '{CONTROL_LABEL}' and '{TREATMENT_LABEL}' in column '{COL_GROUP}'"
        report_obj = convert_numpy(results)
        with open(cfg.report_path, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)
        return report_obj

    control = df[df[COL_GROUP] == CONTROL_LABEL]
    test = df[df[COL_GROUP] == TREATMENT_LABEL]

    results["control_group_size"] = int(len(control))
    results["test_group_size"] = int(len(test))
    results["control_conversion_rate"] = float(control[COL_CONVERTED].mean())
    results["test_conversion_rate"] = float(test[COL_CONVERTED].mean())
    results["control_mean_total_ads"] = float(control[COL_TOTAL_ADS].mean())
    results["test_mean_total_ads"] = float(test[COL_TOTAL_ADS].mean())

    # основной тест на conversion
    x_c = int(control[COL_CONVERTED].sum())
    n_c = int(len(control))
    x_t = int(test[COL_CONVERTED].sum())
    n_t = int(len(test))

    results["chi_square_test"] = chi_square_2x2(x_t, n_t, x_c, n_c)
    results["ztest_conversion"] = ztest_proportions(x_t, n_t, x_c, n_c, ci_level=cfg.ci_level)
    results["bootstrap_conversion_diff"] = bootstrap_diff_proportions_parametric(x_t, n_t, x_c, n_c, cfg.n_boot, cfg.seed, cfg.ci_level)

    # total ads (пример вторичной метрики)
    results["t_test_total_ads"] = ttest_ind(test[COL_TOTAL_ADS].values, control[COL_TOTAL_ADS].values)
    results["bootstrap_total_ads_diff"] = bootstrap_mean_diff(
        test[COL_TOTAL_ADS].values,
        control[COL_TOTAL_ADS].values,
        n_boot=cfg.n_boot,
        seed=cfg.seed,
        ci_level=cfg.ci_level,
    )

    results["mde_power"] = mde_for_proportions(
        p_baseline=results["control_conversion_rate"],
        n1=n_t,
        n2=n_c,
        alpha=cfg.alpha,
        power=0.8,
    )

    raw_p = {
        "conversion_chi2": float(results["chi_square_test"]["p_value"]),
        "conversion_ztest": float(results["ztest_conversion"]["p_value"]),
        "total_ads_ttest": float(results["t_test_total_ads"]["p_value"]),
        "balance_total_ads_ks": float(results["balance_checks"]["numeric"][COL_TOTAL_ADS]["p_value"]),
        "balance_hour_ks": float(results["balance_checks"]["numeric"][COL_MOST_HOUR]["p_value"]),
        "balance_day_chi2": float(results["balance_checks"]["categorical"][COL_MOST_DAY]["p_value"]),
    }
    results["p_adjusted"] = holm_adjust(raw_p, alpha=cfg.alpha)

    # ====== recommendation (Step 1) ======
    conv_sig = bool(results["p_adjusted"]["significant"]["conversion_ztest"])
    conv_diff = float(results["ztest_conversion"]["diff"])
    conv_practical = bool(conv_diff >= cfg.min_uplift_abs)

    srm_status = results["srm"].get("status")
    srm_pass = results["srm"].get("passes") is True
    srm_skipped = srm_status == "skipped"

    balance_pass = bool(results["balance_checks"].get("overall", {}).get("passes_effect_size", False))

    decision = "HOLD"
    rollout = False

    if srm_skipped:
        decision = "HOLD"
        rollout = False
    elif not srm_pass:
        decision = "ROLLBACK"
        rollout = False
    else:
        if balance_pass and conv_sig and conv_practical and conv_diff > 0:
            decision = "ROLLOUT"
            rollout = True
        else:
            decision = "HOLD"
            rollout = False

    results["recommendation"] = {
        "rollout": bool(rollout),
        "decision": decision,
        "reasons": {
            "srm_status": srm_status,
            "srm_passes": bool(srm_pass),
            "needs_expected_split": bool(results["srm"].get("needs_expected_split", False)),
            "balance_passes_effect_size": bool(balance_pass),
            "conversion_significant_after_adjustment": bool(conv_sig),
            "conversion_diff": float(conv_diff),
            "min_uplift_abs": float(cfg.min_uplift_abs),
            "conversion_practical_significance": bool(conv_practical),
        },
    }

    # plots
    artifacts: Dict[str, str] = {}
    if cfg.write_plots:
        p1 = _save_plot_paths(cfg.figures_dir, "eda_hist_total_ads.png")
        plot_hist_total_ads(df, p1)
        artifacts["eda_hist_total_ads"] = os.path.basename(p1)

        p2 = _save_plot_paths(cfg.figures_dir, "eda_bar_conversion_day.png")
        plot_conversion_by_day(df, p2)
        artifacts["eda_bar_conversion_day"] = os.path.basename(p2)

        p3 = _save_plot_paths(cfg.figures_dir, "eda_heatmap_corr.png")
        plot_corr_heatmap(corr, p3)
        artifacts["eda_heatmap_corr"] = os.path.basename(p3)

        p4 = _save_plot_paths(cfg.figures_dir, "viz_bar_conversion_rates.png")
        plot_conversion_rates(results["conversion_by_group"], p4)
        artifacts["viz_bar_conversion_rates"] = os.path.basename(p4)

        p5 = _save_plot_paths(cfg.figures_dir, "viz_box_total_ads.png")
        plot_box_total_ads(control[COL_TOTAL_ADS].values, test[COL_TOTAL_ADS].values, p5)
        artifacts["viz_box_total_ads"] = os.path.basename(p5)

    results["plot_files"] = artifacts

    results["meta"]["runtime_sec"] = float(time.time() - started)

    # записать отчёт
    report_obj = convert_numpy(results)
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    return report_obj
