from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ====== Конфиг/константы ======

REPORT_VERSION = "0.1.0"

DEFAULT_HIST_BINS = 50
DEFAULT_FIGSIZE_WIDE = (10, 6)
DEFAULT_FIGSIZE = (8, 5)

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


def srm_check(group_counts: Dict[str, int], expected_split: Optional[Dict[str, float]], alpha: float) -> Dict[str, Any]:
    groups = list(group_counts.keys())
    n = sum(group_counts.values())

    if len(groups) < 2 or n == 0:
        return {"error": "Need at least 2 groups with non-zero size", "observed": group_counts}

    if expected_split is None:
        expected_split = {g: 1.0 / len(groups) for g in groups}

    obs = np.array([group_counts[g] for g in groups], dtype=float)
    exp = np.array([expected_split[g] * n for g in groups], dtype=float)

    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
    return {
        "observed": {g: int(group_counts[g]) for g in groups},
        "expected": {g: float(expected_split[g]) for g in groups},
        "chi2": float(chi2),
        "p_value": float(p),
        "alpha": float(alpha),
        "passes": bool(p >= alpha),
    }


def balance_checks(df: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    groups = sorted(df[COL_GROUP].unique().tolist())
    if len(groups) != 2:
        return {"error": f"Expected 2 groups for balance checks, got {groups}"}

    g1, g2 = groups
    d1 = df[df[COL_GROUP] == g1]
    d2 = df[df[COL_GROUP] == g2]

    out: Dict[str, Any] = {"alpha": alpha, "numeric": {}, "categorical": {}}

    for col in (COL_TOTAL_ADS, COL_MOST_HOUR):
        ks = stats.ks_2samp(d1[col].values, d2[col].values)
        out["numeric"][col] = {
            "test": "ks_2samp",
            "p_value": float(ks.pvalue),
            "passes": bool(ks.pvalue >= alpha),
            "mean_diff": float(d1[col].mean() - d2[col].mean()),
        }

    ct = pd.crosstab(df[COL_GROUP], df[COL_MOST_DAY])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    out["categorical"][COL_MOST_DAY] = {
        "test": "chi2_contingency",
        "p_value": float(p),
        "dof": int(dof),
        "passes": bool(p >= alpha),
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
    rng = np.random.default_rng(seed)
    p1 = x1 / n1
    p2 = x2 / n2

    s1 = rng.binomial(n1, p1, size=n_boot) / n1
    s2 = rng.binomial(n2, p2, size=n_boot) / n2
    diffs = s1 - s2

    lo = (1 - ci_level) / 2 * 100
    hi = (1 + ci_level) / 2 * 100
    ci = np.percentile(diffs, [lo, hi])
    return {"ci": [float(ci[0]), float(ci[1])], "mean_diff": float(np.mean(diffs)), "method": "parametric_binomial"}


def mde_for_proportions(p_baseline: float, n1: int, n2: int, alpha: float, power: float = 0.8) -> Dict[str, Any]:
    if n1 <= 0 or n2 <= 0:
        return {"error": "n1 and n2 must be > 0"}

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    se = np.sqrt(p_baseline * (1 - p_baseline) * (1 / n1 + 1 / n2))
    mde = (z_alpha + z_beta) * se
    return {"power_target": power, "mde_abs": float(mde), "baseline": float(p_baseline), "alpha": float(alpha)}


# ====== plots ======

def _save_plot_paths(figures_dir: str, name: str) -> str:
    return os.path.join(figures_dir, name)


def plot_hist_total_ads(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE_WIDE)
    for g, sub in df.groupby(COL_GROUP):
        plt.hist(sub[COL_TOTAL_ADS], bins=DEFAULT_HIST_BINS, alpha=0.5, label=str(g))
    plt.title("Распределение общего количества рекламы по группам тестирования")
    plt.xlabel(COL_TOTAL_ADS)
    plt.ylabel("count")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_conversion_by_day(df: pd.DataFrame, out_path: str) -> None:
    conv_day = df.groupby([COL_MOST_DAY, COL_GROUP])[COL_CONVERTED].mean().unstack()
    conv_day = conv_day.reindex(DAY_ORDER)

    plt.figure(figsize=DEFAULT_FIGSIZE_WIDE)
    x = np.arange(len(conv_day.index))
    width = 0.35
    cols = list(conv_day.columns)

    if len(cols) == 2:
        plt.bar(x - width / 2, conv_day[cols[0]].values, width, label=str(cols[0]))
        plt.bar(x + width / 2, conv_day[cols[1]].values, width, label=str(cols[1]))
    else:
        for i, c in enumerate(cols):
            plt.bar(x + (i - len(cols) / 2) * width, conv_day[c].values, width, label=str(c))

    plt.xticks(x, conv_day.index, rotation=30, ha="right")
    plt.title("Коэффициент конверсии по дням и группам")
    plt.ylabel("conversion")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    mat = corr.values
    plt.imshow(mat)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    plt.title("Матрица корреляции")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_conversion_rates(conversion_rates: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.bar(conversion_rates[COL_GROUP], conversion_rates[COL_CONVERTED])
    plt.title("Коэффициенты конверсии по группам")
    plt.ylabel("conversion")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_box_total_ads(control: pd.DataFrame, test: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.boxplot(
        [control[COL_TOTAL_ADS].values, test[COL_TOTAL_ADS].values],
        labels=[CONTROL_LABEL, TREATMENT_LABEL],
        showfliers=False,
    )
    plt.title("Распределение total ads по группам")
    plt.ylabel(COL_TOTAL_ADS)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ====== main API ======

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
    df = df.drop_duplicates(subset=[COL_USER_ID], keep="first")

    # сохранить cleaned
    df.to_csv(cfg.cleaned_path, index=False)

    # meta
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
    results["conversion_by_group"] = {
        k: float(v) for k, v in df.groupby(COL_GROUP)[COL_CONVERTED].mean().to_dict().items()
    }

    # качество данных
    results["missing_values"] = df.isnull().sum().to_dict()
    results["duplicates"] = dup_rows
    results["unique_users"] = int(df[COL_USER_ID].nunique())
    results["total_rows"] = int(len(df))
    results["data_quality"] = data_quality(df)

    # дизайн-валидация
    results["srm"] = srm_check(results["test_group_counts"], expected_split=None, alpha=cfg.alpha)
    results["balance_checks"] = balance_checks(df, alpha=cfg.alpha)

    # EDA summaries
    results["eda_total_ads_hist"] = df.groupby(COL_GROUP)[COL_TOTAL_ADS].describe().to_dict()
    results["eda_conversion_by_day"] = df.groupby([COL_MOST_DAY, COL_GROUP])[COL_CONVERTED].mean().unstack().to_dict()
    corr = df[[COL_CONVERTED, COL_TOTAL_ADS, COL_MOST_HOUR]].corr()
    results["eda_correlation"] = corr.to_dict()

    # A/B split
    control = df[df[COL_GROUP] == CONTROL_LABEL]
    test = df[df[COL_GROUP] == TREATMENT_LABEL]

    results["control_group_size"] = int(len(control))
    results["test_group_size"] = int(len(test))
    results["control_conversion_rate"] = float(control[COL_CONVERTED].mean())
    results["test_conversion_rate"] = float(test[COL_CONVERTED].mean())
    results["control_mean_total_ads"] = float(control[COL_TOTAL_ADS].mean())
    results["test_mean_total_ads"] = float(test[COL_TOTAL_ADS].mean())

    conversion_rates = df.groupby(COL_GROUP)[COL_CONVERTED].mean().reset_index()
    results["viz_conversion_rates"] = conversion_rates.to_dict("records")
    results["viz_total_ads_box"] = df.groupby(COL_GROUP)[COL_TOTAL_ADS].describe().to_dict()

    # stats
    contingency_table = pd.crosstab(df[COL_GROUP], df[COL_CONVERTED])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency_table)
    results["chi_square_test"] = {"chi2": float(chi2), "p_value": float(p_chi), "dof": int(dof)}

    x_c = int(control[COL_CONVERTED].sum())
    n_c = int(len(control))
    x_t = int(test[COL_CONVERTED].sum())
    n_t = int(len(test))

    results["ztest_conversion"] = ztest_proportions(x_t, n_t, x_c, n_c, ci_level=cfg.ci_level)
    results["bootstrap_conversion_diff"] = bootstrap_diff_proportions_parametric(
        x_t, n_t, x_c, n_c, n_boot=cfg.n_boot, seed=cfg.seed, ci_level=cfg.ci_level
    )

    t_stat, p_t = stats.ttest_ind(control[COL_TOTAL_ADS], test[COL_TOTAL_ADS], equal_var=False)
    results["t_test_total_ads"] = {"t_statistic": float(t_stat), "p_value": float(p_t)}
    results["bootstrap_total_ads_diff"] = bootstrap_mean_diff(
        control[COL_TOTAL_ADS].values,
        test[COL_TOTAL_ADS].values,
        n_boot=min(cfg.n_boot, 2000),
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

    conv_sig = results["p_adjusted"]["significant"]["conversion_ztest"]
    design_ok = bool(results["srm"].get("passes", False))
    results["recommendation"] = {
        "rollout": bool(design_ok and conv_sig and results["ztest_conversion"]["diff"] > 0),
        "reasons": {
            "srm_passes": design_ok,
            "conversion_significant_after_adjustment": bool(conv_sig),
            "conversion_diff": float(results["ztest_conversion"]["diff"]),
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
        plot_conversion_rates(conversion_rates, p4)
        artifacts["viz_bar_conversion_rates"] = os.path.basename(p4)

        p5 = _save_plot_paths(cfg.figures_dir, "viz_box_total_ads.png")
        plot_box_total_ads(control, test, p5)
        artifacts["viz_box_total_ads"] = os.path.basename(p5)

    results["plot_files"] = artifacts

    results["meta"]["runtime_sec"] = float(time.time() - started)

    # записать отчёт
    report_obj = convert_numpy(results)
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    return report_obj
