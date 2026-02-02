import argparse
import json
from typing import Dict, Optional

from .pipeline import run_pipeline


def _parse_expected_split(raw: str) -> Dict[str, float]:
    """Парсит строку формата: 'ad=0.96,psa=0.04' (разделители ',' или ';')."""
    s = raw.strip()
    if not s:
        raise ValueError("expected_split is empty")

    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: Dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid expected_split part: '{part}', expected 'group=prob'")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid expected_split part (empty group): '{part}'")
        try:
            fv = float(v)
        except ValueError as e:
            raise ValueError(f"Invalid expected_split value for '{k}': '{v}'") from e
        out[k] = fv
    if not out:
        raise ValueError("expected_split has no parsed pairs")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ab_platform")
    p.add_argument("--input", required=True)
    p.add_argument("--cleaned", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--figures", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--no-plots", action="store_true")

    # Step 1 (контракт эксперимента)
    p.add_argument(
        "--expected-split",
        default=None,
        help="Expected group split for SRM, e.g. 'ad=0.96,psa=0.04'. "
        "If not provided and observed split is far from uniform, SRM will be skipped.",
    )
    p.add_argument(
        "--srm-uniform-tol",
        type=float,
        default=0.02,
        help="Tolerance to assume uniform split when expected_split is not provided (default 0.02 = 2%).",
    )
    p.add_argument(
        "--balance-smd-threshold",
        type=float,
        default=0.10,
        help="Effect size threshold for numeric balance checks (SMD abs max).",
    )
    p.add_argument(
        "--balance-cramerv-threshold",
        type=float,
        default=0.10,
        help="Effect size threshold for categorical balance checks (Cramér's V max).",
    )
    p.add_argument(
        "--min-uplift-abs",
        type=float,
        default=0.0,
        help="Practical significance threshold for conversion uplift (absolute, treatment-control).",
    )

    p.add_argument(
        "--stdout",
        choices=["none", "json", "summary"],
        default="none",
        help="What to print to stdout: none (default), json (full report), summary (short text).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    expected_split: Optional[Dict[str, float]] = None
    if args.expected_split is not None:
        expected_split = _parse_expected_split(args.expected_split)

    report = run_pipeline(
        input_path=args.input,
        cleaned_path=args.cleaned,
        report_path=args.report,
        figures_dir=args.figures,
        seed=args.seed,
        n_boot=args.n_boot,
        alpha=args.alpha,
        ci_level=args.ci_level,
        write_plots=not args.no_plots,
        expected_split=expected_split,
        srm_uniform_tol=args.srm_uniform_tol,
        balance_smd_threshold=args.balance_smd_threshold,
        balance_cramersv_threshold=args.balance_cramerv_threshold,
        min_uplift_abs=args.min_uplift_abs,
    )

    if args.stdout == "json":
        print(json.dumps(report, ensure_ascii=False))
    elif args.stdout == "summary":
        meta = report.get("meta", {})
        srm = report.get("srm", {})
        z = report.get("ztest_conversion", {})
        rec = report.get("recommendation", {})
        print(
            "AB report summary\n"
            f"- runtime_sec: {meta.get('runtime_sec')}\n"
            f"- groups: {report.get('test_group_counts')}\n"
            f"- SRM status: {srm.get('status')}, passes: {srm.get('passes')}, p={srm.get('p_value')}\n"
            f"- conv diff: {z.get('diff')}, p={z.get('p_value')}\n"
            f"- decision: {rec.get('decision')} (rollout={rec.get('rollout')})\n"
            f"- report: {meta.get('report_path')}\n"
        )

    return 0
