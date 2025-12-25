import argparse
import json

from .pipeline import run_pipeline


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

    p.add_argument(
        "--stdout",
        choices=["none", "json", "summary"],
        default="none",
        help="What to print to stdout: none (default), json (full report), summary (short text).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

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
            f"- SRM passes: {srm.get('passes')}, p={srm.get('p_value')}\n"
            f"- conv diff: {z.get('diff')}, p={z.get('p_value')}\n"
            f"- rollout: {rec.get('rollout')}\n"
            f"- report: {meta.get('report_path')}\n"
        )

    return 0
