import json
from pathlib import Path

from ab_platform.pipeline import run_pipeline


def test_pipeline_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    input_csv = root / "data" / "raw" / "marketing_AB.csv"

    cleaned = tmp_path / "clean.csv"
    report = tmp_path / "report.json"
    figures = tmp_path / "figures"

    res = run_pipeline(
        input_path=str(input_csv),
        cleaned_path=str(cleaned),
        report_path=str(report),
        figures_dir=str(figures),
        seed=1,
        n_boot=300,
        alpha=0.05,
        ci_level=0.95,
        write_plots=False,
    )

    assert cleaned.exists()
    assert report.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    for key in ("meta", "data_quality", "srm", "balance_checks", "ztest_conversion", "mde_power", "p_adjusted", "recommendation"):
        assert key in data

    # return value == report object
    assert "meta" in res
