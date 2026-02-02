import KeyValueTable from "./KeyValueTable";
import type { ABReport } from "../types";

type Props = {
  report: ABReport;
};

export default function ReportSummary({ report }: Props) {
  const meta = report?.meta ?? {};
  const rec = report?.recommendation ?? {};
  const srm = report?.srm ?? {};
  const z = report?.ztest_conversion ?? {};
  const mde = report?.mde_power ?? {};
  const counts = report?.test_group_counts ?? {};

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <KeyValueTable
        title="Рекомендация"
        items={[
          ["decision", rec.decision],
          ["rollout", rec.rollout],
          ["reasons.srm_status", rec?.reasons?.srm_status],
          ["reasons.needs_expected_split", rec?.reasons?.needs_expected_split],
          ["reasons.balance_passes_effect_size", rec?.reasons?.balance_passes_effect_size],
          ["reasons.conversion_significant_after_adjustment", rec?.reasons?.conversion_significant_after_adjustment],
          ["reasons.conversion_diff", rec?.reasons?.conversion_diff],
          ["reasons.min_uplift_abs", rec?.reasons?.min_uplift_abs],
        ]}
      />

      <KeyValueTable
        title="Группы"
        items={[
          ["counts", counts],
          ["control_label", meta.control_label],
          ["treatment_label", meta.treatment_label],
          ["control_conversion_rate", report?.control_conversion_rate],
          ["test_conversion_rate", report?.test_conversion_rate],
        ]}
      />

      <KeyValueTable
        title="SRM"
        items={[
          ["status", srm.status],
          ["passes", srm.passes],
          ["p_value", srm.p_value],
          ["chi2", srm.chi2],
          ["expected_provided", srm.expected_provided],
          ["suggested_expected_split", srm.suggested_expected_split],
        ]}
      />

      <KeyValueTable
        title="Conversion z-test"
        items={[
          ["diff", z.diff],
          ["p_value", z.p_value],
          ["ci", z.ci],
          ["p1 (treatment)", z.p1],
          ["p2 (control)", z.p2],
          ["z", z.z],
        ]}
      />

      <KeyValueTable
        title="MDE / Power"
        items={[
          ["power_target", mde.power_target],
          ["mde_abs", mde.mde_abs],
          ["baseline", mde.baseline],
          ["alpha", mde.alpha],
        ]}
      />

      <KeyValueTable
        title="Run meta"
        items={[
          ["report_version", meta.report_version],
          ["runtime_sec", meta.runtime_sec],
          ["seed", meta.seed],
          ["n_boot", meta.n_boot],
          ["alpha", meta.alpha],
          ["ci_level", meta.ci_level],
          ["min_uplift_abs", meta.min_uplift_abs],
          ["expected_split", meta.expected_split],
        ]}
      />
    </div>
  );
}
