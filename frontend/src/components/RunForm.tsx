import { useMemo, useState } from "react";
import type { AnalyzeRequest } from "../types";
import { API_BASE } from "../api";

type Props = {
  disabled: boolean;
  onRun: (req: AnalyzeRequest) => void;
};

function toNum(v: string): number | undefined {
  const s = v.trim();
  if (!s) return undefined;
  const n = Number(s.replace(",", "."));
  return Number.isFinite(n) ? n : undefined;
}

export default function RunForm({ disabled, onRun }: Props) {
  const [plots, setPlots] = useState(true);

  const [seed, setSeed] = useState("42");
  const [nBoot, setNBoot] = useState("5000");
  const [alpha, setAlpha] = useState("0.05");
  const [ciLevel, setCiLevel] = useState("0.95");

  const [expectedSplit, setExpectedSplit] = useState("ad=0.96,psa=0.04");
  const [srmUniformTol, setSrmUniformTol] = useState("0.02");
  const [balanceSmdThreshold, setBalanceSmdThreshold] = useState("0.10");
  const [balanceCramervThreshold, setBalanceCramervThreshold] = useState("0.10");
  const [minUpliftAbs, setMinUpliftAbs] = useState("0.001");

  const req: AnalyzeRequest = useMemo(
    () => ({
      plots,
      seed: toNum(seed),
      nBoot: toNum(nBoot),
      alpha: toNum(alpha),
      ciLevel: toNum(ciLevel),
      expectedSplit: expectedSplit.trim() || undefined,
      srmUniformTol: toNum(srmUniformTol),
      balanceSmdThreshold: toNum(balanceSmdThreshold),
      balanceCramervThreshold: toNum(balanceCramervThreshold),
      minUpliftAbs: toNum(minUpliftAbs),
    }),
    [plots, seed, nBoot, alpha, ciLevel, expectedSplit, srmUniformTol, balanceSmdThreshold, balanceCramervThreshold, minUpliftAbs]
  );

  return (
    <div className="card">
      <h2>Запуск анализа</h2>

      <div className="label">API base</div>
      <input className="input" value={API_BASE} readOnly />

      <div className="label">plots</div>
      <select className="select" value={plots ? "1" : "0"} onChange={(e) => setPlots(e.target.value === "1")}>
        <option value="1">1 — сохранять PNG</option>
        <option value="0">0 — без графиков</option>
      </select>

      <div className="hr" />

      <div className="label">expectedSplit (для SRM)</div>
      <input className="input" value={expectedSplit} onChange={(e) => setExpectedSplit(e.target.value)} placeholder='ad=0.96,psa=0.04' />

      <div className="row">
        <div style={{ flex: 1 }}>
          <div className="label">minUpliftAbs</div>
          <input className="input" value={minUpliftAbs} onChange={(e) => setMinUpliftAbs(e.target.value)} placeholder="0.001" />
        </div>
        <div style={{ flex: 1 }}>
          <div className="label">srmUniformTol</div>
          <input className="input" value={srmUniformTol} onChange={(e) => setSrmUniformTol(e.target.value)} placeholder="0.02" />
        </div>
      </div>

      <div className="row">
        <div style={{ flex: 1 }}>
          <div className="label">balanceSmdThreshold</div>
          <input className="input" value={balanceSmdThreshold} onChange={(e) => setBalanceSmdThreshold(e.target.value)} placeholder="0.10" />
        </div>
        <div style={{ flex: 1 }}>
          <div className="label">balanceCramervThreshold</div>
          <input className="input" value={balanceCramervThreshold} onChange={(e) => setBalanceCramervThreshold(e.target.value)} placeholder="0.10" />
        </div>
      </div>

      <div className="hr" />

      <div className="row">
        <div style={{ flex: 1 }}>
          <div className="label">seed</div>
          <input className="input" value={seed} onChange={(e) => setSeed(e.target.value)} />
        </div>
        <div style={{ flex: 1 }}>
          <div className="label">nBoot</div>
          <input className="input" value={nBoot} onChange={(e) => setNBoot(e.target.value)} />
        </div>
      </div>

      <div className="row">
        <div style={{ flex: 1 }}>
          <div className="label">alpha</div>
          <input className="input" value={alpha} onChange={(e) => setAlpha(e.target.value)} />
        </div>
        <div style={{ flex: 1 }}>
          <div className="label">ciLevel</div>
          <input className="input" value={ciLevel} onChange={(e) => setCiLevel(e.target.value)} />
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <button className="btn" disabled={disabled} onClick={() => onRun(req)}>
          {disabled ? "Запуск…" : "Run analysis"}
        </button>
      </div>

      <div className="small">
        Backend API ожидает:
        <ul style={{ margin: "8px 0 0 18px" }}>
          <li><code>POST /analyze?plots=0|1</code></li>
          <li>body: <code>{"{ seed, nBoot, alpha, ciLevel, expectedSplit, srmUniformTol, balanceSmdThreshold, balanceCramervThreshold, minUpliftAbs }"}</code></li>
          <li>графики доступны на <code>GET /artifacts</code> и <code>GET /artifacts/:name</code></li>
        </ul>
      </div>
    </div>
  );
}
