import { useEffect, useState } from "react";
import { analyze, health } from "./api";
import type { ABReport, AnalyzeRequest } from "./types";
import RunForm from "./components/RunForm";
import ReportSummary from "./components/ReportSummary";
import JsonBlock from "./components/JsonBlock";
import ArtifactsGallery from "./components/ArtifactsGallery";

export default function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [backendErr, setBackendErr] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<ABReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [showJson, setShowJson] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastPlotsEnabled, setLastPlotsEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    health()
      .then(() => {
        if (cancelled) return;
        setBackendOk(true);
        setBackendErr(null);
      })
      .catch((e: any) => {
        if (cancelled) return;
        setBackendOk(false);
        setBackendErr(String(e?.message ?? e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function onRun(req: AnalyzeRequest) {
    setRunning(true);
    setError(null);
    try {
      setLastPlotsEnabled(req.plots);
      const r = await analyze(req);
      setReport(r);
      setRefreshKey((k) => k + 1);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  }

  const decision = report?.recommendation?.decision ?? "—";
  const rollout = report?.recommendation?.rollout;

  return (
    <div className="container">
      <div className="header">
        <h1 className="h1">A/B Tests Mini‑Platform</h1>
        <span className="badge">{decision}{rollout === true ? " ✅" : rollout === false ? " ⛔" : ""}</span>
      </div>

      {backendOk === false ? (
        <div className="error" style={{ marginBottom: 12 }}>
          Backend недоступен: {backendErr}
          <div className="small">
            Проверь, что backend запущен и слушает порт (обычно <code>http://localhost:8080</code>).
          </div>
        </div>
      ) : null}

      <div className="grid">
        <div style={{ display: "grid", gap: 12 }}>
          <RunForm disabled={running} onRun={onRun} />

          {error ? <div className="error">Ошибка запуска: {error}</div> : null}

          <div className="card">
            <h2>Отображение</h2>
            <button className="btn" onClick={() => setShowJson((v) => !v)}>
              {showJson ? "Скрыть raw JSON" : "Показать raw JSON"}
            </button>
            <div className="small">
              Полный отчёт возвращается backend‑ом (это содержимое <code>reports/report.json</code>).
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          {report ? (
            <ReportSummary report={report} />
          ) : (
            <div className="card">
              <h2>Отчёт</h2>
              <div className="small">Нажми “Run analysis”, чтобы получить report.json.</div>
            </div>
          )}

          {showJson && report ? <JsonBlock title="Raw report.json" data={report} /> : null}

          <ArtifactsGallery enabled={lastPlotsEnabled} refreshKey={refreshKey} />
        </div>
      </div>
    </div>
  );
}
