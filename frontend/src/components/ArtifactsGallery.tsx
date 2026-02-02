import { useEffect, useState } from "react";
import { artifactUrl, listArtifacts } from "../api";

type Props = {
  enabled: boolean;
  refreshKey: number;
};

export default function ArtifactsGallery({ enabled, refreshKey }: Props) {
  const [files, setFiles] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);

    listArtifacts()
      .then((res) => {
        if (cancelled) return;
        setFiles(res.files ?? []);
      })
      .catch((e: any) => {
        if (cancelled) return;
        setErr(String(e?.message ?? e));
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [enabled, refreshKey]);

  if (!enabled) {
    return (
      <div className="card">
        <h2>Графики</h2>
        <div className="small">Сейчас plots выключены. Включи “plots=1”, чтобы backend сохранил PNG в `reports/figures/`.</div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Графики</h2>
      {loading ? <div className="small">Загрузка списка артефактов…</div> : null}
      {err ? <div className="error">Ошибка: {err}</div> : null}
      {!loading && !err && files.length === 0 ? (
        <div className="notice">
          PNG не найдены. Проверь:
          <ul style={{ margin: "8px 0 0 18px" }}>
            <li>в запросе `/analyze?plots=1`</li>
            <li>backend пишет в `reports/figures/`</li>
          </ul>
        </div>
      ) : null}

      <div className="gallery" style={{ marginTop: 10 }}>
        {files.map((f) => (
          <div key={f} className="imgWrap">
            <img src={artifactUrl(f)} alt={f} />
            <div className="imgCaption">{f}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
