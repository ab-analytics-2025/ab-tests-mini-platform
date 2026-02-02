type Props = {
  title?: string;
  items: Array<[string, any]>;
};

function fmt(v: any): string {
  if (v === undefined) return "—";
  if (v === null) return "null";
  if (typeof v === "number") {
    const abs = Math.abs(v);
    if (abs !== 0 && abs < 1e-6) return v.toExponential(3);
    return String(v);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return JSON.stringify(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export default function KeyValueTable({ title, items }: Props) {
  return (
    <div className="card">
      {title ? <h2>{title}</h2> : null}
      <div className="kv">
        {items.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <div className="k">{k}</div>
            <div className="v">{fmt(v)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
