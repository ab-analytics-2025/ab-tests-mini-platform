import type { AnalyzeRequest, ABReport, ArtifactList } from "./types";

function normalizeBaseUrl(raw: string): string {
  // allow "/api" proxy style or absolute URL
  if (!raw) return "";
  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

export const API_BASE = normalizeBaseUrl(import.meta.env.VITE_API_BASE ?? "http://localhost:8080");

async function httpJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const msg =
      (data && (data.details || data.error || data.message)) ||
      `HTTP ${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data as T;
}

export async function analyze(req: AnalyzeRequest): Promise<ABReport> {
  const plots = req.plots ? 1 : 0;
  const url = `${API_BASE}/analyze?plots=${plots}`;

  const body: any = {};
  const put = (k: string, v: any) => {
    if (v === undefined || v === null || (typeof v === "string" && v.trim() === "")) return;
    body[k] = v;
  };

  put("seed", req.seed);
  put("nBoot", req.nBoot);
  put("alpha", req.alpha);
  put("ciLevel", req.ciLevel);

  put("expectedSplit", req.expectedSplit);
  put("srmUniformTol", req.srmUniformTol);
  put("balanceSmdThreshold", req.balanceSmdThreshold);
  put("balanceCramervThreshold", req.balanceCramervThreshold);
  put("minUpliftAbs", req.minUpliftAbs);

  return httpJson<ABReport>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function listArtifacts(): Promise<ArtifactList> {
  return httpJson<ArtifactList>(`${API_BASE}/artifacts`);
}

export function artifactUrl(name: string): string {
  return `${API_BASE}/artifacts/${encodeURIComponent(name)}`;
}

export async function health(): Promise<{ status: string }> {
  return httpJson<{ status: string }>(`${API_BASE}/health`);
}
