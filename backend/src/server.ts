import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";

import { loadConfig } from "./config";
import { runAnalysis } from "./ab_pipeline";

const cfg = loadConfig();

const app = express();
app.use(express.json({ limit: "1mb" }));

app.use(
  cors({
    origin: cfg.corsOrigin,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

let isRunning = false;

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// POST /analyze?plots=1
app.post("/analyze", async (req, res) => {
  if (isRunning) return res.status(409).json({ error: "Analysis already running" });

  const plots = String(req.query.plots ?? "1") === "1";

  const seed = req.body?.seed;
  const nBoot = req.body?.nBoot;
  const alpha = req.body?.alpha;
  const ciLevel = req.body?.ciLevel;

  try {
    isRunning = true;
    const report = await runAnalysis(cfg, {
      withPlots: plots,
      seed,
      nBoot,
      alpha,
      ciLevel,
    });
    res.json(report);
  } catch (e: any) {
    res.status(500).json({ error: "Analyze failed", details: String(e?.message ?? e) });
  } finally {
    isRunning = false;
  }
});

app.get("/artifacts/:name", (req, res) => {
  const name = path.basename(req.params.name);
  const filePath = path.join(cfg.figuresDir, name);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: "Artifact not found", name });
  }
  res.sendFile(filePath);
});

// список артефактов
app.get("/artifacts", (_req, res) => {
  if (!fs.existsSync(cfg.figuresDir)) return res.json({ files: [] });
  const files = fs.readdirSync(cfg.figuresDir).filter((f) => f.toLowerCase().endsWith(".png"));
  res.json({ files });
});

app.listen(cfg.port, () => {
  console.log(`Backend: http://localhost:${cfg.port}`);
  console.log(`Health:  http://localhost:${cfg.port}/health`);
  console.log(`Analyze: POST http://localhost:${cfg.port}/analyze?plots=1`);
  console.log(`Artifacts: http://localhost:${cfg.port}/artifacts`);
});
