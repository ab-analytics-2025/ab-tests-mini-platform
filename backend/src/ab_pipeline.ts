import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import type { AppConfig } from "./config";

export type AnalyzeOptions = {
  withPlots: boolean;
  seed?: number;
  nBoot?: number;
  alpha?: number;
  ciLevel?: number;

  // Step 1
  expectedSplit?: string; // e.g. "ad=0.96,psa=0.04"
  srmUniformTol?: number;
  balanceSmdThreshold?: number;
  balanceCramervThreshold?: number;
  minUpliftAbs?: number;
};

function readJsonFile(filePath: string): any {
  const raw = fs.readFileSync(filePath, { encoding: "utf-8" });
  return JSON.parse(raw);
}

function ensureDir(p: string): void {
  fs.mkdirSync(p, { recursive: true });
}

export async function runAnalysis(cfg: AppConfig, opts: AnalyzeOptions): Promise<any> {
  ensureDir(path.dirname(cfg.reportPath));
  ensureDir(path.dirname(cfg.cleanedPath));
  ensureDir(cfg.figuresDir);

  const args: string[] = [
    ...cfg.pythonCmdArgs,
    "-m",
    "ab_platform",
    "--input",
    cfg.inputPath,
    "--cleaned",
    cfg.cleanedPath,
    "--report",
    cfg.reportPath,
    "--figures",
    cfg.figuresDir,
    "--stdout",
    "none",
  ];

  if (!opts.withPlots) args.push("--no-plots");

  if (opts.seed !== undefined) args.push("--seed", String(opts.seed));
  if (opts.nBoot !== undefined) args.push("--n-boot", String(opts.nBoot));
  if (opts.alpha !== undefined) args.push("--alpha", String(opts.alpha));
  if (opts.ciLevel !== undefined) args.push("--ci-level", String(opts.ciLevel));

  // Step 1 flags
  if (opts.expectedSplit !== undefined) args.push("--expected-split", String(opts.expectedSplit));
  if (opts.srmUniformTol !== undefined) args.push("--srm-uniform-tol", String(opts.srmUniformTol));
  if (opts.balanceSmdThreshold !== undefined) args.push("--balance-smd-threshold", String(opts.balanceSmdThreshold));
  if (opts.balanceCramervThreshold !== undefined)
    args.push("--balance-cramerv-threshold", String(opts.balanceCramervThreshold));
  if (opts.minUpliftAbs !== undefined) args.push("--min-uplift-abs", String(opts.minUpliftAbs));

  const code: number = await new Promise((resolve, reject) => {
    const p = spawn(cfg.pythonCmd, args, {
      cwd: cfg.projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });

    let stderr = "";
    p.stderr.on("data", (d) => (stderr += d.toString()));

    p.on("error", reject);
    p.on("close", (c) => {
      if (c !== 0) {
        reject(new Error(`Pipeline failed (code=${c}). ${stderr}`));
        return;
      }
      resolve(c ?? 0);
    });
  });

  void code;

  if (!fs.existsSync(cfg.reportPath)) {
    throw new Error(`Report not found: ${cfg.reportPath}`);
  }

  return readJsonFile(cfg.reportPath);
}
