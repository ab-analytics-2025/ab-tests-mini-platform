import path from "path";

function envInt(name: string, def: number): number {
  const v = process.env[name];
  if (!v) return def;
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
}

function envStr(name: string, def: string): string {
  return process.env[name] ?? def;
}

export type AppConfig = {
  port: number;
  corsOrigin: string;
  projectRoot: string;

  pythonCmd: string;
  pythonCmdArgs: string[];

  inputPath: string;
  cleanedPath: string;
  reportPath: string;
  figuresDir: string;
};

function splitCmd(cmd: string): { bin: string; args: string[] } {
  const parts = cmd.trim().split(/\s+/).filter(Boolean);
  return { bin: parts[0] ?? "python", args: parts.slice(1) };
}

export function loadConfig(): AppConfig {
  const projectRoot = path.resolve(envStr("PROJECT_ROOT", path.resolve(process.cwd(), "..")));

  const pythonCmdRaw = envStr("PYTHON_CMD", "python");
  const { bin, args } = splitCmd(pythonCmdRaw);

  const inputPath = envStr("AB_INPUT", path.join(projectRoot, "data", "raw", "marketing_AB.csv"));
  const cleanedPath = envStr("AB_CLEANED", path.join(projectRoot, "data", "cleaned", "marketing_AB_clean.csv"));
  const reportPath = envStr("AB_REPORT", path.join(projectRoot, "reports", "report.json"));
  const figuresDir = envStr("AB_FIGURES", path.join(projectRoot, "reports", "figures"));

  return {
    port: envInt("PORT", 3000),
    corsOrigin: envStr("CORS_ORIGIN", "http://localhost:5173"),
    projectRoot,

    pythonCmd: bin,
    pythonCmdArgs: args,

    inputPath,
    cleanedPath,
    reportPath,
    figuresDir,
  };
}
