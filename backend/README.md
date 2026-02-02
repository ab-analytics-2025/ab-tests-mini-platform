# Backend (API)

Backend запускает Python пайплайн и отдаёт результат как JSON + PNG артефакты.

## Эндпоинты
- `GET /health` → `{ "status": "ok" }`
- `POST /analyze?plots=0|1` → JSON отчёт
- `GET /artifacts` → `{ "files": ["plot1.png", ...] }`
- `GET /artifacts/:name` → PNG файл

---

## Запуск (Windows PowerShell)

### 1) Установка
```powershell
cd backend
npm install
````

### 2) Переменные окружения

Backend должен знать корень проекта и python из venv:

```powershell
$env:PROJECT_ROOT = (Resolve-Path ..).Path
$env:PYTHON_CMD = (Resolve-Path ..\venv\Scripts\python.exe).Path
```

Опционально:

```powershell
$env:AB_EXPECTED_SPLIT = "ad=0.96,psa=0.04"
# Переопределение путей (если нужно)
# $env:AB_INPUT   = "...\data\raw\marketing_AB.csv"
# $env:AB_CLEANED = "...\data\cleaned\marketing_AB_clean.csv"
# $env:AB_REPORT  = "...\reports\report.json"
# $env:AB_FIGURES = "...\reports\figures"
```

### 3) Запуск

```powershell
npm run dev
```

---

## Проверка

```powershell
irm http://localhost:8080/health

irm -Method Post "http://localhost:8080/analyze?plots=0" `
  -ContentType "application/json" `
  -Body '{"minUpliftAbs":0.001, "expectedSplit":"ad=0.96,psa=0.04"}'

irm http://localhost:8080/artifacts
```

---

## Параметры запроса /analyze

Поддерживаются поля:
`seed, nBoot, alpha, ciLevel, expectedSplit, srmUniformTol, balanceSmdThreshold, balanceCramervThreshold, minUpliftAbs`
