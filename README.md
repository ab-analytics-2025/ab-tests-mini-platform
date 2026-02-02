# A/B Tests Mini-Platform

Мини-платформа для воспроизводимого анализа A/B-эксперимента:

**CSV → clean → SRM + balance → статистика + CI/MDE → report.json + PNG → API → UI**

Датасет: `data/raw/marketing_AB.csv` (группы `ad` vs `psa`, метрика `converted`).

---

## Структура

- `data/raw/` — исходные данные
- `data/cleaned/` — очищенные данные (output пайплайна)
- `src/ab_platform/` — Python пайплайн (CLI, проверки, статистика, отчёт)
- `reports/` — `report.json`, `figures/*.png`
- `backend/` — Node/TS API (запуск пайплайна + отдача артефактов)
- `frontend/` — UI витрина результатов
- `tests/` — unit + smoke
- `docs/` — ARCHITECTURE/DATA_README/TESTING + ADR

---

## Быстрый старт (Windows PowerShell)

### 0) Python окружение
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
````

### 1) Запуск пайплайна (CLI)

```powershell
python -m ab_platform `
  --input data/raw/marketing_AB.csv `
  --cleaned data/cleaned/marketing_AB_clean.csv `
  --report reports/report.json `
  --figures reports/figures `
  --expected-split "ad=0.96,psa=0.04" `
  --stdout summary
```

### 2) Тесты

```powershell
pytest -q
```

### 3) Backend (API)

См. `backend/README.md`.

### 4) Frontend (UI)

См. `frontend/README.md`.

---

## Артефакты

* cleaned: `data/cleaned/marketing_AB_clean.csv`
* отчёт: `reports/report.json`
* графики: `reports/figures/*.png` (если включён plots)

---

## Документация

* `docs/ARCHITECTURE.md` — потоки/артефакты + Mermaid
* `docs/DATA_README.md` — словарь данных + правила очистки
* `docs/TESTING.md` — как тестировать + красные флаги
* `docs/ADR/` — ключевые решения (SRM expected_split, balance gating по effect size)

---

