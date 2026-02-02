# A/B Tests Mini-Platform

Мини-платформа для **воспроизводимого** анализа A/B‑эксперимента: от сырого CSV → очистка → проверки дизайна (SRM + balance) → статистический вывод (CI/MDE) → отчёт + графики → API → UI.

Датасет: `data/raw/marketing_AB.csv` (Ads vs PSA).

---

## Что умеет (по требованиям задачи)

- **Очистка данных**: приведение типов, `converted` → 0/1, дедупликация по `user id`, базовые проверки качества.
- **Валидация дизайна**:
  - **SRM** (χ²): относительно `expected_split` (если задан) или `skipped`, если сплит явно неравный и `expected_split` не указан.
  - **Balance** по ковариатам: числовые (SMD + KS), категориальные (Cramér’s V + χ²), gating по **effect size**.
- **Статистика**:
  - z‑test / χ² для долей + 95% ДИ,
  - bootstrap ДИ для uplift,
  - оценка **MDE/мощности**,
  - корректировка множественных сравнений (Holm‑Bonferroni).
- **Артефакты**: `data/cleaned/*.csv`, `reports/report.json`, `reports/figures/*.png`.
- **Backend API**: запускает пайплайн и отдаёт отчёт/артефакты.
- **Frontend UI**: витрина отчёта (в разработке, следующий шаг).

---

## Структура проекта

| Папка / файл | Назначение |
|---|---|
| `data/raw/` | сырые данные |
| `data/cleaned/` | очищенные данные (output пайплайна) |
| `src/ab_platform/` | Python пайплайн (CLI + расчёты + отчёт) |
| `tests/` | unit + pipeline smoke |
| `docs/` | ARCHITECTURE / DATA_README / TESTING / ADR |
| `reports/` | `report.json` и `figures/*.png` |
| `backend/` | Node/TS API сервис |
| `frontend/` | UI |

---

## Установка

Рекомендуется виртуальное окружение:

```bash
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Linux/Mac:
# source venv/bin/activate
```

Зависимости:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

---

## Запуск пайплайна (CLI)

### Windows PowerShell

```powershell
python -m ab_platform `
  --input data/raw/marketing_AB.csv `
  --cleaned data/cleaned/marketing_AB_clean.csv `
  --report reports/report.json `
  --expected-split "ad=0.96,psa=0.04" `
  --stdout summary
```

Графики (PNG):

```powershell
python -m ab_platform `
  --input data/raw/marketing_AB.csv `
  --cleaned data/cleaned/marketing_AB_clean.csv `
  --report reports/report.json `
  --figures reports/figures `
  --expected-split "ad=0.96,psa=0.04" `
  --stdout summary
```

### Linux/Mac

```bash
python -m ab_platform   --input data/raw/marketing_AB.csv   --cleaned data/cleaned/marketing_AB_clean.csv   --report reports/report.json   --figures reports/figures   --expected-split "ad=0.96,psa=0.04"   --stdout summary
```

---

## Backend API

### Эндпоинты

- `GET /health` → `{ "status": "ok" }`
- `POST /analyze?plots=0|1` → JSON отчёт
- `GET /artifacts` → список PNG
- `GET /artifacts/:name` → конкретный PNG

### Переменные окружения

- `PROJECT_ROOT` — корень репозитория
- `PYTHON_CMD` — путь к python (например, `...\venv\Scripts\python.exe`)
- (опционально) `AB_EXPECTED_SPLIT` — дефолтный `expected_split`

Также можно переопределить пути артефактов:
`AB_INPUT`, `AB_CLEANED`, `AB_REPORT`, `AB_FIGURES`.

### Запуск

```powershell
cd backend
npm ci
$env:PROJECT_ROOT="B:\Polytech\Maga\Shabalin\ab-tests-mini-platform"
$env:PYTHON_CMD="B:\Polytech\Maga\Shabalin\ab-tests-mini-platform\venv\Scripts\python.exe"
$env:AB_EXPECTED_SPLIT="ad=0.96,psa=0.04"
npm run dev
```

Smoke:

```powershell
irm http://localhost:8080/health
irm -Method Post "http://localhost:8080/analyze?plots=0" -ContentType "application/json" -Body '{"minUpliftAbs":0.001}'
```

---

## Тестирование
```powershell
pytest -q
```

```powershell
pytest -q --cov=ab_platform --cov-report=term-missing
```


---

## Документация

- `docs/ARCHITECTURE.md` — потоки данных + где лежат артефакты
- `docs/DATA_README.md` — словарь данных + правила очистки
- `docs/TESTING.md` — стратегия тестирования + “красные флаги”
- `docs/ADR/` — ключевые архитектурные решения

---

## Команда и роли

| № | Роль | Основные обязанности | Участник |
|---:|---|---|---|
| 1 | Менеджер проекта | Планирование, доска задач, контроль сроков, статус‑репорты по проекту | _(ФИО / GitHub-ник)_ |
| 2 | Tech Lead | Архитектура, качество кода, ревью | _(ФИО / GitHub-ник)_ |
| 3 | Data Engineer | Загрузка/очистка/контракт cleaned данных | _(ФИО / GitHub-ник)_ |
| 4 | Statistician | SRM/balance, тесты, CI/MDE, интерпретация | _(ФИО / GitHub-ник)_ |
| 5 | Data Analyst | Визуализации, выводы, витрина результатов | _Микехин Никита Вячеславович / FireNike2002_ |
| 6 | QA | Unit + data-quality тесты, “красные флаги” | _(ФИО / GitHub-ник)_ |
| 7 | DevOps | Воспроизводимость, CI, окружение | _(ФИО / GitHub-ник)_ |
