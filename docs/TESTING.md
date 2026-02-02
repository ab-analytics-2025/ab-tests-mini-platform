# TESTING

Цель тестирования: гарантировать, что пайплайн
- корректно чистит данные,
- корректно считает SRM/balance,
- корректно считает статистику и формирует отчёт,
- корректно работает через API.

---

## Пирамида тестов (как реализовано)

### 1) Unit‑тесты (ядро логики)
Тестируем “чистые” функции без I/O:

- SRM: логика `expected_split` vs `skipped`
- balance: SMD, Cramér’s V, thresholds
- статистика: z‑test, bootstrap, MDE, Holm‑adjust

Файлы:
- `tests/test_unit_srm_balance.py`
- `tests/test_unit_stats.py`

### 2) Integration: pipeline smoke
Запуск `run_pipeline(...)` на реальном CSV:

- создаёт cleaned CSV + report.json
- `report.json` содержит ключевые секции

Файл:
- `tests/test_smoke_pipeline.py`

### 3) E2E: API smoke (ручной)
Поднимаем backend и проверяем:

- `GET /health` → ok
- `POST /analyze` → JSON отчёт (200)
- `GET /artifacts`/`GET /artifacts/:name` → PNG (если plots включены)

---

## Как запускать

### Python tests
```powershell
pytest -q
```

```powershell
pytest -q --cov=ab_platform --cov-report=term-missing
```

### Pipeline smoke (CLI)
```powershell
python -m ab_platform `
  --input data/raw/marketing_AB.csv `
  --cleaned data/cleaned/marketing_AB_clean.csv `
  --report reports/report.json `
  --figures reports/figures `
  --expected-split "ad=0.96,psa=0.04" `
  --stdout summary
```

### Backend smoke
```powershell
cd backend
npm ci
npm run dev
irm http://localhost:8080/health
irm -Method Post "http://localhost:8080/analyze?plots=0" -ContentType "application/json" -Body "{}"
```

---

## Красные флаги (что считать ошибкой)

### Данные
- `user id` не уникален после очистки
- `converted` не бинарный 0/1 после очистки
- `most ads hour` вне [0..23]
- `total ads` отрицательный
- пропуски в обязательных колонках

### Дизайн
- SRM = `fail` при корректно заданном `expected_split`
- SRM = `pass/fail` при неравном сплите без `expected_split` (должно быть `skipped`)
- Balance “fails” только из‑за p-value при микроскопических effect sizes (gating должен быть по effect size)

### Статистика
- CI перевёрнуты (low > high)
- uplift/разница долей не совпадает по знаку между z‑test и bootstrap
- Holm‑adjust даёт p_adj меньше исходного p

### API
- `POST /analyze` не создаёт/не читает `reports/report.json`
- backend падает без понятной ошибки при неверных `PROJECT_ROOT`/`PYTHON_CMD`
