# Frontend (UI)

UI — витрина отчёта. Делает:
- `POST /analyze?plots=0|1` → получает JSON отчёт
- `GET /artifacts` и `GET /artifacts/:name` → показывает PNG (если plots включён)

---

## Запуск (Windows PowerShell)

### Требования
- Backend запущен и доступен (по умолчанию `http://localhost:8080`)

Проверка:
```powershell
irm http://localhost:8080/health
````

### 1) Установка

```powershell
cd frontend
npm install
```

### 2) Запуск

```powershell
npm run dev
```

Открой:
`http://localhost:5173`

---

## Настройка API base

По умолчанию UI ходит на `http://localhost:8080`.

Если нужно изменить — создай `frontend/.env.local`:

```bash
VITE_API_BASE=http://localhost:8080
```

---

## Если нужны графики

Выбери `plots = 1` в UI (или `/analyze?plots=1`). Тогда backend сохранит PNG, и UI подтянет их через `/artifacts`.
