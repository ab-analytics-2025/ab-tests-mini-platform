# Data README — `marketing_AB.csv`

Источник данных: Kaggle dataset **faviovaz/marketing-ab-testing** (файл `marketing_AB.csv`).  
Назначение: учебный датасет для демонстрации A/B‑анализа “Ads vs PSA”.

---

## Схема (data dictionary)

| Колонка | Тип (ожидаемо) | Описание |
|---|---:|---|
| `user id` | int | Уникальный идентификатор пользователя (единица рандомизации). |
| `test group` | str | Группа эксперимента: `ad` (treatment) или `psa` (control). |
| `converted` | bool / str / int | Конверсия: True/False или 1/0. В пайплайне приводится к 0/1. |
| `total ads` | int | Сколько рекламных показов увидел пользователь (в сумме). |
| `most ads day` | str | День недели, когда пользователь видел максимальное число показов. |
| `most ads hour` | int | Час (0..23), когда пользователь видел максимальное число показов. |

Ожидаемые домены:
- `test group` ∈ {`ad`, `psa`}
- `converted` ∈ {0,1} (после очистки)
- `most ads day` ∈ {Monday..Sunday}
- `most ads hour` ∈ [0..23]
- `total ads` ≥ 0

---

## Правила очистки (pipeline contract)

Пайплайн делает следующую очистку и приводит данные к контракту:

1. Удаление технической колонки `Unnamed: 0` (если есть).
2. Приведение `converted` к бинарному `0/1`.
3. Дедупликация по `user id` (оставляем **первую** строку на пользователя).
4. Проверки качества:
   - обязательные колонки присутствуют
   - `user id` уникален
   - нет пропусков в ключевых колонках
   - домены/диапазоны соблюдены (`hour`, `total ads`)

Результат: `data/cleaned/marketing_AB_clean.csv`.

---

## Ограничения и допущения

1. **Неизвестный дизайн аллокации (split)**  
   В датасете доли групп сильно не 50/50. Для осмысленной проверки SRM нужен параметр `expected_split`.
   - Если `expected_split` не указан, SRM будет `skipped` при неравном сплите.

2. **Ковариаты для баланса ограничены**  
   В данных нет “канала/страны/сегмента”, поэтому баланс проверяется только по доступным признакам:
   `total ads`, `most ads hour`, `most ads day`.

3. **Exposure covariates**  
   `total ads` и “most ads …” могут зависеть от активности пользователя, поэтому balance оцениваем по **effect size**,
   а p-value оставляем как справочную статистику.

---

## Лицензия данных

По данным об источнике Kaggle датасета `faviovaz/marketing-ab-testing`, лицензия указана как **CC0-1.0** (public domain dedication).

---

## Быстрая проверка целостности (data-quality)

Секция `data_quality` формируется в `reports/report.json`.

Мини‑проверка руками:
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/marketing_AB.csv'); print(df.columns.tolist()); print(df.isna().sum().to_dict())"
```
