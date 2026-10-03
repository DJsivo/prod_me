## Структура проекта

```text
prod_me/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## Практика: Pipeline и GridSearchCV

Решение задания находится в `src/models/titanic_pipeline.py`, кастомный
трансформер — в `src/features/missing_count.py`. Готовый запуск использует
локальный Titanic (`data/external/titanic.csv`), интернет для обучения не нужен.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.models.titanic_pipeline
```

В активированном Python-окружении достаточно `python -m src.models.titanic_pipeline`.
Запускайте команду из корня репозитория. Результаты печатаются в консоль и
сохраняются в `reports/titanic_metrics.json`; разбор — в `reports/titanic_practice.md`.

### Соответствие заданию

- Цель: `survived`. Числовые признаки: `age`, `sibsp`, `parch`, `fare`;
  категориальные: `pclass`, `sex`, `embarked`, `deck`. Класс билета трактуется
  как категория. `alive` дублирует целевую переменную и исключён из входов.
- Стратифицированное разбиение train/test 80/20 с `random_state=42`.
- `MissingCountAdder` добавляет число пропусков в исходной строке до заполнения.
  Этот признак обрабатывается вместе с числовыми колонками.
- `ColumnTransformer`: `SimpleImputer(strategy="median")` и `StandardScaler`
  для числовых колонок; `SimpleImputer(strategy="most_frequent")` и
  `OneHotEncoder(handle_unknown="ignore")` для категориальных.
- Финальная модель — `LogisticRegression`. `GridSearchCV` сравнивает
  `classifier__C=[0.1, 1, 10]` и `classifier__class_weight=[None, "balanced"]`
  по ROC AUC в пяти стратифицированных фолдах с перемешиванием.
- Весь Pipeline обучается внутри CV только на train. Отложенная test-выборка
  используется после выбора параметров; выводятся `best_params_`, accuracy,
  precision, recall, F1, ROC AUC и матрица ошибок.

### Источник данных

CSV взят из [seaborn-data, titanic.csv](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)
3 октября 2026 года. Локальная копия: 891 пассажир, 15 исходных колонок.
SHA-256: `81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2`.
Хеш также сохраняется при запуске для проверки использованных данных.

`data/raw/sample_students.csv.dvc` относится к исходному каркасу проекта;
эта практика не зависит от наличия его DVC-кэша.

### Проверки

```powershell
python -m unittest discover -s tests -v
git diff --check
```

Тесты проверяют подсчёт пропусков и сохранность исходной таблицы,
совместимость трансформера с клонированием, обучение статистик только на
train, неизвестные категории, хеш данных и работу Pipeline внутри CV.
