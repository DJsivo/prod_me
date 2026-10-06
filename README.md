## Практика: Pipeline и GridSearchCV

Решение с кодом, результатами и выводами: [notebooks/titanic_pipeline.ipynb](notebooks/titanic_pipeline.ipynb).

Датасет Titanic хранится в `data/external/titanic.csv`.
Источник: [seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).

```powershell
uv sync --locked
uv run --locked python -m ipykernel install --user --name prod-me --display-name "Python (prod-me)"
uv run --locked jupyter lab
```

Выберите ядро `Python (prod-me)` и выполните ячейки по порядку.
Данные загружаются локально. Версия Python задана в `.python-version`,
зависимости — в `pyproject.toml` и `uv.lock`.

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
├── pyproject.toml
├── uv.lock
└── README.md
```
