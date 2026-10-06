## Практика: Featuretools и DFS

Решение: [notebooks/featuretools_dfs.ipynb](notebooks/featuretools_dfs.ipynb).
В ноутбуке создаются четыре связанные таблицы и генерируются признаки клиентов
с помощью DFS (`max_depth=2`). Результаты выполнения сохранены.

```powershell
uv sync --locked
uv run --locked python -m ipykernel install --user --name prod-me --display-name "Python (prod-me)"
uv run --locked jupyter lab
```

Выберите ядро `Python (prod-me)` и выполните ячейки по порядку.
Данные синтетические и создаются в ноутбуке; отдельные файлы не нужны.
Версия Python задана в `.python-version`, зависимости — в `pyproject.toml`
и `uv.lock`.

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
