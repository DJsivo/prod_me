## Практика: Featuretools и DFS

Решение: [notebooks/featuretools_dfs.ipynb](notebooks/featuretools_dfs.ipynb).
В ноутбуке создаются четыре связанные таблицы и генерируются признаки клиентов
с помощью DFS (`max_depth=2`). Результаты выполнения сохранены.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name prod-me --display-name "Python (prod-me)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Выберите ядро `Python (prod-me)` и выполните ячейки по порядку.
Данные синтетические и создаются в ноутбуке; отдельные файлы не нужны.

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
