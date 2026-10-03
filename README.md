## Практика: Pipeline и GridSearchCV

Решение с кодом, результатами и выводами: [notebooks/titanic_pipeline.ipynb](notebooks/titanic_pipeline.ipynb).

Датасет Titanic хранится в `data/external/titanic.csv`.
Источник: [seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Откройте ноутбук и выполните все ячейки по порядку. Данные загружаются локально.

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
