# Advanced Data Methods

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

## Структура веток

```text
main                              # Общий каркас проекта и оглавление
├── practices                     # Задания с лекционных (практических) занятий
│   ├── project-structure         # Структура проекта и настройка DVC
│   ├── ml-pipeline               # Pipeline и подбор гиперпараметров
│   └── featuretools-dfs          # Автоматическая генерация признаков
└── labs                          # Пять работ из методички
    ├── data-reproducibility      # Воспроизводимость и конвейер обработки
    ├── feature-engineering      # Пропуски, выбросы и отбор признаков
    ├── boosting-interpretability # Бустинг, ансамбли и объяснение моделей
    ├── neural-fraud-detection    # Нейросеть для обнаружения мошенничества
    └── readmission-automl        # PySpark, AutoML и повторные госпитализации
```

Ветки заданий создаются от `practices` или `labs` и называются по технической
теме, без номеров и учебных префиксов. Объединение готовых работ выполняется
отдельно, когда оно понадобится.

## Задания с лекционных (практических) занятий

| Тема | Ветка |
|---|---|
| Структура проекта и DVC | [project-structure](https://github.com/DJsivo/prod_me/tree/project-structure) |
| Pipeline и подбор параметров | [ml-pipeline](https://github.com/DJsivo/prod_me/tree/ml-pipeline) |
| Генерация признаков Featuretools | [featuretools-dfs](https://github.com/DJsivo/prod_me/tree/featuretools-dfs) |

## Работы по методичке

| Тема | Ветка |
|---|---|
| Воспроизводимость исследований | [data-reproducibility](https://github.com/DJsivo/prod_me/tree/data-reproducibility) |
| Инженерия и отбор признаков | [feature-engineering](https://github.com/DJsivo/prod_me/tree/feature-engineering) |
| Бустинг, ансамбли и интерпретация | [boosting-interpretability](https://github.com/DJsivo/prod_me/tree/boosting-interpretability) |
| Нейросеть и визуализация мошеннических транзакций | [neural-fraud-detection](https://github.com/DJsivo/prod_me/tree/neural-fraud-detection) |
| Spark, AutoML и повторные госпитализации | [readmission-automl](https://github.com/DJsivo/prod_me/tree/readmission-automl) |

## Запуск

В ветке задания:

```powershell
uv sync --locked
uv run --locked jupyter lab
```

Зависимости каждой ветки фиксируются в `pyproject.toml` и `uv.lock`;
инструкции и результаты находятся в её README и ноутбуке.
