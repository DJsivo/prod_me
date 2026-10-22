# Advanced Data Methods

Материалы разделены на две независимые линии работы:

- [practices](https://github.com/DJsivo/prod_me/tree/practices) — задания с занятий: скриншоты, текст и презентации лекций.
- [labs](https://github.com/DJsivo/prod_me/tree/labs) — пять работ из методички.

Каждое задание выполняется в отдельной ветке от своей линии. Названия веток
описывают тему на английском, без номеров и учебных префиксов. В `main`
хранятся общий каркас и навигация. Готовые работы можно объединять в своей
линии, а позднее — в общей ветке.

```text
main
├── practices
│   ├── project-structure
│   ├── ml-pipeline
│   └── featuretools-dfs
└── labs
    ├── data-reproducibility
    ├── feature-engineering
    ├── boosting-interpretability
    ├── neural-fraud-detection
    └── readmission-automl
```

## Задания с занятий

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
