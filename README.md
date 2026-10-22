# Структура проекта и DVC

```text
prod_me/
├── data/
│   ├── raw/           # Исходные данные
│   ├── processed/     # Подготовленные данные
│   └── external/      # Внешние источники
├── notebooks/         # Анализ и эксперименты
├── src/               # Повторно используемый код
├── models/            # Сохранённые модели
└── reports/           # Результаты и изображения
```

Окружение управляется через `uv`; зависимости зафиксированы в `uv.lock`.

```powershell
uv sync --locked
uv run --locked dvc status
uv run --locked jupyter lab
```

DVC уже инициализирован. Данные и модели добавляются командами `uv run dvc add`;
в Git сохраняются только их `.dvc`-файлы. Окружение, кэши и содержимое
наборов данных не отслеживаются Git. Исходный учебный пример `data/raw/sample_students.csv`
восстановлен из сохранённого DVC-кэша и также отслеживается DVC.

[Задания с занятий](https://github.com/DJsivo/prod_me/tree/practices) · [Общее оглавление](https://github.com/DJsivo/prod_me/tree/main)
