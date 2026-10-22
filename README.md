# Employee Attrition: воспроизводимый конвейер

Данные IBM HR: 1 470 сотрудников, цель `Attrition`. Вариант 14.
Исходный файл: [IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

```powershell
uv sync --locked
uv run --locked jupyter lab
```

Откройте `notebooks/employee_attrition.ipynb`. Исходный CSV должен находиться
в `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`. Данные и модель отслеживаются
DVC; при настроенном remote их можно получить командой `uv run dvc pull`.
Общий DVC remote в репозитории пока не настроен, поэтому при новом клонировании
исходный CSV нужно скачать по ссылке выше.

В двух числовых колонках создаётся 10% пропусков. Пайплайн добавляет
`YearsBeforeCurrentRole`, заполняет пропуски, масштабирует числовые признаки,
кодирует категории и обучает логистическую регрессию. Test — 20%,
стратификация, `random_state=42`.

Результаты сохранены в ноутбуке; модель — `models/employee_attrition_pipeline.joblib`.
Класс трансформера находится в `src/features/tenure.py`, чтобы модель можно
было загрузить в новом Python-процессе. Снимки исходной проверки находятся
в `reports/screenshots/`.

[Работы по методичке](https://github.com/DJsivo/prod_me/tree/labs) · [Общее оглавление](https://github.com/DJsivo/prod_me/tree/main)
