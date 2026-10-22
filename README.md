# Инженерия и отбор признаков

Вариант 14: классификация Attrition в одиночной таблице IBM HR.
[Исполненный notebook](notebooks/feature_engineering.ipynb) сравнивает медиану, KNN и Iterative (5-fold ROC-AUC), показывает импутацию и LOF/PCA, добавляет ручные признаки и сравнивает отбор MI/XGBoost. Обучаемая предобработка, LOF и отбор выполняются внутри fold. Test 20% фиксируется первым и используется только для итоговой оценки.

Источник: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Предоставленный исходный CSV `WA_Fn-UseC_-HR-Employee-Attrition.csv`: 1470 строк, 35 столбцов, SHA256 `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659`. Raw сохраняется неизменным. Производная таблица содержит MCAR 12% в трёх колонках, синтетические выбросы и фиксированную принадлежность к split; подробности — в notebook.

## Запуск

Из корня ветки, Python 3.11.6 и uv:

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --locked dvc pull
py -3.11 -m uv run --locked python -m src.run_notebook
py -3.11 -m uv lock --check
py -3.11 -m uv run --locked dvc status
```

DVC remote не настроен: `dvc pull` применим после настройки доступного хранилища; при отсутствии кэша скачайте CSV по ссылке источника в `data/raw/` и сверьте SHA256. Notebook заново создаст processed и модели; после пересчёта обновите указатели командой `py -3.11 -m uv run --locked dvc add data/processed/employee_attrition_corrupted.csv models/employee_attrition.joblib models/employee_attrition_metrics.json`. Исполнение создаёт временный kernelspec с текущим интерпретатором, не требует регистрации глобального ядра.

## Результат

Импутация median: train CV ROC-AUC 0.8320 ± 0.0318; KNN 0.8287; Iterative 0.8308. Лучшая диагностическая конфигурация — исходные признаки. Ручные признаки, LOF и отбор в этом опыте не дали прироста среднего CV. Для выполнения всех стадий итоговый конвейер выбирается среди MI/XGBoost по CV, выбран XGBoost. На untouched test (294 строки, seed 42): ROC-AUC 0.7890, accuracy 0.7653, F1 0.4733, average precision 0.5830. Порог 0.5 задан заранее. Эти оценки относятся к искусственно повреждённой версии данных; небольшие различия CV не доказывают превосходство методов. Iterative достиг лимита 5 итераций без сходимости; предупреждение сохранено в notebook.

Сохранённый `models/employee_attrition.joblib` загружается с импортируемыми классами из `src.features.outliers`; загрузка и совпадение вероятностей проверяются в notebook. Raw, processed и оба артефакта модели отслеживаются реальными `.dvc` указателями.

[Все работы по методичке](https://github.com/DJsivo/prod_me/tree/labs) · [Общее оглавление](https://github.com/DJsivo/prod_me/tree/main) · [Задания с занятий](https://github.com/DJsivo/prod_me/tree/practices)
