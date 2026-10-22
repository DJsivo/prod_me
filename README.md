# Бустинг и интерпретация увольнений

Вариант 14: IBM HR Employee Attrition, 1470 сотрудников. Источник — [IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Неизменённый CSV в `data/raw` отслеживается DVC.

В [ноутбуке](notebooks/boosting_interpretability.ipynb) сравниваются XGBoost, LightGBM и CatBoost, выполняется Optuna (50 испытаний, CV=5), SHAP, стекинг LightGBM + RandomForest + SVC, взвешивание и ADASYN. Разбиение 80/20, seed=42. Все преобразования и ресемплинг обучаются внутри фолдов; итоговый подход выбирается по train CV ROC-AUC, test используется один раз. PR-AUC означает average precision; F1 имеет порог 0.5.

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --locked dvc pull
py -3.11 -m uv run --locked jupyter lab
```

DVC remote не задан: для `dvc pull` нужен доступ к исходному cache или настроенное хранилище. При отсутствии cache скачайте CSV по ссылке источника и проверьте `uv run --locked dvc status`. Итоговая модель сохраняется в `models/attrition_model.joblib`, также под DVC. Таблицы CV, независимого test и все графики сохранены в notebook.

[Работы по методичке](https://github.com/DJsivo/prod_me/tree/labs) · [Общее оглавление](https://github.com/DJsivo/prod_me/tree/main)

Результат: CatBoost выиграл сравнение стандартных моделей (train CV ROC-AUC 0.8229); Optuna повысила его до 0.8352. Стекинг выбран по train CV 0.8376. Единственная итоговая оценка на test: ROC-AUC 0.8223, PR-AUC 0.5774, F1 0.4638. Веса и ADASYN повысили F1 на CV, но снизили ROC-AUC/PR-AUC. Разница CV стекинга и настроенного бустинга мала относительно разброса фолдов.
