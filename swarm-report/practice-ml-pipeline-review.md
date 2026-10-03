# Независимая проверка практики

Вердикт: **принято**. Проверены критерии из `practice-ml-pipeline-route.md`,
README, оба модуля реализации, локальный CSV, JSON-результат и
`reports/titanic_practice.md`. Существенных замечаний нет.

## Соответствие заданию

- CSV содержит 891 строку и смешанные признаки; выбранные восемь признаков
  включают реальные пропуски. `survived` и дублирующий цель `alive` исключены
  явным списком входных признаков.
- Разбиение train/test выполнено до `fit`: 712/179 строк, стратификация,
  `random_state=42`. Индексы train/test не пересекаются.
- `MissingCountAdder` находится до импутации: считает пропуски в исходных
  входах, не меняет DataFrame, поддерживает sklearn clone и проверяет fit.
- Числовая ветвь использует медиану и StandardScaler, категориальная —
  most_frequent и OneHotEncoder. Неизвестные категории на predict поддержаны.
  Последний шаг — LogisticRegression.
- GridSearchCV получает весь Pipeline и только train; проверяет шесть сочетаний
  C и class_weight на пяти стратифицированных фолдах. Все обучаемые параметры
  препроцессинга остаются внутри Pipeline, поэтому вычисляются на train каждого
  фолда. Test используется после выбора модели. Утечки при чтении кода не найдены.
- Выведены best_params_ и метрики test, присутствуют код и содержательные
  выводы. Данные локальные, инструкция запуска из корня согласована с CLI.

## Фактические проверки

Python для проверки:
`C:\Users\t-pud\OneDrive\Документы\ChatGPT\Prod_me\.venv\Scripts\python.exe`.
В командах ниже `$python` обозначает этот полный путь. Рабочий каталог — корень
ветки `codex/practice-ml-pipeline`. Установка зависимостей в новое окружение
не проверялась; использовано уже доступное окружение, его версии совпали с JSON.

1. `& $python -m unittest discover -s tests -v` — exit code 0,
   **5 тестов, OK**. Проверены исходные счётчики пропусков и отсутствие мутации,
   clone/fit lifecycle, train-only статистики медианы/моды и масштабирования на
   специально подобранных данных, неизвестные test-категории, отсутствие
   target leakage, разбиение, SHA и реальный поиск по шести кандидатам.
2. `& $python -m src.models.titanic_pipeline --output "$env:TEMP\titanic_independent_review.json"`
   — exit code 0. Полный запуск: 5 фолдов, 6 кандидатов; C=0.1,
   class_weight=None. CV ROC AUC 0.860268; test accuracy 0.815642,
   precision 0.821429, recall 0.666667, F1 0.736000, ROC AUC 0.842161;
   матрица ошибок `[[100, 10], [23, 46]]`.
3. `& $python -c 'import json, os; from pathlib import Path; actual=json.loads((Path(os.environ["TEMP"])/"titanic_independent_review.json").read_text(encoding="utf-8")); saved=json.loads(Path("reports/titanic_metrics.json").read_text(encoding="utf-8")); assert actual == saved; print("Independent CLI report exactly matches saved JSON")'`
   — exit code 0, **повторный отчёт полностью совпадает с сохранённым JSON**,
   включая версии, кандидатов, метрики, разбиение и SHA.
4. `Get-FileHash data/external/titanic.csv -Algorithm SHA256` — exit code 0,
   хеш `81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2`
   совпадает с README и JSON. Формулы, числа и выводы в
   `reports/titanic_practice.md` согласованы с реальным запуском.

Изменения независимого проверяющего: только `tests/test_titanic_pipeline.py`
и этот отчёт. Реализация не изменялась; коммиты и push не выполнялись.

Дополнительно: `git check-attr text -- data/external/titanic.csv` вернул `text: unset`; правило `.gitattributes` сохраняет байты CSV при checkout, текущий SHA-256 остался тем же.
