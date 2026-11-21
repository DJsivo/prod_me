from pathlib import Path
import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import json, sys
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell('''# Обнаружение мошенничества по банковским транзакциям

Вариант 14. Сравниваем LightGBM и две полносвязные сети PyTorch на Credit Card Fraud. Исходные данные содержат 284807 транзакций и 492 мошенничества. Time, Amount и V1–V28 — числовые признаки; V1–V28 уже анонимизированы методом PCA. Категорий нет, поэтому Embedding не требуется. Данные доступны через официальный учебный источник TensorFlow; исходный CSV проверяется по контрольной сумме.'''),nbformat.v4.new_code_cell('''from pathlib import Path
import sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from fraud import load_data, run, projections
raw = load_data()
print(raw.shape, 'мошенничеств:', int(raw.Class.sum()))
display(raw.head(), raw.isna().sum().to_frame('пропуски').T)
'''),nbformat.v4.new_markdown_cell('''## Подготовка и обучение

Удаляем точные дубликаты до стратифицированного разделения train/validation/test 60/20/20, seed=42. В train сохраняем все мошенничества и 30000 обычных операций для воспроизводимого ограниченного эксперимента; validation и test сохраняют естественные доли классов. Медианы и StandardScaler обучаются только на используемом train.

Обучение сети выполняется на CUDA. Сравниваются архитектуры 128–64 и 64–32, Dropout 0.2/0.4, learning rate 0.001/0.0005. Обе сети используют BatchNorm, ReLU, Adam, batch=256 и BCEWithLogitsLoss с весом редкого класса. Максимум 35 эпох, patience=5 по validation AP. Валидация выбирает конфигурацию, checkpoint и F1-порог. Test не используется для выбора.'''),nbformat.v4.new_code_cell('''data, preprocessing, features, report = run()
display(pd.DataFrame(report['split']).T)
print('Устройство:', report['device'], 'удалено дубликатов:', report['duplicates_removed'])
print('Выбрано:', report['selected_config'], 'эпоха:', report['best_epoch'])
display(pd.DataFrame(report['test']).T)
print('AP на валидации:', report['validation_ap'])
'''),nbformat.v4.new_code_cell('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,4))
for i, history in enumerate(report['histories']):
    ax.plot([r['epoch'] for r in history], [r['validation_ap'] for r in history], label=f'конфигурация {i+1}')
ax.set(xlabel='Эпоха', ylabel='Average precision на валидации', title='Выбор модели по валидации')
ax.legend(); plt.show()
'''),nbformat.v4.new_markdown_cell('''## Визуализация

PCA сохраняет доступные 30 компонент, затем t-SNE и UMAP строят двумерные проекции. Для видимости редкого класса используются 2500 обычных и 400 мошеннических транзакций. Доля мошенничества на рисунке намеренно завышена. Проекции описательные, строятся после тестирования и не влияют на подбор модели. Цвета обозначают реальные классы.'''),nbformat.v4.new_code_cell('projections(data, preprocessing, features)'),nbformat.v4.new_markdown_cell('''## Выводы и ограничения

Точные метрики, выбранная архитектура, эпоха, пороги и устройство представлены в таблицах выше и reports/fraud_metrics.json. Сравниваем модели по ROC-AUC, average precision и F1, а не только accuracy: обычные транзакции составляют подавляющее большинство.

PCA показывает линейную структуру и перекрытие классов; t-SNE и UMAP подчёркивают локальные группы, но расстояния между островками не доказывают разделимость и качество модели. Подвыборка train и веса изменяют распределение обучения, поэтому scores не являются калиброванными вероятностями. Случайное разделение — учебный benchmark; для применения необходимы временная проверка, группировка клиентов и калибровка. Сохранённые веса и предобработка проверяются повторной загрузкой.''')]
nb.metadata={'kernelspec':{'display_name':'Python (fraud)','language':'python','name':'python3'},'language_info':{'name':'python'}}
path=root/'notebooks/neural_fraud_detection.ipynb'
nbformat.write(nb,path)
kernel_root=root/'kernels'
spec=kernel_root/'fraud'; spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Fraud environment','language':'python'}))
manager=KernelManager(kernel_name='fraud',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
client=NotebookClient(nb,timeout=1800,km=manager,resources={'metadata':{'path':str(root)}})
client.execute()
report = json.loads((root/'reports/fraud_metrics.json').read_text(encoding='utf-8'))
a, b = report['test']['mlp'], report['test']['lightgbm']
nb.cells[-1].source += (f"\n\nФактический запуск: {report['device']}, {report['gpu_name']}. "
    f"Выбрана MLP {report['selected_config']['widths']}, эпоха {report['best_epoch']}. "
    f"Test AP: MLP {a['average_precision']:.6f}, LightGBM {b['average_precision']:.6f}; "
    f"ROC-AUC: {a['roc_auc']:.6f} и {b['roc_auc']:.6f}. "
    f"Матрица MLP (строки — реальность, столбцы — прогноз): {a['confusion_matrix']}. "
    + ("LightGBM лучше ранжирует редкие мошенничества по AP." if b['average_precision'] > a['average_precision'] else ("MLP лучше ранжирует редкие мошенничества по AP." if a['average_precision'] > b['average_precision'] else "AP моделей совпала." )))
nbformat.write(nb,path)
print('Выполнено:',path)
