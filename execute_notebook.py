from pathlib import Path
import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import json, sys
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell('''# Neural credit-card fraud detection

Variant 14 of the tabular neural-network assignment. Acceptance: genuine data; seed 42 train/validation/test; train-only imputation/scaling; LightGBM baseline; two PyTorch MLP configurations selected by validation average precision with early stopping; untouched test evaluation; PCA, t-SNE and UMAP; reloadable saved artifacts.

Data: [official TensorFlow tutorial and public CSV mirror](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data), originally ULB/Worldline [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The unchanged source has 284,807 rows and 492 frauds, with Time, V1–V28, Amount and Class. V1–V28 are already anonymised PCA features. No categorical columns exist, so categorical mode/label encoding and nn.Embedding are inapplicable.'''),nbformat.v4.new_code_cell('''from pathlib import Path
import sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from fraud import load_data, run, projections
raw = load_data()
print(raw.shape, 'fraud:', int(raw.Class.sum()))
display(raw.head(), raw.isna().sum().to_frame('missing').T)
'''),nbformat.v4.new_markdown_cell('''## Leakage control and training

Remove exact duplicate transactions before a stratified 60/20/20 split. This prevents identical records in separate partitions. Keep validation and test at their natural deduplicated prevalence. For CPU efficiency, retain all training frauds and sample 30,000 training normals; report both prevalences explicitly. Fit median imputation and StandardScaler on this training subset only. No synthetic data are introduced.

Train two MLPs with BatchNorm, ReLU, Dropout, Adam and BCEWithLogitsLoss weighted by the *used training* class ratio. Batch size 256; at most 35 epochs; patience 5 on validation AP. Compare 128–64/dropout .2/lr .001 and 64–32/dropout .4/lr .0005. Validation AP selects the checkpoint and architecture. The selected model is not refit on validation. Choose each model's F1 threshold on validation only. Test is used only after all choices. CPU PyTorch wheel is intentional on this machine; the device check is recorded below and CUDA is used if available in a compatible installation.'''),nbformat.v4.new_code_cell('''data, preprocessing, features, report = run()
display(pd.DataFrame(report['split']).T)
print('Device:', report['device'], 'duplicates removed:', report['duplicates_removed'])
print('Selected:', report['selected_config'], 'epoch:', report['best_epoch'])
display(pd.DataFrame(report['test']).T)
print('Validation AP:', report['validation_ap'])
'''),nbformat.v4.new_code_cell('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,4))
for i, history in enumerate(report['histories']):
    ax.plot([r['epoch'] for r in history], [r['validation_ap'] for r in history], label=f'configuration {i+1}')
ax.set(xlabel='Epoch', ylabel='Validation average precision', title='Validation model selection')
ax.legend(); plt.show()
'''),nbformat.v4.new_markdown_cell('''## Descriptive projections

Project 2,500 sampled normal transactions and 400 sampled frauds, using the fitted training scaler. PCA retains all 30 available dimensions before nonlinear projections (the requirement is up to 50); plot its first two axes separately. The enriched 2,900-row sample intentionally improves visibility of the rare class and does not estimate real prevalence. The same observations and labels are used by all three plots; labels do not fit the projections. These descriptive plots cover data from all splits and are created after test evaluation; they do not influence model selection.'''),nbformat.v4.new_code_cell('projections(data, preprocessing, features)'),nbformat.v4.new_markdown_cell('''## Interpretation and limits

The executed run selected the 128-64 MLP at epoch 12 (validation AP .811114); the smaller configuration reached .763850. Test AP is .717185 for MLP versus .791736 for LightGBM, so the boosting baseline ranks the rare frauds better on this split. Both detect 71/95 test frauds at validation-selected thresholds; false positives are 9 and 8 respectively.

In these plots, PCA places many frauds far from the dense normal cloud but retains overlapping cases near its centre. t-SNE and UMAP show fraud-rich regions as well as fraudulent observations embedded among normal transactions. This supports partial structure, not complete separation.

Accuracy is dominated by normal transactions; ROC-AUC and especially average precision reveal rare-event ranking quality. F1 and confusion matrices refer to validation-selected thresholds. Training undersampling and class weighting alter the learned probability prior: scores are useful for ranking but are not calibrated operational fraud probabilities. The representative test prevalence is therefore essential.

PCA shows the strongest linear variation; overlap in two axes cannot prove that the original 30-dimensional problem is inseparable. t-SNE and UMAP emphasise local neighbourhoods and may form islands even in continuous data. Apparent class concentrations can suggest structure, but neither island distances nor visual separation prove generalisation. The random transaction split is educational; a time-based evaluation and stronger transaction/customer grouping would be needed for deployment. Duplicate removal changes the original benchmark population and the exact counts are reported above. Saved weights and preprocessing are round-trip checked before success.''')]
nb.metadata={'kernelspec':{'display_name':'Python (fraud)','language':'python','name':'python3'},'language_info':{'name':'python'}}
path=root/'notebooks/neural_fraud_detection.ipynb'
nbformat.write(nb,path)
kernel_root=root/'kernels'
spec=kernel_root/'fraud'; spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Fraud environment','language':'python'}))
manager=KernelManager(kernel_name='fraud',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
client=NotebookClient(nb,timeout=1800,km=manager,resources={'metadata':{'path':str(root)}})
client.execute()
nbformat.write(nb,path)
print('Executed:',path)
