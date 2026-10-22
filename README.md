# Neural fraud detection

Tabular neural-network assignment, variant 14: genuine ULB/Worldline Credit Card Fraud Detection. The raw CSV is downloaded unchanged from the [official TensorFlow tutorial mirror](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data); [original dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). No account or credentials are needed.

## Reproduce

Python 3.11.6, dependencies fixed by `uv.lock`; PyTorch uses the official CPU wheel index. On a CUDA machine install a matching CUDA build explicitly and regenerate the lock before reporting GPU results. The implementation records the chosen device and supports CUDA when available.

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --locked python execute_notebook.py
py -3.11 -m uv run --locked dvc add data/raw/creditcard.csv models/fraud_bundle
py -3.11 -m uv run --locked dvc status
```

The notebook runner writes an executed `notebooks/neural_fraud_detection.ipynb` from a clean kernel. `load_data()` downloads the source if absent; this is needed on a fresh clone because no remote DVC storage is configured. The resulting source SHA-256 and exact split counts are recorded in `reports/fraud_metrics.json`. Model artifacts are regenerable by executing the notebook; DVC tracks the source and `models/fraud_bundle` directory. `dvc pull` requires a separately configured remote and is not claimed to work here.

## Method

Remove exact duplicates before splitting to avoid identical transactions across partitions. Stratified seed-42 split: 60% training, 20% validation, 20% test. Only training is undersampled: all its frauds and 30,000 normals. Holdouts retain natural prevalence. Fit median imputation and standardisation on training only. All 30 inputs are numeric; categorical embeddings do not apply.

Compare LightGBM to two PyTorch MLPs (128–64, dropout .2, lr .001; 64–32, dropout .4, lr .0005). Both use BatchNorm, ReLU, Adam, batch 256, weighted BCE, 35-epoch cap and patience 5. Validation average precision selects checkpoints/configuration. Validation F1 selects decision thresholds. Test is evaluated afterwards, never used to select parameters. Saved architecture, state dict and preprocessing are round-trip tested.

PCA retains the available 30 dimensions, then t-SNE and UMAP operate on a descriptive enriched sample of 2,500 normal and 400 fraudulent transactions. All three plots are saved as notebook outputs. Sample enrichment is for visibility only; embedding clusters do not prove class separation or operational performance.

## Results

The clean CPU run selected the 128-64 MLP, dropout .2, lr .001 at epoch 12 (validation AP .811114). The alternative configuration reached validation AP .763850. LightGBM validation AP was .871440.

| Test model | ROC-AUC | Average precision | F1 | Accuracy |
|---|---:|---:|---:|---:|
| MLP | .959869 | .717185 | .811429 | .999418 |
| LightGBM | .971304 | .791736 | .816092 | .999436 |

LightGBM ranked rare fraud cases better on this split. Both selected thresholds detected 71 of 95 test frauds; MLP produced 9 false positives and LightGBM 8. The test contains 56,746 rows; validation 56,745; training before sampling 170,235 and after sampling 30,284 (284 frauds). Source deduplication removed 1,081 exact duplicates, leaving 473 frauds. Used training prevalence is .9378%, versus .1674% on test.

PCA shows many frauds extending away from the dense normal cloud but also overlap near its centre. t-SNE and UMAP both show fraud-rich regions alongside fraudulent points mixed with normal transactions. These visual patterns do not establish complete separability.

Exact metrics, validation histories, thresholds and confusion matrices: `reports/fraud_metrics.json` and the executed notebook. Average precision is the primary ranking metric because accuracy is misleading under extreme imbalance. Undersampling plus class weighting means scores are not calibrated fraud probabilities. A random split of historical anonymous transactions is an educational benchmark; deployment needs temporal/grouped evaluation and calibration.

[Assignment index](https://github.com/DJsivo/prod_me/tree/labs) · [Project index](https://github.com/DJsivo/prod_me/tree/main)
