# Early diabetes readmission

Spark preparation and two PyCaret searches for the UCI Diabetes 130-US Hospitals dataset (variant 14). The assignment explicitly permits the original approximately 100,000 records. Source: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008 (CC BY 4.0; Strack et al., 2014).

The target is readmission within 30 days (`<30`), versus `NO`/`>30`. Keep the earliest observed encounter per patient before splitting; patient identifiers and the original target never enter predictors. This evaluates first observed encounters with information available at discharge. Spark replaces `?`, removes duplicate encounters, casts numerical fields and creates three features. All learned preprocessing stays inside training CV.

## Reproduce

Python 3.11 and Java 11 are required for the locked Spark 3.5 environment. Set `JAVA_HOME` to a local JDK 11 installation, for example on this Windows machine:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot'
py -3.11 -m uv sync --locked
py -3.11 -m uv run --locked dvc checkout
py -3.11 -m uv run --locked jupyter lab
```

Run every cell of `notebooks/readmission_automl.ipynb` from a fresh kernel. Alternatively, `py -3.11 -m uv run --locked python -m src.readmission` runs the same pipeline. Raw data are immutable. DVC cache is local; no remote dataset store is configured. If the cache is unavailable, download the UCI archive and extract `diabetic_data.csv` into `data/raw/` before execution.

Patients are sorted by identifier before the seeded split, independent of Spark partition order. The external stratified 20% holdout is separated before an 18,000-patient training sample. Each search compares three algorithms with three stratified CV folds and a two-minute budget. The first uses logistic regression (baseline), decision tree and random forest (`lr,dt,rf`); the second uses AdaBoost, gradient boosting and linear discriminant analysis (`ada,gbc,lda`). Model selection uses CV ROC-AUC. Report external ROC-AUC, accuracy, F1 and average precision (PR-AUC); the default 0.5 threshold is unchanged. The held-out patients do not participate in finalization.

`reports/results.json` contains exact counts, versions, run identifiers and metrics. `reports/leaderboard_*.csv` contains actual CV comparisons. `models/*.pkl` stores complete preprocessing pipelines. The notebook reloads a saved model and verifies finite probabilities.

## Local experiment comparison

```powershell
py -3.11 -m uv run --locked mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8787
```

Open http://127.0.0.1:8787 and the `diabetes-readmission-variant-fourteen` experiment. MLflow stores parameters, metrics, counts, library versions, leaderboards and both models locally. Stop the UI with Ctrl+C. Local database/artifact directories are ignored; rerunning regenerates them. The screenshot in `reports/screenshots/` shows the actual local UI after execution.

This is an educational retrospective benchmark. First-encounter selection and CPU sampling limit generalization; clinical deployment would require temporal and external validation.
