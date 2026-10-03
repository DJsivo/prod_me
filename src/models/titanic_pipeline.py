"""Run the Titanic preprocessing and logistic-regression practice offline.

Usage from the repository root: python -m src.models.titanic_pipeline
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.missing_count import MissingCountAdder


ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked", "deck"]
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "deck"]
PARAM_GRID = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}
RANDOM_STATE = 42


def build_pipeline():
    """All learned preprocessing is inside CV; counts precede imputation."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ["missing_count"]),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("missing_count", MissingCountAdder()),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def load_dataset(path):
    frame = pd.read_csv(path)
    required = FEATURES + ["survived"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame["survived"].isna().any() or set(frame["survived"].unique()) != {0, 1}:
        raise ValueError("survived must contain both binary classes 0 and 1, without missing values.")
    # An explicit allowlist excludes alive and other target-derived columns.
    X = frame.loc[:, FEATURES].copy()
    for feature in NUMERIC_FEATURES:
        X[feature] = pd.to_numeric(X[feature], errors="raise")
        if np.isinf(X[feature].to_numpy(dtype=float)).any():
            raise ValueError(f"Numeric column {feature!r} contains infinity.")
    return X, frame["survived"].astype(int)


def run(data_path, output_path):
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(), PARAM_GRID, scoring="roc_auc", cv=cv,
        n_jobs=1, refit=True, error_score="raise",
    )
    search.fit(X_train, y_train)
    prediction = search.predict(X_test)
    probability = search.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
    }
    report = {
        "dataset": {
            "file": data_path.name,
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "rows": len(X), "train_rows": len(X_train), "test_rows": len(X_test),
            "features": FEATURES, "target": "survived",
            "missing_values": {name: int(value) for name, value in X.isna().sum().items()},
        },
        "split": {"test_size": 0.2, "random_state": RANDOM_STATE, "stratify": "survived"},
        "cv": {
            "n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE,
            "scoring": "roc_auc", "best_score": float(search.best_score_),
            "candidates": [
                {"params": params, "mean_test_score": float(mean), "std_test_score": float(std)}
                for params, mean, std in zip(
                    search.cv_results_["params"], search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
                )
            ],
        },
        "best_params": search.best_params_,
        "test_metrics": metrics,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Dataset: {len(X)} rows; train: {len(X_train)}; test: {len(X_test)}")
    print(f"GridSearchCV: 5 folds, {len(search.cv_results_['params'])} candidates, scoring=roc_auc")
    print(f"best_params_: {search.best_params_}")
    print(f"Best CV ROC AUC: {search.best_score_:.6f}")
    print("Test metrics (positive class: survived=1, prediction threshold: 0.5):")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {name}: {metrics[name]:.6f}")
    print(f"Confusion matrix (rows=true, columns=predicted; labels=[0, 1]): {metrics['confusion_matrix']}")
    print(f"Saved report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/titanic.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/titanic_metrics.json")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
