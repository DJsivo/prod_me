"""Independent checks for the practice's preprocessing and data contracts."""

import hashlib
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.features.missing_count import MissingCountAdder
from src.models.titanic_pipeline import (
    ROOT, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PARAM_GRID,
    RANDOM_STATE, build_pipeline, load_dataset,
)


class MissingCountTests(unittest.TestCase):
    def test_counts_original_missing_values_without_mutating_input(self):
        original = pd.DataFrame({"age": [10.0, np.nan, np.nan],
                                 "sex": ["male", "female", np.nan]}, index=[8, 4, 2])
        snapshot = original.copy(deep=True)
        result = MissingCountAdder().fit_transform(original)
        self.assertEqual(result["missing_count"].tolist(), [0, 1, 2])
        pd.testing.assert_frame_equal(original, snapshot)
        self.assertEqual(result.index.tolist(), [8, 4, 2])
        self.assertTrue(result["age"].isna().iloc[1])

    def test_clone_and_fit_lifecycle(self):
        X = pd.DataFrame({"age": [1.0, np.nan]})
        transformer = MissingCountAdder(feature_name="count").fit(X)
        self.assertEqual(transformer.get_feature_names_out().tolist(), ["age", "count"])
        fresh = clone(transformer)
        self.assertEqual(fresh.feature_name, "count")
        with self.assertRaises(NotFittedError):
            fresh.transform(X)
        self.assertEqual(fresh.fit_transform(X)["count"].tolist(), [0, 1])
        with self.assertRaises(ValueError):
            transformer.transform(X.rename(columns={"age": "fare"}))


class PipelineTests(unittest.TestCase):
    def test_train_statistics_and_unknown_category(self):
        train = pd.DataFrame({
            "pclass": [1, 2, 1, 2], "sex": ["female", "male", "female", "male"],
            "age": [10.0, 20.0, np.nan, 40.0], "sibsp": [0, 1, 0, 1],
            "parch": [0, 0, 1, 1], "fare": [10.0, 20.0, 30.0, 40.0],
            "embarked": ["S", "S", np.nan, "C"],
            "deck": ["A", "A", np.nan, "B"],
        })[FEATURES]
        model = build_pipeline().fit(train, [0, 1, 0, 1])
        num = model.named_steps["preprocessor"].named_transformers_["num"]
        np.testing.assert_allclose(num.named_steps["imputer"].statistics_,
                                   [20.0, 0.5, 0.5, 25.0, 0.0])
        np.testing.assert_allclose(num.named_steps["scaler"].mean_,
                                   [22.5, 0.5, 0.5, 25.0, 0.75])
        cat = model.named_steps["preprocessor"].named_transformers_["cat"]
        self.assertEqual(cat.named_steps["imputer"].statistics_.tolist(),
                         [1, "female", "S", "A"])
        held_out = train.iloc[[0]].copy()
        held_out.loc[:, "age"] = 10000.0
        held_out.loc[:, "deck"] = "UNSEEN"
        held_out.loc[:, "embarked"] = "UNSEEN"
        self.assertEqual(model.predict(held_out).shape, (1,))
        self.assertTrue(np.isfinite(model.predict_proba(held_out)).all())
        np.testing.assert_allclose(num.named_steps["scaler"].mean_,
                                   [22.5, 0.5, 0.5, 25.0, 0.75])

    def test_real_data_allowlist_split_and_recorded_hash(self):
        data = ROOT / "data/external/titanic.csv"
        X, y = load_dataset(data)
        self.assertEqual(X.shape, (891, 8))
        self.assertEqual(X.columns.tolist(), FEATURES)
        self.assertTrue(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURES))
        self.assertFalse({"survived", "alive", "who", "adult_male"} & set(X.columns))
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(X.isna().sum()["age"], 177)
        self.assertEqual(X.isna().sum()["deck"], 688)
        train, test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        self.assertEqual((len(train), len(test)), (712, 179))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertLess(abs(y_train.mean() - y_test.mean()), 0.01)
        report = json.loads((ROOT / "reports/titanic_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(),
                         report["dataset"]["sha256"])

    def test_pipeline_and_cv_contract(self):
        pipeline = build_pipeline()
        self.assertEqual(list(pipeline.named_steps),
                         ["missing_count", "preprocessor", "classifier"])
        self.assertIsInstance(pipeline.named_steps["classifier"], LogisticRegression)
        params = pipeline.get_params()
        self.assertEqual(params["preprocessor__num__imputer__strategy"], "median")
        self.assertEqual(params["preprocessor__cat__imputer__strategy"], "most_frequent")
        self.assertEqual(params["preprocessor__cat__encoder__handle_unknown"], "ignore")
        self.assertEqual(len(PARAM_GRID), 2)
        self.assertTrue(all(key in params and len(values) > 1
                            for key, values in PARAM_GRID.items()))
        X, y = load_dataset(ROOT / "data/external/titanic.csv")
        # Exercise cloned preprocessing across folds with actual mixed/missing data.
        search = GridSearchCV(pipeline, PARAM_GRID, scoring="roc_auc",
                              cv=StratifiedKFold(2, shuffle=True, random_state=42),
                              error_score="raise")
        search.fit(X.iloc[:120], y.iloc[:120])
        self.assertEqual(len(search.cv_results_["params"]), 6)
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())


if __name__ == "__main__":
    unittest.main()
