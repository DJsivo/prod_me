"""A clone-compatible transformer that counts missing values before imputation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MissingCountAdder(TransformerMixin, BaseEstimator):
    """Append a missing-value count to a DataFrame without modifying its input."""

    def __init__(self, feature_name="missing_count"):
        self.feature_name = feature_name

    def _validate_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingCountAdder requires a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("Input columns must be unique.")
        if not all(isinstance(name, str) for name in X.columns):
            raise ValueError("Input column names must be strings.")
        if not isinstance(self.feature_name, str) or not self.feature_name:
            raise ValueError("feature_name must be a nonempty string.")
        if self.feature_name in X.columns:
            raise ValueError(f"Input already contains {self.feature_name!r}.")

    def fit(self, X, y=None):
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._validate_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Input columns and their order must match fit().")
        result = X.copy()
        result[self.feature_name] = X.isna().sum(axis=1)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match fitted columns.")
        return np.append(self.feature_names_in_, self.feature_name)
