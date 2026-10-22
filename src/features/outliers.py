"""Training-only local outlier filtering and deterministic HR features."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


class LOFResampler(BaseEstimator):
    """Remove training anomalies together with their labels; inference skips sampling."""
    def __init__(self, n_neighbors=25, contamination=0.025):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit_resample(self, X, y):
        self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                          contamination=self.contamination, n_jobs=1)
        self.inlier_mask_ = self.detector_.fit_predict(X) == 1
        self.n_removed_ = int((~self.inlier_mask_).sum())
        return np.asarray(X)[self.inlier_mask_], np.asarray(y)[self.inlier_mask_]


class HRFeatures(TransformerMixin, BaseEstimator):
    """Pointwise features: no population statistics or target access."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        result["YearsBeforeCurrentRole"] = (result["YearsAtCompany"] -
                                            result["YearsInCurrentRole"])
        result["IncomePerWorkingYear"] = result["MonthlyIncome"] / (result["TotalWorkingYears"] + 1)
        result["IncomePerJobLevel"] = result["MonthlyIncome"] / result["JobLevel"].clip(lower=1)
        return result


def mixed_mutual_information(X, y):
    """Treat encoded binary indicators as discrete; other features as continuous."""
    from sklearn.feature_selection import mutual_info_classif
    values = np.asarray(X)
    binary = np.all((values == 0) | (values == 1), axis=0)
    return mutual_info_classif(values, y, discrete_features=binary, random_state=42)
