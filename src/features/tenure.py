from sklearn.base import BaseEstimator, TransformerMixin


class YearsBeforeCurrentRoleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, years_at_company_column="YearsAtCompany", years_in_current_role_column="YearsInCurrentRole"):
        self.years_at_company_column = years_at_company_column
        self.years_in_current_role_column = years_in_current_role_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["YearsBeforeCurrentRole"] = (
            X[self.years_at_company_column] - X[self.years_in_current_role_column]
        )
        return X
