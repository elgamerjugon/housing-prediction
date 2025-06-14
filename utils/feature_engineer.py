import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Apply log LotFrontage, LotArea, TotalBasementSF, 1flSF, GrLvArea

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def trasnform(self, X):
        X = X.copy()
        for col in self.columns:
            X["log_" + col] = np.log1p(X[col])
        X.drop(columns=self.columns, inpalce=True)
        return X