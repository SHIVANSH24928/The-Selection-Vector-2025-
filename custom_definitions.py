# custom_definitions.py

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

# This is the custom class from your notebook
class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.count_maps_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            counts = X_df[col].value_counts()
            self.count_maps_[col] = counts.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns).copy()
        for col in self.columns:
            # Map known categories; map new ones to 0
            X_df[col] = X_df[col].map(self.count_maps_).fillna(0)
        return X_df.values

def rename_columns(X):
    return pd.DataFrame(X, columns=[i for i in range(X.shape[1])])