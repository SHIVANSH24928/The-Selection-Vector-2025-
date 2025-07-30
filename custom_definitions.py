

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Cement'] = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Water'] = df['Water_component_4kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Superplasticizer'] = df['Superplasticizer_component_5kg_in_a_m3_mixture'].replace(0, 0.0001)

        df['water_cement_ratio'] = df['Water'] / df['Cement']
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        df['total_aggregate'] = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['plasticizer_per_cement'] = df['Superplasticizer'] / df['Cement']
        df['cement_agg_ratio'] = df['Cement'] / df['total_aggregate']
        df['water_binder_ratio'] = df['Water'] / df['total_binder']
        df['mix_density'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture'] +
            df['Water'] +
            df['Superplasticizer'] +
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        return df

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.skewed_cols = [
            'Age_day',
            'water_cement_ratio',
            'Superplasticizer_component_5kg_in_a_m3_mixture',
            'plasticizer_per_cement',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'cement_agg_ratio',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Cement_component_1kg_in_a_m3_mixture'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.skewed_cols:
            if col in X.columns:
                if X[col].min() > 0:
                    X_transformed[col] = np.log1p(X[col])
                else:
                    X_transformed[col] = np.log1p(X[col] - X[col].min() + 1)
        return X_transformed

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' or X_copy[col].dtype == 'string':
                X_copy[col] = X_copy[col].astype(str).str.lower()
        return X_copy


class RemoveUnusualChars(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, pattern=r'[^a-zA-Z0-9\s]'):
        self.columns = columns
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        columns_to_clean = self.columns or X_copy.select_dtypes(include=['object', 'string']).columns

        for col in columns_to_clean:
            X_copy[col] = X_copy[col].astype(str).apply(lambda x: re.sub(self.pattern, '', x))

        return X_copy


def split_columns(df):
    df = df.copy()
    for col in df.columns:
        if ',' in col:
            parts = col.split(',')
            df[parts[0]] = df[col].str[0]
            df[parts[1]] = df[col].str[1]
            df.drop(columns=col, inplace=True)
    return df