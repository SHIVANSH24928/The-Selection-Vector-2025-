

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
import re


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



class PipelineWithLabelDecoder:
    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded_pred = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_encoded_pred)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Leela_Venkata_Sai_Nerella(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = self.featuressplit(X)
        X = self.remove(X)
        return X

    def featuressplit(self, X):
        feature1, feature6, feature8 = [], [], []
        feature10, feature15, feature21 = [], [], []

        for i in range(len(X)):
            # Last column: feature_1 and feature_6
            cell = X.iloc[i, -1]
            a, b = self.safe_split(cell)
            feature1.append(a)
            feature6.append(b)

            # Second last column: feature_21 and feature_10
            cell = X.iloc[i, -2]
            a, b = self.safe_split(cell)
            feature21.append(a)
            feature10.append(b)

            # Third last column: feature_8 and feature_15
            cell = X.iloc[i, -3]
            a, b = self.safe_split(cell)
            feature8.append(a)
            feature15.append(b)

        X["feature_1"] = feature1
        X["feature_6"] = feature6
        X["feature_8"] = feature8
        X["feature_10"] = feature10
        X["feature_15"] = feature15
        X["feature_21"] = feature21

        # Drop last 3 columns (which were split)
        X = X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1)

        return X

    

    def safe_split(self, cell):
        if pd.isna(cell):
            return np.nan, np.nan
        try:
            a, b = str(cell).split(",", 1)
            return a, b
        except:
            return np.nan, np.nan

    def remove(self, X):
        for col in X.columns:
            X[col] = X[col].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x))
        # Try converting to numeric where possible
            X[col] = pd.to_numeric(X[col], errors='ignore')  # or use 'coerce' if you want NaNs instead of invalid
        return X



def split_columns(df):
    df = df.copy()
    for col in df.columns:
        if ',' in col:
            parts = col.split(',')
            df[parts[0]] = df[col].str[0]
            df[parts[1]] = df[col].str[1]
            df.drop(columns=col, inplace=True)
    return df


def fill_missing(df):

    return df.fillna('')
    
def split_bycomma(X_df):
    X = X_df.copy()
    X[['feature_8', 'feature_15']] = X['feature_8,feature_15'].str.split(',', expand=True)
    X[['feature_21', 'feature_10']] = X['feature_21,feature_10'].str.split(',', expand=True)
    X[['feature_1', 'feature_6']] = X['feature_1,feature_6'].str.split(',', expand=True)
    X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1, inplace=True)
    return X
fillna_transformer = FunctionTransformer(fill_missing, validate=False)
