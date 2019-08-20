import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[self.column].values.reshape(-1,1)

class FeatureOneHot(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[self.column].fillna(0).values.reshape(-1,1)

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return np.log(data_frame[self.column].values.reshape(-1,1))

class dummySelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[self.columns].fillna(0)
