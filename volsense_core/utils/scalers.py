import torch
import numpy as np
import pandas as pd

class TorchStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def _to_tensor(self, X):
        # ✅ Handles both DataFrames and ndarrays
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            X = X.values if isinstance(X, pd.DataFrame) else X
        return torch.as_tensor(X, dtype=torch.float32)

    def fit(self, X):
        X = self._to_tensor(X)
        self.mean_ = X.mean(dim=0, keepdim=True)
        # Add unbiased=False to match sklearn's behavior
        self.std_ = X.std(dim=0, keepdim=True, unbiased=False) 
        return self

    def transform(self, X):
        X = self._to_tensor(X)
        return (X - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = self._to_tensor(X)
        return X * (self.std_ + 1e-8) + self.mean_