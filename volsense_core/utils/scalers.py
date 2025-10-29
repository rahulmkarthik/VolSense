import torch

class TorchStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        self.mean_ = X.mean(dim=0, keepdim=True)
        self.std_ = X.std(dim=0, keepdim=True)
        return self

    def transform(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        return (X - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        return X * (self.std_ + 1e-8) + self.mean_
