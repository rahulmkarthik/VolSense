# volsense_pkg/models/lstm_forecaster.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MultiVolDataset(Dataset):
    """
    Creates rolling windows of realised volatility for one or more forecast horizons,
    with built-in scaling/normalisation.

    If 'horizon' is an int, returns a single y per sample.
    If 'horizon' is a list/tuple, returns a vector of y's per sample (multi-horizon).
    """
    def __init__(self, df: pd.DataFrame, window: int = 30, horizon=1, scaler=None):
        df = df.copy()

        # Standardize the date column
        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                raise KeyError(f"Expected a 'date' column. Got: {list(df.columns)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.window = window
        self.horizons = horizon if isinstance(horizon, (list, tuple)) else [horizon]

        # Fit/Use scaler
        self.scaler = scaler or StandardScaler()
        vols = df["realized_vol"].values.reshape(-1, 1).astype(np.float32)
        self.scaler.fit(vols)
        df["scaled_vol"] = self.scaler.transform(vols)

        self.samples = []
        max_h = max(self.horizons)

        for _, group in df.groupby("ticker"):
            series = group["scaled_vol"].values.astype(np.float32)
            for i in range(len(series) - window - max_h + 1):
                X = series[i:i + window]
                y_vals = [series[i + window + h - 1] for h in self.horizons]
                self.samples.append((X, y_vals))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y_vals = self.samples[idx]
        y_tensor = torch.tensor(y_vals, dtype=torch.float32)
        if y_tensor.numel() == 1:
            y_tensor = y_tensor.squeeze()  # keep old behaviour for single horizon
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), y_tensor

    def inverse_transform(self, arr):
        """
        Invert the scaling on predictions or labels.
        Accepts 1D or 2D numpy arrays or tensors.
        """
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.array(arr).reshape(-1, 1)
        return self.scaler.inverse_transform(arr).ravel()
    

class BaseLSTM(nn.Module):
    """
    Simple LSTM forecaster that outputs one or more horizons.
    If n_horizons=1, behaves exactly like the old version.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2, n_horizons=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_horizons)
        self.n_horizons = n_horizons

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # [batch, n_horizons]
        if self.n_horizons == 1:
            return out.squeeze(-1)  # keep old behaviour
        return out  # [batch, n_horizons]


def train_lstm(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    """
    Trains an LSTM Forecaster. Works for single or multi-horizon automatically.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item() * len(X)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader.dataset):.4f} "
              f"| Val Loss: {val_loss/len(val_loader.dataset):.4f}")

    return model


def evaluate_lstm(model, loader, device="cpu"):
    """
    Evaluates an LSTM Forecaster on a DataLoader. 
    Returns (preds, actuals) as np arrays. 
    Shapes:
      - single horizon: (N,)
      - multi horizon: (N, n_horizons)
    """
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y.numpy())
    return np.array(preds), np.array(actuals)