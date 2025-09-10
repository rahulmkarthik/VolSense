# volsense_pkg/models/lstm_forecaster.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# volsense_pkg/models/lstm_forecaster.py

class MultiVolDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int = 30, horizon: int = 1):
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
        self.horizon = horizon
        self.samples = []

        for _, group in df.groupby("ticker"):
            series = group["realized_vol"].values.astype(np.float32)
            for i in range(len(series) - window - horizon + 1):
                X = series[i:i + window]
                y = series[i + window + horizon - 1]
                self.samples.append((X, y))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X).unsqueeze(-1), torch.tensor(y)



class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last hidden state
        return out.squeeze()


def train_lstm(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
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
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y.numpy())
    return np.array(preds), np.array(actuals)
