# ============================================================
# 🌍 Global Vol Forecaster — Simplified, Auto-Configurable Version
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class GlobalVolDataset(Dataset):
    def __init__(self, df, window, horizons, stride, features, target_col, ticker_to_id):
        self.df = df
        self.window = window
        self.horizons = horizons if isinstance(horizons, int) else horizons[0]
        self.stride = stride
        self.features = features
        self.target_col = target_col
        self.ticker_to_id = ticker_to_id

        self.samples = []
        for tkr, g in df.groupby("ticker"):
            vals = g[self.features + [self.target_col]].values
            tid = ticker_to_id[tkr]
            for i in range(0, len(vals) - window - self.horizons, stride):
                x = vals[i:i + window, :-1]           # features only
                y = vals[i + window + self.horizons - 1, -1]
                self.samples.append((tid, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tid, x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        tid = torch.tensor(tid, dtype=torch.long)
        return tid, x, y


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class GlobalVolForecaster(nn.Module):
    def __init__(self, n_tickers, input_dim, hidden_dim=128, num_layers=3,
                 dropout=0.1, emb_dim=12, n_horizons=1):
        super().__init__()
        self.tok = nn.Embedding(n_tickers, emb_dim)
        self.lstm = nn.LSTM(input_size=input_dim + emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_horizons)
        )

    def forward(self, tkr_id, X):
        if X.ndim == 2:  # [B, F]
            X = X.unsqueeze(1)
        B, W, F = X.shape
        emb = self.tok(tkr_id).unsqueeze(1).expand(B, W, -1)
        x = torch.cat([X, emb], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ------------------------------------------------------------
# Config Dataclass
# ------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 5e-4
    batch_size: int = 256
    device: str = "cpu"
    val_start: str = "2025-01-01"
    window: int = 40
    horizons: int = 1
    stride: int = 1
    target_col: str = "realized_vol_log"
    extra_features: list = None


# ------------------------------------------------------------
# Build splits and scalers automatically
# ------------------------------------------------------------
def build_global_splits(df, cfg):
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # Default features (return + extras)
    features = ["return"]
    if cfg.extra_features:
        features += cfg.extra_features

    # Per-ticker scaling
    scalers = {}
    scaled_frames = []
    for tkr, g in df.groupby("ticker"):
        sc_x, sc_y = StandardScaler(), StandardScaler()
        x_scaled = sc_x.fit_transform(g[features])
        y_scaled = sc_y.fit_transform(g[[cfg.target_col]])
        g_scaled = g.copy()
        for i, f in enumerate(features):
            g_scaled[f] = x_scaled[:, i]
        g_scaled[cfg.target_col] = y_scaled
        scaled_frames.append(g_scaled)
        scalers[tkr] = {"x": sc_x, "y": sc_y}

    df_scaled = pd.concat(scaled_frames, ignore_index=True)
    ticker_to_id = {tkr: i for i, tkr in enumerate(df_scaled["ticker"].unique())}

    train_df = df_scaled[df_scaled["date"] < cfg.val_start]
    val_df = df_scaled[df_scaled["date"] >= cfg.val_start]

    train_ds = GlobalVolDataset(train_df, cfg.window, cfg.horizons, cfg.stride,
                                features, cfg.target_col, ticker_to_id)
    val_ds = GlobalVolDataset(val_df, cfg.window, cfg.horizons, cfg.stride,
                              features, cfg.target_col, ticker_to_id)
    return train_ds, val_ds, ticker_to_id, scalers, features


# ------------------------------------------------------------
# Unified training loop
# ------------------------------------------------------------
def train_global_model(df, cfg):
    train_ds, val_ds, ticker_to_id, scalers, features = build_global_splits(df, cfg)

    model = GlobalVolForecaster(
        n_tickers=len(ticker_to_id),
        input_dim=len(features),
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        emb_dim=12,
        n_horizons=1
    ).to(cfg.device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train": [], "val": []}
    print(f"\n🚀 Training GlobalVolForecaster on {len(ticker_to_id)} tickers...\n")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for tid, X, y in train_loader:
            tid, X, y = tid.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            pred = model(tid, X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(tid)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for tid, X, y in val_loader:
                tid, X, y = tid.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
                pred = model(tid, X)
                loss = loss_fn(pred, y)
                vl_loss += loss.item() * len(tid)
        vl_loss /= len(val_loader.dataset)

        print(f"Epoch {ep}/{cfg.epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}")
        history["train"].append(tr_loss)
        history["val"].append(vl_loss)

    print("\n✅ Training complete.")
    return model, history, val_loader, ticker_to_id, scalers, features
