######BACKUP###########################################


# ============================================================
# volsense_pkg/models/global_vol_forecaster.py
# Enhanced GlobalVolForecaster with optional Attention + Residual Head
# Backward compatible with existing training code
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


# ============================================================
# 🧩 Config Dataclass
# ============================================================
@dataclass
class TrainConfig:
    window: int = 30
    horizons: int = 1
    stride: int = 1
    val_start: str = "2025-01-01"
    target_col: str = "realized_vol_log"
    extra_features: list = None
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 128
    oversample_high_vol: bool = False
    cosine_schedule: bool = False
    device: str = "cpu"


# ============================================================
# 🧠 GlobalVolForecaster Model
# ============================================================
class GlobalVolForecaster(nn.Module):
    def __init__(
        self,
        n_tickers: int,
        window: int,
        n_horizons: int = 1,
        emb_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        attention: bool = False,
        residual_head: bool = False,
        input_size: int = 1,  # ✅ NEW: safe default for LSTM
    ):
        super().__init__()

        self.window = window
        self.n_horizons = n_horizons
        self.emb_dim = emb_dim
        self.attention_enabled = attention
        self.residual_head = residual_head
        self.input_size = input_size

        # --- Embedding for ticker IDs
        self.tok = nn.Embedding(n_tickers, emb_dim)

        # --- Core LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.input_size + emb_dim,  # ✅ FIXED HERE
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Optional Attention
        if self.attention_enabled:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons),
        )

        if self.residual_head:
            self.residual = nn.Linear(hidden_dim, n_horizons)

    def forward(self, tkr_id, X):
        if X.ndim == 2:  # [B, F] → [B, 1, F]
            X = X.unsqueeze(1)

        B, W, F = X.shape
        emb = self.tok(tkr_id).unsqueeze(1).expand(B, W, -1)
        X = torch.cat([X, emb], dim=-1)

        out, _ = self.lstm(X)

        if self.attention_enabled:
            out, _ = self.attn(out, out, out)

        last = out[:, -1, :]
        yhat = self.head(last)

        if self.residual_head:
            yhat = yhat + self.residual(last)

        return yhat.squeeze(-1)



# ============================================================
# 🧮 Dataset Builder
# ============================================================
def build_global_splits(df, cfg):
    """
    Builds train/val datasets based on cfg parameters.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"])

    required_cols = ["date", "ticker", cfg.target_col, "return"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # --- Ticker encoding
    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)

    # --- Scale per ticker
    features = ["return"] + (cfg.extra_features or [])
    scalers = {}
    scaled_frames = []

    for t in tickers:
        sub = df[df["ticker"] == t].copy()
        scaler = StandardScaler()
        sub_features = sub[features].fillna(0.0)
        sub[features] = scaler.fit_transform(sub_features)
        scalers[t] = scaler
        scaled_frames.append(sub)

    df_scaled = pd.concat(scaled_frames, ignore_index=True)

    train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(drop=True)
    val_df = df_scaled[df_scaled["date"] >= cfg.val_start].reset_index(drop=True)

    class GlobalVolDataset(torch.utils.data.Dataset):
        def __init__(self, df, cfg):
            self.df = df
            self.cfg = cfg
            self.grouped = {t: g.reset_index(drop=True) for t, g in df.groupby("ticker_id")}
            self.samples = []
            for tid, g in self.grouped.items():
                for i in range(0, len(g) - cfg.window - cfg.horizons, cfg.stride):
                    self.samples.append((tid, i))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            tid, start = self.samples[idx]
            g = self.grouped[tid]
            X = g.iloc[start:start + self.cfg.window]
            y = g.iloc[start + self.cfg.window + self.cfg.horizons - 1][self.cfg.target_col]
            feats = ["return"] + (self.cfg.extra_features or [])
            X = torch.tensor(X[feats].values, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            tid = torch.tensor(tid, dtype=torch.long)
            return tid, X, y

    train_ds = GlobalVolDataset(train_df, cfg)
    val_ds = GlobalVolDataset(val_df, cfg)

    return train_ds, val_ds, ticker_to_id, scalers


# ============================================================
# 🚀 Training Loop
# ============================================================
def train_global_model(df, cfg):
    """
    Unified training entrypoint — builds splits, model, trains and returns artifacts.
    """
    from torch.utils.data import DataLoader

    # Build datasets
    train_ds, val_ds, ticker_to_id, scalers = build_global_splits(df, cfg)

    # Create model
    features = ["return"] + (cfg.extra_features or [])

    model = GlobalVolForecaster(
        n_tickers=len(ticker_to_id),
        window=cfg.window,
        n_horizons=cfg.horizons,
        emb_dim=16,
        hidden_dim=160,
        num_layers=3,
        dropout=0.2,
        attention=True,
        residual_head=True,
        input_size=len(features),  # ✅ Pass actual feature count
    )

    features = ["return"] + (cfg.extra_features or [])

    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    if cfg.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    history = {"train": [], "val": []}
    print(f"\n🚀 Training GlobalVolForecaster on {len(ticker_to_id)} tickers...\n")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for t_id, X, y in train_loader:
            t_id, X, y = t_id.to(device), X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(t_id, X)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(t_id)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_id, X, y in val_loader:
                t_id, X, y = t_id.to(device), X.to(device), y.to(device)
                yhat = model(t_id, X)
                loss = criterion(yhat, y)
                val_loss += loss.item() * len(t_id)
        val_loss /= len(val_loader.dataset)
        if scheduler:
            scheduler.step()

        print(f"Epoch {ep}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        history["train"].append(train_loss)
        history["val"].append(val_loss)

    print(f"\n✅ Training complete with feature set: {features}\n")
    return model, history, val_loader, ticker_to_id, scalers, features

