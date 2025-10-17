######BACKUP OF SINGLE HORIZON v5 and MULTI HORIZON V2######


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

# Backup of v2
# ============================================================
# volsense_pkg/models/global_vol_forecaster.py
# Enhanced GlobalVolForecaster with optional Attention + Residual Head
# Backward compatible with existing training code
# ============================================================

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import json
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


# ============================================================
# 🧩 Config Dataclass
# ============================================================
@dataclass
class TrainConfig:
    window: int = 30
    horizons: int | list = 1
    stride: int = 1
    val_start: str = "2025-01-01"
    target_col: str = "realized_vol_log"
    extra_features: list | None = None
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 128
    oversample_high_vol: bool = False
    cosine_schedule: bool = False
    device: str = "cpu"

    # 🔧 New knobs (all optional / sane defaults)
    dropout: float = 0.3                 # stronger regularization
    use_layernorm: bool = True           # layer norm on LSTM outputs
    separate_heads: bool = True          # per-horizon decoders
    loss_horizon_weights: list | None = None  # e.g., [0.5, 0.3, 0.2]
    dynamic_window_jitter: int = 0       # e.g., 5 → sample window∈[W-5,W+5] for train
    grad_clip: float = 1.0               # gradient clipping
    num_workers: int = 0                 # speedup if you have CPU cores
    pin_memory: bool = False


# ============================================================
# 🧠 GlobalVolForecaster Model
# ============================================================
class GlobalVolForecaster(nn.Module):
    def __init__(
        self,
        n_tickers,
        window,
        n_horizons,
        emb_dim,
        hidden_dim,
        num_layers,
        dropout,
        attention=False,
        residual_head=False,
        input_size=None,
        use_layernorm=True,
        separate_heads=True,
    ):
        super().__init__()
        self.window = window
        self.input_size = input_size
        self.attention_enabled = attention
        self.residual_head = residual_head
        self.use_layernorm = use_layernorm
        self.separate_heads = separate_heads

        # Ticker embedding
        self.tok = nn.Embedding(n_tickers, emb_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.input_size + emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Optional temporal attention
        if self.attention_enabled:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )

        self.ln = nn.LayerNorm(hidden_dim) if self.use_layernorm else nn.Identity()

        # How many horizons?
        self.n_horizons = n_horizons if isinstance(n_horizons, int) else len(n_horizons)

        # Decoders
        if self.separate_heads and self.n_horizons > 1:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                ) for _ in range(self.n_horizons)
            ])
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.n_horizons),
            )

        if self.residual_head:
            self.residual = nn.Linear(hidden_dim, self.n_horizons)

    def forward(self, tkr_id, X):
        if X.ndim == 2:
            X = X.unsqueeze(1)

        B, W, F = X.shape
        emb = self.tok(tkr_id).unsqueeze(1).expand(B, W, -1)
        X = torch.cat([X, emb], dim=-1)

        out, _ = self.lstm(X)
        if self.attention_enabled:
            out, _ = self.attn(out, out, out)
        out = self.ln(out)

        last = out[:, -1, :]

        if self.separate_heads and self.n_horizons > 1:
            ys = [head(last) for head in self.heads]      # list of [B,1]
            yhat = torch.cat(ys, dim=-1)                  # [B, H]
        else:
            yhat = self.head(last)                        # [B, H]

        if self.residual_head:
            yhat = yhat + self.residual(last)

        if self.n_horizons == 1:
            return yhat.squeeze(-1)
        return yhat



# ============================================================
# 🧮 Dataset Builder
# ============================================================
def build_global_splits(df: pd.DataFrame, cfg: TrainConfig):
    """
    Builds train/val datasets based on cfg parameters.
    Scales features per ticker (NOT the target).
    Returns: train_ds, val_ds, ticker_to_id, scalers
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"])

    required_cols = ["date", "ticker", cfg.target_col, "return"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # --- Encode tickers
    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)

    # --- Feature set (per-ticker standardization)
    features = ["return"] + (cfg.extra_features or [])
    scalers: dict[str, StandardScaler] = {}
    scaled_frames = []

    for t in tickers:
        sub = df[df["ticker"] == t].copy()
        scaler = StandardScaler()
        sub_features = sub[features].astype(float).fillna(0.0)
        sub[features] = scaler.fit_transform(sub_features)
        scalers[t] = scaler
        scaled_frames.append(sub)

    df_scaled = pd.concat(scaled_frames, ignore_index=True)

    # --- Train / Val split
    train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(drop=True)
    val_df   = df_scaled[df_scaled["date"] >= cfg.val_start].reset_index(drop=True)

    # --- Dataset
    class GlobalVolDataset(torch.utils.data.Dataset):
        def __init__(self, df, cfg, is_train=False):
            self.df = df
            self.cfg = cfg
            self.is_train = is_train
            self.grouped = {t: g.reset_index(drop=True) for t, g in df.groupby("ticker_id")}
            self.samples = []
            horizon_max = max(cfg.horizons) if isinstance(cfg.horizons, (list, tuple)) else cfg.horizons
            for tid, g in self.grouped.items():
                for i in range(0, len(g) - cfg.window - horizon_max, cfg.stride):
                    self.samples.append((tid, i))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            tid, start = self.samples[idx]
            g = self.grouped[tid]

            # --- Window jitter ---
            W = self.cfg.window
            if self.is_train and self.cfg.dynamic_window_jitter > 0:
                delta = np.random.randint(-self.cfg.dynamic_window_jitter,
                                        self.cfg.dynamic_window_jitter + 1)
                W = max(10, self.cfg.window + int(delta))  # clamp small windows

            feats = ["return"] + (self.cfg.extra_features or [])
            X_df = g.iloc[start:start + W]

            # --- Targets ---
            if isinstance(self.cfg.horizons, int):
                t_idx = start + W + self.cfg.horizons - 1
                t_idx = min(t_idx, len(g) - 1)
                y_vals = [g.iloc[t_idx][self.cfg.target_col]]
            else:
                y_vals = []
                for h in self.cfg.horizons:
                    t_idx = start + W + h - 1
                    y_vals.append(g.iloc[t_idx][self.cfg.target_col] if t_idx < len(g) else np.nan)

            X = torch.tensor(X_df[feats].values, dtype=torch.float32)
            y = torch.tensor(y_vals, dtype=torch.float32)
            y = torch.nan_to_num(y, nan=0.0)

            # --- ✅ Pad to fixed length (cfg.window) ---
            if W < self.cfg.window:
                pad_len = self.cfg.window - W
                pad = torch.zeros((pad_len, X.shape[1]), dtype=torch.float32)
                X = torch.cat([X, pad], dim=0)
            elif W > self.cfg.window:
                X = X[:self.cfg.window]

            return torch.tensor(tid, dtype=torch.long), X, y



    train_ds = GlobalVolDataset(train_df, cfg, is_train=True)
    val_ds   = GlobalVolDataset(val_df, cfg, is_train=False)
    return train_ds, val_ds, ticker_to_id, scalers


# ============================================================
# 🚀 Training Loop
# ============================================================
def train_global_model(df: pd.DataFrame, cfg: TrainConfig):
    """
    Unified training entrypoint — builds splits, model, trains and returns artifacts.
    Returns: model, history, val_loader, ticker_to_id, scalers, features
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
        dropout=cfg.dropout,           # <- use cfg
        attention=True,
        residual_head=True,
        input_size=len(features),      # features only (embedding is added inside the model)
        use_layernorm=cfg.use_layernorm,
        separate_heads=cfg.separate_heads,
    )


    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def mh_mse(pred, target):
        # pred/target: [B, H] or [B]
        if pred.ndim == 1:  # single horizon
            return torch.mean((pred - target) ** 2)
        w = cfg.loss_horizon_weights
        if w is None:
            return torch.mean((pred - target) ** 2)
        w = torch.tensor(w, device=pred.device, dtype=pred.dtype)  # [H]
        # broadcast to [B,H], average across batch
        return torch.mean(((pred - target) ** 2) * w)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6) \
                if cfg.cosine_schedule else None


    history = {"train": [], "val": []}
    print(f"\n🚀 Training GlobalVolForecaster on {len(ticker_to_id)} tickers...\n")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for t_id, X, y in train_loader:
            t_id, X, y = t_id.to(device), X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(t_id, X)
            loss = mh_mse(yhat, y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item() * len(t_id)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_id, X, y in val_loader:
                t_id, X, y = t_id.to(device), X.to(device), y.to(device)
                yhat = model(t_id, X)
                val_loss += mh_mse(yhat, y).item() * len(t_id)
        val_loss /= len(val_loader.dataset)

        if scheduler: scheduler.step()

        print(f"Epoch {ep}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        history["train"].append(train_loss); history["val"].append(val_loss)


    print(f"\n✅ Training complete with feature set: {features}\n")
    return model, history, val_loader, ticker_to_id, scalers, features


# ============================================================
# 🔹 Helper Utilities for Inference and Checkpointing
# ============================================================

def make_last_windows(df: pd.DataFrame, window: int):
    """
    Extract last `window` timesteps per ticker for prediction.
    """
    df = df.copy().sort_values(["ticker", "date"])
    last_windows = []
    for t, g in df.groupby("ticker"):
        g = g.tail(window).copy()
        g["ticker"] = t
        last_windows.append(g)
    return pd.concat(last_windows, ignore_index=True)


def predict_next_day(model, df_last_windows, ticker_to_id, scalers, window, device="cpu"):
    """
    Run inference for the next horizon(s) per ticker.
    """
    model.eval()
    preds = []
    features = [c for c in df_last_windows.columns if c not in ["date", "ticker", "realized_vol", "realized_vol_log"]]

    with torch.no_grad():
        for t, g in df_last_windows.groupby("ticker"):
            t_id = torch.tensor([ticker_to_id[t]], dtype=torch.long, device=device)
            scaler = scalers[t]
            X_scaled = scaler.transform(g[features].fillna(0.0))
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(0)
            yhat = model(t_id, X_tensor)
            preds.append({"ticker": t, "forecast_vol_scaled": yhat.cpu().numpy().flatten().tolist()})

    out = pd.DataFrame(preds)
    return out


def save_checkpoint(path, model, ticker_to_id, scalers):
    """
    Save model weights + metadata.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    meta = {"tickers": list(ticker_to_id.keys())}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    scaler_path = os.path.splitext(path)[0] + "_scalers.pt"
    torch.save(scalers, scaler_path)
    print(f"💾 Checkpoint saved: {path}, {meta_path}, {scaler_path}")


def load_checkpoint(path, device="cpu"):
    """
    Load model + metadata.
    """
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    scaler_path = os.path.splitext(path)[0] + "_scalers.pt"

    with open(meta_path, "r") as f:
        meta = json.load(f)
    scalers = torch.load(scaler_path, map_location=device)

    n_tickers = len(meta["tickers"])
    model = GlobalVolForecaster(
        n_tickers=n_tickers,
        window=30,
        n_horizons=1,
        emb_dim=16,
        hidden_dim=160,
        num_layers=3,
        dropout=0.2,
        attention=True,
        residual_head=True,
        input_size=1,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    ticker_to_id = {t: i for i, t in enumerate(meta["tickers"])}
    print(f"✅ Loaded checkpoint for {n_tickers} tickers from {path}")
    return model, ticker_to_id, scalers


