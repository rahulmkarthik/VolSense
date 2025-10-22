import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Tuple

def collate_with_dates(batch):
    """
    Custom collate function that ignores datetime stacking issues.
    Keeps date arrays as a plain list instead of converting to tensors.
    """
    xs, ys, dates = zip(*batch)  # unpack triplets
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys, list(dates)


# ============================================================
# 1) Training configuration
# ============================================================
@dataclass
class TrainConfig:
    """
    Configuration for BaseLSTM training.

    Notes:
    - Targets are standardized inside the Dataset using train set stats
      (y_mean, y_std). These are copied back onto cfg and reused by val/test.
    - For non-negative outputs (when NOT standardizing, e.g. raw vols),
      set output_activation="softplus". With standardized targets, use "none".
    """
    window: int = 30
    horizons: List[int] = None
    val_start: str = None
    target_col: str = "realized_vol"

    batch_size: int = 128
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "cpu"

    dropout: float = 0.2
    hidden_dim: int = 128
    num_layers: int = 3
    use_layernorm: bool = True
    use_attention: bool = True
    feat_dropout_p: float = 0.1
    residual_head: bool = True
    output_activation: str = "none"  # "none" | "softplus" | "relu"

    num_workers: int = 0
    pin_memory: bool = False


# ============================================================
# 2) Dataset
# ============================================================
class MultiVolDataset(Dataset):
    """
    Windowed multi-horizon dataset for single- or multi-ticker realized vol data.

    Expected columns in df:
        ["date", "ticker", <features...>, target_col]
    Behavior:
        • Sorts by (ticker, date)
        • Standardizes features with StandardScaler (train fit, val transform)
        • Standardizes target using train set y_mean/y_std; reuses for val
        • Drops constant features to avoid StandardScaler NaNs
        • Builds rolling windows of length `cfg.window` with multi-horizon labels
    """
    def __init__(self, df: pd.DataFrame, cfg: TrainConfig, scaler: StandardScaler = None):
        df = df.copy()

        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                raise KeyError("Expected a 'date' column in the input DataFrame.")
        if "ticker" not in df.columns:
            raise KeyError("Expected a 'ticker' column in the input DataFrame.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.window = cfg.window
        self.horizons = cfg.horizons if isinstance(cfg.horizons, (list, tuple)) else [cfg.horizons]
        self.scaler = scaler or StandardScaler()

        # ----- Target standardization (fit on train, reuse on val/test) -----
        self.is_train = scaler is None
        if self.is_train:
            self.y_mean = float(df[cfg.target_col].mean())
            self.y_std = float(df[cfg.target_col].std())
            # Persist onto cfg so val/test can reuse exactly the same stats
            setattr(cfg, "y_mean", self.y_mean)
            setattr(cfg, "y_std", self.y_std)
        else:
            self.y_mean = float(getattr(cfg, "y_mean"))
            self.y_std = float(getattr(cfg, "y_std"))

        # Standardize target; keep it unconstrained (can be negative)
        df[cfg.target_col] = (df[cfg.target_col] - self.y_mean) / (self.y_std + 1e-8)

        # ----- Feature prep -----
        feat_cols = [c for c in df.columns if c not in ["date", "ticker", cfg.target_col]]
        X = (
            df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

        const_cols = [c for c in X.columns if X[c].std() == 0]
        if const_cols:
            print(f"⚠️ Dropping constant features: {const_cols}")
            X = X.drop(columns=const_cols)
            df = df.drop(columns=const_cols)
            feat_cols = [c for c in feat_cols if c not in const_cols]

        # Fit/transform features
        if self.is_train:
            self.scaler.fit(X)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=feat_cols)
        df[feat_cols] = X_scaled

        # ----- Build samples -----
        self.feat_cols = feat_cols
        self.samples: List[Tuple[np.ndarray, List[float]]] = []
        self.sample_dates: List[pd.Timestamp] = []  # <--- added for chronological eval
        max_h = max(self.horizons)

        if len(df) < self.window + max_h:
            raise ValueError(
                f"Insufficient samples: need at least {self.window + max_h}, got {len(df)}."
            )

        for _, group in df.groupby("ticker"):
            arr = group[feat_cols].values.astype(np.float32)
            y_series = group[cfg.target_col].values.astype(np.float32)
            dates = group["date"].values
            for i in range(len(arr) - cfg.window - max_h + 1):
                X_window = arr[i:i + cfg.window]
                y_vals = [y_series[i + cfg.window + h - 1] for h in self.horizons]
                last_date = dates[i + cfg.window - 1]
                self.samples.append((X_window, y_vals))
                self.sample_dates.append(last_date)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        X, y_vals = self.samples[idx]
        date = self.sample_dates[idx]
        y = torch.tensor(y_vals, dtype=torch.float32)
        if y.numel() == 1:
            y = y.squeeze(0)
        return torch.tensor(X, dtype=torch.float32), y, date  # <--- now returns date


# ============================================================
# 3) Model
# ============================================================
class FeatureDropout(nn.Module):
    """Feature-wise dropout mask shared across time steps (stochastic feature erasing)."""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class BaseLSTM(nn.Module):
    """
    LSTM-based multi-horizon forecaster with optional self-attention and residual head.

    Input:  (B, W, F)   where W=window, F=features
    Output: (B, H)      where H=len(horizons)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        n_horizons: int = 1,
        use_layernorm: bool = True,
        use_attention: bool = True,
        feat_dropout_p: float = 0.1,
        residual_head: bool = True,
        output_activation: str = "none",  # "none" | "softplus" | "relu"
    ):
        super().__init__()
        self.use_attention = use_attention
        self.residual_head = residual_head
        self.n_horizons = n_horizons

        self.feat_dropout = FeatureDropout(feat_dropout_p)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.attn = (
            nn.MultiheadAttention(hidden_dim, num_heads=2, dropout=dropout, batch_first=True)
            if use_attention else None
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons),
        )
        self.res_proj = nn.Linear(hidden_dim, n_horizons) if residual_head else None

        if output_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0)
        elif output_activation == "relu":
            self.out_act = nn.ReLU()
        else:
            self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feat_dropout(x)
        out, _ = self.lstm(x)
        if self.attn is not None:
            out, _ = self.attn(out, out, out)
        out = self.ln(out)
        last = out[:, -1, :]
        yhat = self.head(last)
        if self.res_proj is not None:
            yhat = yhat + self.res_proj(last)
        yhat = self.out_act(yhat)
        return yhat.squeeze(-1) if self.n_horizons == 1 else yhat


# ============================================================
# 4) Training / evaluation utilities
# ============================================================
def build_splits(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_start = pd.to_datetime(cfg.val_start)
    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()
    return train_df, val_df


def build_dataloaders(df: pd.DataFrame, cfg: TrainConfig):
    train_df, val_df = build_splits(df, cfg)

    train_ds = MultiVolDataset(train_df, cfg)
    val_ds   = MultiVolDataset(val_df, cfg, scaler=train_ds.scaler)

    # Use custom collate_fn to safely handle datetime objects
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate_with_dates
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate_with_dates
    )

    return train_loader, val_loader, train_ds.feat_cols


def train_baselstm(df: pd.DataFrame, cfg: TrainConfig):
    train_loader, val_loader, feat_cols = build_dataloaders(df, cfg)

    model = BaseLSTM(
        input_dim=len(feat_cols),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        n_horizons=len(cfg.horizons),
        use_layernorm=cfg.use_layernorm,
        use_attention=cfg.use_attention,
        feat_dropout_p=cfg.feat_dropout_p,
        residual_head=cfg.residual_head,
        output_activation=cfg.output_activation,
    )

    device = cfg.device
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    crit = nn.MSELoss(reduction="mean")

    hist = {"train": [], "val": []}

    for ep in range(cfg.epochs):
        # ---------- Train ----------
        model.train()
        tr_sum, n_train = 0.0, 0
        for batch in train_loader:  # ✅ supports (X, y, date)
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y = batch
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            preds = model(X)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            y     = torch.nan_to_num(y,     nan=0.0, posinf=1e6, neginf=-1e6)

            loss = crit(preds, y)
            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = X.size(0)
            tr_sum += loss.item() * bs
            n_train += bs

        tr_loss = tr_sum / max(1, n_train)

        # ---------- Validate ----------
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:  # ✅ supports (X, y, date)
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                X, y = X.to(device), y.to(device)
                preds = model(X)
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = crit(preds, y)

                bs = X.size(0)
                val_sum += loss.item() * bs
                n_val += bs

        val_loss = val_sum / max(1, n_val)
        sched.step()

        hist["train"].append(tr_loss)
        hist["val"].append(val_loss)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep+1}/{cfg.epochs} | LR: {cur_lr:.2e} | Train: {tr_loss:.6f} | Val: {val_loss:.6f}")

    return model, hist, (train_loader, val_loader)


def evaluate_baselstm(model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: str = "cpu"):
    """
    Evaluate model on a DataLoader and return **de-standardized** predictions and targets.

    If cfg was trained with standardization (y_mean/y_std set), this function
    converts both preds and actuals back to the original volatility scale.

    Returns
    -------
    preds : np.ndarray, shape (N, H)
    actuals : np.ndarray, shape (N, H)
    dates : np.ndarray of pd.Timestamp for chronological plotting
    """
    model.eval()
    preds, actuals, dates = [], [], []
    with torch.no_grad():
        for batch in loader:  # ✅ supports (X, y, date)
            if len(batch) == 3:
                X, y, d = batch
                dates.extend(d)
            else:
                X, y = batch
            X = X.to(device)
            yhat = model(X).cpu().numpy()
            preds.append(yhat)
            actuals.append(y.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    if hasattr(cfg, "y_mean") and hasattr(cfg, "y_std"):
        preds   = preds   * cfg.y_std + cfg.y_mean
        actuals = actuals * cfg.y_std + cfg.y_mean

    return preds, actuals

# ============================================================
# 🔄 Unified Output Standardizer
# ============================================================
def standardize_outputs(
    dates,
    tickers,
    forecast_vols,
    realized_vols=None,
    model_name="UnknownModel",
    horizons=None,
):
    """
    Standardizes model outputs for evaluation/backtesting.

    Parameters
    ----------
    dates : list or array
        Forecast or validation dates.
    tickers : list or array
        Corresponding tickers.
    forecast_vols : np.ndarray
        Model-predicted volatilities, shape (N, H) or (N,).
    realized_vols : np.ndarray or list, optional
        True realized volatilities (same shape as forecast_vols).
    model_name : str
        Model identifier, e.g. 'BaseLSTM', 'GlobalVolForecaster', 'ARCHForecaster'.
    horizons : list[int] or None
        Forecast horizons, e.g. [1,5,10]. Defaults to range(H) if not provided.

    Returns
    -------
    pd.DataFrame
        Columns: ['date','ticker','horizon','forecast_vol','realized_vol','model']
    """

    dates = np.asarray(dates)
    tickers = np.asarray(tickers)
    forecast_vols = np.atleast_2d(forecast_vols)
    n, h = forecast_vols.shape
    if realized_vols is None:
        realized_vols = np.full_like(forecast_vols, np.nan)
    else:
        realized_vols = np.atleast_2d(realized_vols)

    if horizons is None:
        horizons = list(range(1, h + 1))

    records = []
    for i in range(n):
        for j, horizon in enumerate(horizons):
            records.append(
                dict(
                    date=pd.to_datetime(dates[i]),
                    ticker=tickers[i],
                    horizon=horizon,
                    forecast_vol=forecast_vols[i, j],
                    realized_vol=realized_vols[i, j],
                    model=model_name,
                )
            )
    return pd.DataFrame.from_records(records)

