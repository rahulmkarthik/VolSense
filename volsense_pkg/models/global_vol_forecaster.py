# volsense_pkg/models/global_vol_forecaster.py
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

EPS = 1e-6


# ---------------------------
# Robust per-ticker scalers
# ---------------------------
@dataclass
class RobustScaler1D:
    median_: float = 0.0
    mad_: float = 1.0  # median absolute deviation

    def fit(self, x: np.ndarray):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        self.median_, self.mad_ = float(med), float(mad)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.median_) / self.mad_

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        return x_scaled * self.mad_ + self.median_


# ---------------------------
# Dataset (global, multi-ticker)
# ---------------------------
class GlobalVolDataset(Dataset):
    """
    Expects df with columns: ['date', 'ticker', 'realized_vol'] (date = datetime64)
    Builds rolling windows of length `window` over log(realized_vol+eps) per ticker.

    Target: horizons (list[int]) steps ahead, in the SAME (scaled) space as X.
    Per-ticker robust scaling (median/MAD) on log-vol.

    Notes:
      - If your df also has 'return' you could extend X to include more features.
      - For CPU speed, we keep X = [log-vol] only.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 30,
        horizons: Iterable[int] | int = 1,
        stride: int = 2,
        ticker_to_id: Optional[Dict[str, int]] = None,
        scalers: Optional[Dict[str, RobustScaler1D]] = None,
        restrict_to_dates: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ):
        df = df.copy()
        if "date" not in df.columns:
            raise KeyError(f"Expected 'date' column, got: {list(df.columns)}")
        if "ticker" not in df.columns:
            raise KeyError(f"Expected 'ticker' column, got: {list(df.columns)}")
        if "realized_vol" not in df.columns:
            raise KeyError(f"Expected 'realized_vol' column, got: {list(df.columns)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.window = int(window)
        self.horizons = list(horizons) if isinstance(horizons, (list, tuple)) else [int(horizons)]
        self.max_h = max(self.horizons)
        self.stride = max(1, int(stride))

        # Ticker ids
        if ticker_to_id is None:
            uniq = df["ticker"].unique().tolist()
            self.ticker_to_id = {t: i for i, t in enumerate(uniq)}
        else:
            self.ticker_to_id = dict(ticker_to_id)

        # Build per-ticker scalers on log-vol
        self.scalers: Dict[str, RobustScaler1D] = scalers or {}
        self.samples: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (ticker_id, X[window,1], y[len(horizons)])

        # Build windows per ticker
        for tkr, g in df.groupby("ticker", sort=False):
            series = g["realized_vol"].values.astype(np.float32)
            logv = np.log(series + EPS)

            # Fit scaler if not provided
            sc = self.scalers.get(tkr)
            if sc is None:
                sc = RobustScaler1D().fit(logv)
                self.scalers[tkr] = sc
            logv_scaled = sc.transform(logv)

            # Optional date filter *on targets end date*
            dates = g["date"].values

            # Build windows
            N = len(logv_scaled)
            end = N - self.max_h
            i = 0
            while i + self.window <= end:
                # Targets defined at i + window + h - 1
                tgt_idx = i + self.window - 1
                # each horizon target index:
                y_idx = [tgt_idx + h for h in self.horizons]
                # ensure within bounds
                if y_idx[-1] < N:
                    # If date filter: keep samples whose LAST target date falls in range
                    keep = True
                    if restrict_to_dates is not None:
                        d0, d1 = restrict_to_dates
                        tgt_date = pd.Timestamp(dates[y_idx[-1]])
                        if (d0 is not None and tgt_date < d0) or (d1 is not None and tgt_date > d1):
                            keep = False
                    if keep:
                        Xw = logv_scaled[i : i + self.window].reshape(self.window, 1)  # [W,1]
                        yv = logv_scaled[y_idx].astype(np.float32)                    # [H]
                        self.samples.append((self.ticker_to_id[tkr], Xw, yv))
                i += self.stride

        if len(self.samples) == 0:
            raise ValueError("No samples constructed. Check dates/window/horizons/stride.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tkr_id, Xw, yv = self.samples[idx]
        return (
            torch.tensor(tkr_id, dtype=torch.long),
            torch.tensor(Xw, dtype=torch.float32),
            torch.tensor(yv if len(yv) > 1 else yv[0], dtype=torch.float32),
        )


# ---------------------------
# Model
# ---------------------------
class GlobalVolForecaster(nn.Module):
    """
    Ticker embedding + LSTM over past log-vol window → MLP head → H horizons (scaled log-vol space).
    """
    def __init__(
        self,
        n_tickers: int,
        window: int,
        n_horizons: int = 1,
        emb_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.window = window

        self.tok = nn.Embedding(n_tickers, emb_dim)
        self.lstm = nn.LSTM(
            input_size=1 + emb_dim,  # log-vol + broadcasted ticker embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, tkr_id: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        tkr_id: [B] long
        X:      [B, W, 1] (scaled log-vol)
        """
        B, W, _ = X.shape
        e = self.tok(tkr_id)                    # [B, E]
        e = e.unsqueeze(1).expand(B, W, -1)     # [B, W, E]
        x = torch.cat([X, e], dim=-1)           # [B, W, 1+E]
        out, _ = self.lstm(x)                   # [B, W, H]
        last = out[:, -1, :]                    # [B, H]
        yhat = self.head(last)                  # [B, n_horizons] (scaled log-vol)
        return yhat.squeeze(-1)                 # if n_horizons=1 → [B]


# ---------------------------
# Training / Eval helpers
# ---------------------------
@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    oversample_high_vol: bool = True
    early_stop_patience: int = 3
    device: str = "cpu"


def _make_loaders(
    train_ds: GlobalVolDataset,
    val_ds: GlobalVolDataset,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    # Optional oversampling of high-vol windows (by target at first horizon)
    if cfg.oversample_high_vol:
        with torch.no_grad():
            y0 = torch.stack([s[2] if isinstance(s[2], torch.Tensor) else torch.tensor(s[2]) for s in train_ds])  # [N,H] or [N]
        y0 = y0[:, 0] if y0.ndim == 2 else y0
        q75 = torch.quantile(y0, 0.75).item()
        weights = torch.where(y0 >= q75, 2.0, 1.0).numpy()  # upweight top quartile
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def train_global_model(
    model: GlobalVolForecaster,
    train_ds: GlobalVolDataset,
    val_ds: GlobalVolDataset,
    cfg: TrainConfig,
) -> Dict[str, List[float]]:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_loader, val_loader = _make_loaders(train_ds, val_ds, cfg)

    history = {"train": [], "val": []}
    best_val = math.inf
    patience = cfg.early_stop_patience
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        # ---- train
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for t_id, X, y in train_loader:
            t_id, X, y = t_id.to(device), X.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(t_id, X)
            if y.ndim == 1 and yhat.ndim == 0:
                yhat = yhat.unsqueeze(0)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(t_id)
            tr_n += len(t_id)

        # ---- val
        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for t_id, X, y in val_loader:
                t_id, X, y = t_id.to(device), X.to(device), y.to(device)
                yhat = model(t_id, X)
                loss = loss_fn(yhat, y)
                va_loss += loss.item() * len(t_id)
                va_n += len(t_id)

        tr_avg = tr_loss / max(1, tr_n)
        va_avg = va_loss / max(1, va_n)
        history["train"].append(tr_avg)
        history["val"].append(va_avg)
        print(f"Epoch {ep}/{cfg.epochs} | Train Loss: {tr_avg:.4f} | Val Loss: {va_avg:.4f}")

        # ---- early stopping
        if va_avg < best_val - 1e-6:
            best_val = va_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------
# Build splits and inference
# ---------------------------
def build_global_splits(
    df: pd.DataFrame,
    window: int = 30,
    horizons: Iterable[int] | int = 1,
    stride: int = 2,
    val_start: str | pd.Timestamp = "2025-01-01",
) -> Tuple[GlobalVolDataset, GlobalVolDataset, Dict[str, int], Dict[str, RobustScaler1D]]:
    """
    Time-based split: everything with target-date < val_start goes to train,
    >= val_start goes to val.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    val_start = pd.Timestamp(val_start)

    # We create *two datasets* over the same df, restricting by the target date range.
    # Reuse identical ticker->id and scalers so train and val live in the same space.
    tmp = GlobalVolDataset(df, window=window, horizons=horizons, stride=stride)
    t2i = tmp.ticker_to_id
    scalers = tmp.scalers

    train_ds = GlobalVolDataset(
        df, window=window, horizons=horizons, stride=stride,
        ticker_to_id=t2i, scalers=scalers,
        restrict_to_dates=(None, val_start - pd.Timedelta(days=1)),
    )
    val_ds = GlobalVolDataset(
        df, window=window, horizons=horizons, stride=stride,
        ticker_to_id=t2i, scalers=scalers,
        restrict_to_dates=(val_start, None),
    )
    return train_ds, val_ds, t2i, scalers


@torch.no_grad()
def predict_next_day(
    model: GlobalVolForecaster,
    df_last_windows: pd.DataFrame,
    ticker_to_id: Dict[str, int],
    scalers: Dict[str, RobustScaler1D],
    window: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    df_last_windows: one row per ticker with the *last* `window` log-vol values, columns:
        ['ticker', 'log_rv_0', 'log_rv_1', ... 'log_rv_{window-1}'] where 0 is oldest.
    Returns a DataFrame with columns ['ticker', 'pred_vol_{h}'] in original vol space.
    """
    model.eval()
    device = torch.device(device)
    rows = []
    for _, r in df_last_windows.iterrows():
        tkr = r["ticker"]
        t_id = torch.tensor([ticker_to_id[tkr]], dtype=torch.long, device=device)
        logw = r[[f"log_rv_{i}" for i in range(window)]].values.astype(np.float32)
        X = torch.tensor(logw.reshape(1, window, 1), dtype=torch.float32, device=device)
        yhat_scaled = model(t_id, X).cpu().numpy().reshape(-1)  # scaled log-vol
        # invert scaler + exp
        sc = scalers[tkr]
        yhat_log = sc.inverse_transform(yhat_scaled)
        vols = np.exp(yhat_log)  # back to vol
        out = {"ticker": tkr}
        for h, v in enumerate(vols, start=1):
            out[f"pred_vol_{h}"] = float(v)
        rows.append(out)
    return pd.DataFrame(rows)


def make_last_windows(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Helper to prepare `df_last_windows` for predict_next_day().
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date")
        logv = np.log(g["realized_vol"].values + EPS)
        if len(logv) < window:
            continue
        w = logv[-window:]
        row = {"ticker": tkr}
        for i, v in enumerate(w):
            row[f"log_rv_{i}"] = float(v)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------
# Checkpoint I/O
# ---------------------------
def save_checkpoint(
    path: str,
    model: GlobalVolForecaster,
    ticker_to_id: Dict[str, int],
    scalers: Dict[str, RobustScaler1D],
):
    obj = {
        "state_dict": model.state_dict(),
        "ticker_to_id": ticker_to_id,
        "scalers": {k: (v.median_, v.mad_) for k, v in scalers.items()},
        "window": model.window,
        "n_horizons": model.n_horizons,
        "n_tickers": model.tok.num_embeddings,
    }
    torch.save(obj, path)


def load_checkpoint(path: str, emb_dim=8, hidden_dim=64, num_layers=2, dropout=0.2, device="cpu"):
    obj = torch.load(path, map_location=device)
    n_tickers = obj["n_tickers"]
    window = obj["window"]
    n_horizons = obj["n_horizons"]

    model = GlobalVolForecaster(
        n_tickers=n_tickers,
        window=window,
        n_horizons=n_horizons,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(obj["state_dict"])

    # rebuild scalers
    scalers = {}
    for k, (med, mad) in obj["scalers"].items():
        sc = RobustScaler1D(median_=float(med), mad_=float(mad))
        scalers[k] = sc

    return model, obj["ticker_to_id"], scalers
