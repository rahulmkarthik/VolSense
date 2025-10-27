# ============================================================
# volsense_pkg/models/global_vol_forecaster.py  (v3)
# GlobalVolForecaster with attention, per-horizon heads,
# feature-dropout, variational (recurrent) dropout, EMA weights,
# cosine restarts, early stopping — still backward-compatible.
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Dict


# ============================================================
# 🧩 Config Dataclass
# ============================================================
@dataclass
class TrainConfig:
    window: int = 30
    horizons: int | list = 1
    stride: int = 2                     # 👈 default stride=3 for decorrelation
    val_start: str = "2025-01-01"
    val_end: str | None = None        # optional: end of validation window (exclusive)
    val_mode: str = "causal"          # "causal" or "holdout_slice"
    embargo_days: int = 0             # drop these days on either side of val window from train
    target_col: str = "realized_vol_log"
    extra_features: list | None = None

    # training
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 256
    device: str = "cpu"

    # regularization / arch
    dropout: float = 0.3                # LSTM inter-layer dropout
    use_layernorm: bool = True
    separate_heads: bool = True         # per-horizon decoders
    attention: bool = True
    residual_head: bool = True

    # v3 knobs
    feat_dropout_p: float = 0.1         # randomly drop feature channels (input)
    variational_dropout_p: float = 0.1  # recurrent-style mask on inputs per batch
    loss_horizon_weights: list | None = None  # e.g. [0.6, 0.25, 0.15]

    # v4 knobs
    dynamic_window_jitter: int = 0
    grad_clip: float = 1.0

    cosine_schedule: bool = True
    cosine_restarts: bool = True        # use SGDR
    oversample_high_vol: bool = False   # kept for compat; not used by default

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.995

    # early stop
    early_stop: bool = True
    patience: int = 4
    min_delta: float = 0.0

    # dataloader
    num_workers: int = 0
    pin_memory: bool = False


# ============================================================
# 🧠 Utility modules (feature dropout / variational dropout)
# ============================================================
class FeatureDropout(nn.Module):
    """Channel-wise (feature) dropout: same mask for all time-steps."""
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, F]
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device, dtype=x.dtype).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class VariationalInputDropout(nn.Module):
    """Variational (recurrent) dropout on inputs — same mask across time."""
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device, dtype=x.dtype).bernoulli_(1 - self.p)
        return x * mask


# ============================================================
# 🧠 GlobalVolForecaster Model
# ============================================================
class GlobalVolForecaster(nn.Module):
    def __init__(
        self,
        n_tickers: int,
        window: int,
        n_horizons: int | list,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attention: bool = True,
        residual_head: bool = True,
        input_size: int | None = None,
        use_layernorm: bool = True,
        separate_heads: bool = True,
        feat_dropout_p: float = 0.1,
        variational_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.input_size = int(input_size) if input_size is not None else 1
        self.attention_enabled = attention
        self.residual_head = residual_head
        self.use_layernorm = use_layernorm
        self.separate_heads = separate_heads

        # Input regularizers
        self.feat_do = FeatureDropout(feat_dropout_p)
        self.var_do = VariationalInputDropout(variational_dropout_p)

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

        # Horizons
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

    def forward(self, tkr_id: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # X: [B, W, F] or [B, F]
        if X.ndim == 2:
            X = X.unsqueeze(1)

        B, W, F = X.shape
        X = self.feat_do(X)
        X = self.var_do(X)

        emb = self.tok(tkr_id).unsqueeze(1).expand(B, W, -1)
        X = torch.cat([X, emb], dim=-1)  # [B, W, F+E]

        out, _ = self.lstm(X)            # [B, W, H]
        if self.attention_enabled:
            out, _ = self.attn(out, out, out)
        out = self.ln(out)

        last = out[:, -1, :]             # [B, H]

        if self.separate_heads and self.n_horizons > 1:
            ys = [head(last) for head in self.heads]   # list of [B,1]
            yhat = torch.cat(ys, dim=-1)               # [B, H]
        else:
            yhat = self.head(last)                     # [B, H]

        if self.residual_head:
            yhat = yhat + self.residual(last)

        return yhat.squeeze(-1) if self.n_horizons == 1 else yhat


# ============================================================
# 🧮 Dataset Builder (per-ticker feature scaling; no target scaling)
# ============================================================
def build_global_splits(df: pd.DataFrame, cfg: TrainConfig):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    required = ["date", "ticker", cfg.target_col, "return"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)

    features = ["return"] + (cfg.extra_features or [])
    scalers: Dict[str, StandardScaler] = {}
    scaled_parts = []
    for t in tickers:
        sub = df[df["ticker"] == t].copy()
        scaler = StandardScaler()
        sub[features] = scaler.fit_transform(sub[features].astype(float).fillna(0.0))
        scalers[t] = scaler
        scaled_parts.append(sub)
    df_scaled = pd.concat(scaled_parts, ignore_index=True)

    # inside build_global_splits(...) after df_scaled is built
    df_scaled["date"] = pd.to_datetime(df_scaled["date"])
    val_end = getattr(cfg, "val_end", None)
    val_mode = getattr(cfg, "val_mode", "causal")
    embargo = int(getattr(cfg, "embargo_days", 0))

    if val_end is None:
        # original behavior: single cut by val_start
        train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(drop=True)
        val_df   = df_scaled[df_scaled["date"] >= cfg.val_start].reset_index(drop=True)
    else:
        mask_val = (df_scaled["date"] >= cfg.val_start) & (df_scaled["date"] < val_end)

        if val_mode == "holdout_slice":
            # train on everything except the held-out slice (+/- embargo window)
            if embargo > 0:
                # widen the excluded region by embargo on both sides
                lo = pd.to_datetime(cfg.val_start) - pd.Timedelta(days=embargo)
                hi = pd.to_datetime(val_end) + pd.Timedelta(days=embargo)
                mask_excl = (df_scaled["date"] >= lo) & (df_scaled["date"] < hi)
            else:
                mask_excl = mask_val
            train_df = df_scaled[~mask_excl].reset_index(drop=True)
            val_df   = df_scaled[mask_val].reset_index(drop=True)

        else:  # "causal"
            # forward test flavor: train strictly before val_start; validate within [start, end)
            train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(drop=True)
            val_df   = df_scaled[mask_val].reset_index(drop=True)


    class GlobalVolDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame, cfg: TrainConfig, is_train: bool):
            self.df = df
            self.cfg = cfg
            self.is_train = is_train
            self.groups = {tid: g.reset_index(drop=True) for tid, g in df.groupby("ticker_id")}
            self.samples = []
            Hmax = max(cfg.horizons) if isinstance(cfg.horizons, (list, tuple)) else cfg.horizons
            for tid, g in self.groups.items():
                for i in range(0, len(g) - cfg.window - Hmax, cfg.stride):
                    self.samples.append((tid, i))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            tid, start = self.samples[idx]
            g = self.groups[tid]

            W = self.cfg.window
            if self.is_train and hasattr(self.cfg, "dynamic_window_jitter") and self.cfg.dynamic_window_jitter > 0:
                delta = np.random.randint(-self.cfg.dynamic_window_jitter,
                                          self.cfg.dynamic_window_jitter + 1)
                W = max(10, self.cfg.window + int(delta))

            feats = ["return"] + (self.cfg.extra_features or [])
            X_df = g.iloc[start:start + W][feats]

            # Build target(s)
            if isinstance(self.cfg.horizons, int):
                t_idx = min(start + W + self.cfg.horizons - 1, len(g) - 1)
                y_vals = [g.iloc[t_idx][self.cfg.target_col]]
            else:
                y_vals = []
                for h in self.cfg.horizons:
                    t_idx = start + W + h - 1
                    y_vals.append(g.iloc[t_idx][self.cfg.target_col] if t_idx < len(g) else np.nan)

            X = torch.tensor(X_df.values, dtype=torch.float32)
            # --- Pad or crop to fixed cfg.window ---
            if X.shape[0] < self.cfg.window:
                pad = torch.zeros(self.cfg.window - X.shape[0], X.shape[1], dtype=torch.float32)
                X = torch.cat([X, pad], dim=0)
            elif X.shape[0] > self.cfg.window:
                X = X[-self.cfg.window:]

            y = torch.tensor(y_vals, dtype=torch.float32)
            y = torch.nan_to_num(y, nan=0.0)

            return torch.tensor(tid, dtype=torch.long), X, y

    train_ds = GlobalVolDataset(train_df, cfg, is_train=True)
    val_ds   = GlobalVolDataset(val_df, cfg, is_train=False)
    return train_ds, val_ds, ticker_to_id, scalers


# ============================================================
# 🔁 EMA helper
# ============================================================
def clone_model_like(model: nn.Module) -> nn.Module:
    ema = type(model)(**model._init_args) if hasattr(model, "_init_args") else None
    if ema is None:
        # generic fallback: deep copy of state dict to identically-constructed model
        import copy
        ema = copy.deepcopy(model)
    for p in ema.parameters(): p.requires_grad_(False)
    return ema

def update_ema(ema: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        for (n_e, p_e), (_, p_m) in zip(ema.named_parameters(), model.named_parameters()):
            p_e.data.mul_(decay).add_(p_m.data, alpha=1 - decay)
        for (n_e, b_e), (_, b_m) in zip(ema.named_buffers(), model.named_buffers()):
            b_e.data.copy_(b_m.data)


# ============================================================
# 🚀 Training Loop (no manual loops outside; returns artifacts)
# ============================================================
def train_global_model(df: pd.DataFrame, cfg: TrainConfig):
    from torch.utils.data import DataLoader
    # Datasets / loaders
    train_ds, val_ds, ticker_to_id, scalers = build_global_splits(df, cfg)
    features = ["return"] + (cfg.extra_features or [])

    model = GlobalVolForecaster(
        n_tickers=len(ticker_to_id),
        window=cfg.window,
        n_horizons=cfg.horizons,
        emb_dim=16,
        hidden_dim=160,
        num_layers=3,
        dropout=cfg.dropout,
        attention=cfg.attention,
        residual_head=cfg.residual_head,
        input_size=len(features),
        use_layernorm=cfg.use_layernorm,
        separate_heads=cfg.separate_heads,
        feat_dropout_p=cfg.feat_dropout_p,
        variational_dropout_p=cfg.variational_dropout_p,
    )
    # Save init args for EMA clone
    model._init_args = dict(
        n_tickers=len(ticker_to_id), window=cfg.window, n_horizons=cfg.horizons,
        emb_dim=16, hidden_dim=160, num_layers=3, dropout=cfg.dropout,
        attention=cfg.attention, residual_head=cfg.residual_head,
        input_size=len(features), use_layernorm=cfg.use_layernorm,
        separate_heads=cfg.separate_heads,
        feat_dropout_p=cfg.feat_dropout_p, variational_dropout_p=cfg.variational_dropout_p
    )

    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # scheduler
    if cfg.cosine_schedule and cfg.cosine_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_anneal := max(2, cfg.epochs // 4), eta_min=1e-6
        )
    elif cfg.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # multi-horizon MSE with optional horizon weights
    def mh_mse(pred, target):
        if pred.ndim == 1 or pred.shape[-1] == 1:
            return torch.mean((pred - target.squeeze(-1))**2)
        if cfg.loss_horizon_weights is None:
            return torch.mean((pred - target)**2)
        w = torch.tensor(cfg.loss_horizon_weights, device=pred.device, dtype=pred.dtype)
        return torch.mean(((pred - target)**2) * w)

    # EMA
    ema_model = clone_model_like(model) if cfg.use_ema else None
    if ema_model is not None:
        ema_model.load_state_dict(model.state_dict())
        ema_model.to(device)    # ✅ move EMA model to same device as main model


    # early stopping
    best_val = float("inf")
    best_sd = None
    patience_left = cfg.patience

    history = {"train": [], "val": []}
    print(f"\n🚀 Training GlobalVolForecaster on {len(ticker_to_id)} tickers...\n")

    for ep in range(1, cfg.epochs + 1):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        for t_id, X, y in train_loader:
            t_id, X, y = t_id.to(device), X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(t_id, X)
            loss = mh_mse(yhat, y)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema_model is not None:
                update_ema(ema_model, model, cfg.ema_decay)
            train_loss += loss.item() * len(t_id)
        train_loss /= len(train_loader.dataset)

        # ----- Validate -----
        def _eval(m):
            m.eval()
            vl = 0.0
            with torch.no_grad():
                for t_id, X, y in val_loader:
                    t_id, X, y = t_id.to(device), X.to(device), y.to(device)
                    yhat = m(t_id, X)
                    vl += mh_mse(yhat, y).item() * len(t_id)
            return vl / len(val_loader.dataset)

        val_loss = _eval(model)
        if ema_model is not None:
            val_loss = min(val_loss, _eval(ema_model))  # pick better of EMA/current

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(ep - 1)  # epoch-based step for restarts
            else:
                scheduler.step()

        print(f"Epoch {ep}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        history["train"].append(train_loss); history["val"].append(val_loss)

        # Early stopping
        if cfg.early_stop:
            if val_loss + cfg.min_delta < best_val:
                best_val = val_loss
                best_sd = (ema_model or model).state_dict()
                patience_left = cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("⏹️  Early stopping.")
                    break

    # Load best weights (EMA if available)
    if best_sd is not None:
        (ema_model or model).load_state_dict(best_sd)
        if ema_model is not None:
            model.load_state_dict(ema_model.state_dict())

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


def predict_next_day(model, df_last_windows, ticker_to_id, scalers, window, device="cpu", show_progress=True):
    """
    Generate next-day (or multi-horizon) volatility forecasts for each ticker
    using the most recent feature window. Handles feature mismatches between
    training and inference DataFrames, supports multi-horizon models, and
    returns a standardized DataFrame suitable for downstream evaluation.

    Parameters
    ----------
    model : GlobalVolForecaster
        Trained global volatility model.
    df_last_windows : pd.DataFrame
        DataFrame containing the most recent feature window per ticker.
        Must include columns ['ticker', 'date', <features>].
    ticker_to_id : dict
        Mapping from ticker symbol to integer ID.
    scalers : dict[str, StandardScaler]
        Fitted scalers for each ticker (used during training).
    window : int
        Window length used during model training.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['ticker', 'horizon', 'forecast_vol_scaled']
    """
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    model.eval()
    preds = []

    tickers = df_last_windows["ticker"].unique()
    #print(f"Predicting next-day vols for {len(tickers)} tickers...")

    # Disable tqdm if nested
    iterator = tqdm(df_last_windows.groupby("ticker"), desc="Predicting next-day vols", disable=not show_progress)

    for t, g in iterator:
        if t not in ticker_to_id:
            print(f"⚠️ Skipping {t}: not found in training ticker map.")
            continue

        t_id = torch.tensor([ticker_to_id[t]], dtype=torch.long, device=device)
        scaler = scalers[t]

        # --- Align feature sets between training and inference ---
        fitted_feats = list(getattr(scaler, "feature_names_in_", []))
        current_feats = list(g.columns)

        if fitted_feats:
            missing = [f for f in fitted_feats if f not in current_feats]
            extra = [f for f in current_feats if f not in fitted_feats]

            if missing:
                print(f"⚠️ Missing features for {t}: {missing} (filled with zeros)")
                for f in missing:
                    g[f] = 0.0
            if extra:
                g = g.drop(columns=extra, errors="ignore")

            g = g[fitted_feats]
        else:
            print(f"⚠️ Scaler for {t} has no feature_names_in_; using numeric columns only.")
            g = g.select_dtypes(include=np.number)

        # --- Scale and predict ---
        X_scaled = scaler.transform(g.fillna(0.0))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            yhat = model(t_id, X_tensor)

        yhat_np = np.atleast_1d(yhat.cpu().numpy().flatten())

        # --- Expand multi-horizon forecasts ---
        for i, val in enumerate(yhat_np):
            preds.append({
                "ticker": t,
                "horizon": i + 1,  # or use cfg.horizons if available
                "forecast_vol_scaled": float(val)
            })

    # --- Create standardized DataFrame ---
    out = pd.DataFrame(preds)

    # Defensive fallback in case of unexpected output
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=["ticker", "horizon", "forecast_vol_scaled"])

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
    import numpy as np
    import pandas as pd

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

