import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from volsense_core.utils.scalers import TorchStandardScaler


# --------------------------------------------------------------------
# 🔧 Config
# --------------------------------------------------------------------
@dataclass
class VolNetXConfig:
    window: int = 65
    horizons: List[int] = (1, 5, 10)
    input_size: int = 16
    hidden_dim: int = 160
    emb_dim: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    use_transformer: bool = True
    use_ticker_embedding: bool = True
    use_feature_attention: bool = True
    separate_heads: bool = True
    use_layernorm: bool = True
    max_ticker_count: int = 512
    loss_horizon_weights: List[float] = (0.55, 0.25, 0.2)
    device: str = "cpu"
    early_stop: bool = True
    patience: int = 5
    lr: float = 1e-3
    cosine_schedule: bool = False
    batch_size: int = 64
    epochs: int = 20
    val_mode: str = "causal" # 'causal' or 'holdout_slice'
    val_start: str = None
    val_end: str = None


# --------------------------------------------------------------------
# 🧠 Feature-Wise Attention (Fixed)
# --------------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, F = x.shape
        x_flat = x.view(-1, F)               # [B*T, F]
        attn_flat = self.attn(x_flat)        # [B*T, F]
        attn = attn_flat.view(B, T, F)       # [B, T, F]
        weights = self.softmax(attn)
        return x * weights


class VolNetX(nn.Module):
    def __init__(self,
                 n_tickers: int,
                 window: int,
                 input_size: int,
                 horizons: List[int],
                 hidden_dim: int = 160,
                 emb_dim: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_transformer: bool = True,
                 use_ticker_embedding: bool = True,
                 use_feature_attention: bool = True,
                 separate_heads: bool = True,
                 use_layernorm: bool = True):
        super().__init__()
        self.window = window
        self.horizons = horizons
        self.separate_heads = separate_heads
        self.use_transformer = use_transformer
        self.use_ticker_embedding = use_ticker_embedding
        self.use_feature_attention = use_feature_attention

        self.emb = nn.Embedding(n_tickers, emb_dim) if use_ticker_embedding else None
        final_input_dim = input_size + (emb_dim if use_ticker_embedding else 0)

        if use_feature_attention:
            self.feature_attn = FeatureAttention(final_input_dim)


        self.encoder = nn.LSTM(
            input_size=final_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)

        out_dim = len(horizons)
        if separate_heads:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in horizons
            ])
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )

    def forward(self, tidx, x):
        assert x.ndim == 3 and x.shape[1] == self.window, f"Expected shape [B, T={self.window}, F], got {x.shape}"


        if self.use_ticker_embedding and self.emb is not None:
            emb = self.emb(tidx)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb], dim=-1)

        if self.use_feature_attention:
            x = self.feature_attn(x)
        
        x, _ = self.encoder(x)

        x = x[:, -1]  # Take the last timestep only — [B, H]

        if hasattr(self, "norm"):
            x = self.norm(x)

        if self.separate_heads:
            return torch.cat([h(x) for h in self.heads], dim=-1)  # [B, H]
        else:
            return self.head(x)



# --------------------------------------------------------------------
# 🏋️ Training Utility
# --------------------------------------------------------------------
def train_volnetx(config: VolNetXConfig, train_loader, val_loader, n_tickers: int):

    model = VolNetX(
        n_tickers=n_tickers,
        window=config.window,
        input_size=config.input_size,
        horizons=config.horizons,
        hidden_dim=config.hidden_dim,
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_transformer=config.use_transformer,
        use_ticker_embedding=config.use_ticker_embedding,
        use_feature_attention=config.use_feature_attention,
        separate_heads=config.separate_heads,
        use_layernorm=config.use_layernorm
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=5e-5)

    # -----------------------------------------------------------
    # 🚀 NEW: Cosine Annealing Scheduler
    # -----------------------------------------------------------
    scheduler = None
    if getattr(config, "cosine_schedule", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs, 
            eta_min=1e-6
        )
    # -----------------------------------------------------------

    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for x, tidx, y in train_loader:
            x, tidx, y = x.to(config.device), tidx.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            preds = model(tidx, x)
            # 🚀 UPGRADE 2: Weighted Loss (Focus on High Vol)
            # We weight the loss by the target magnitude to punish errors on spikes
            loss = 0
            for i, w in enumerate(config.loss_horizon_weights):
                mse = (preds[:, i] - y[:, i]) ** 2
                # Simple weighting: if log_vol > -1.0 (approx 36% vol), double the penalty
                weight = 1.0 + (y[:, i] > -1.0).float() * 1.5 
                loss += w * (weight * mse).mean()
            loss.backward()
            # 🚀 UPGRADE 3: Gradient Clipping (The Stability Fix)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())

        # 🚀 STEP SCHEDULER (After optimization steps)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.lr

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, tidx, y in val_loader:
                x, tidx, y = x.to(config.device), tidx.to(config.device), y.to(config.device)
                preds = model(tidx, x)
                loss = sum(w * loss_fn(preds[:, i], y[:, i]) for i, w in enumerate(config.loss_horizon_weights))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{config.epochs} | LR: {current_lr:.2e} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if config.early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping triggered.")
                    break

    return model

        
# --------------------------------------------------------------------
# 🛠️ Dataset Preparation (With Internal Scaling)
# --------------------------------------------------------------------
def build_volnetx_dataset(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "realized_vol_log",
    config: VolNetXConfig = None,
    scaler: Optional[TorchStandardScaler] = None, # <--- NEW ARG
) -> Tuple[Dict[str, int], DataLoader, DataLoader, TensorDataset, TensorDataset, TorchStandardScaler]: # <--- UPDATED RETURN
    
    # Setup defaults
    window = config.window if config else 65
    horizons = config.horizons if config else [1, 5, 10]
    batch_size = config.batch_size if config else 64
    
    df = df.sort_values(["ticker", "date"]).copy()
    tickers = df["ticker"].unique().tolist()
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["tidx"] = df["ticker"].map(ticker_to_id)

    # --- Validation Split Logic ---
    val_mask = pd.Series(False, index=df.index)
    if config and config.val_mode == "causal":
        if config.val_start:
            val_mask = df["date"] >= pd.to_datetime(config.val_start)
        else:
            dates = df["date"].sort_values().unique()
            split_idx = int(len(dates) * 0.8)
            val_mask = df["date"] >= dates[split_idx]
    elif config and config.val_mode == "holdout_slice":
        if config.val_start and config.val_end:
            start_dt = pd.to_datetime(config.val_start)
            end_dt = pd.to_datetime(config.val_end)
            val_mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)

    df["split"] = val_mask.astype(int)

    # --- 🚀 NEW: Internal Scaling Logic ---
    if scaler is None:
        print("   ⚖️ Fitting new global scaler (Train split only)...")
        scaler = TorchStandardScaler()
        # Fit only on Training data (split == 0) to prevent leakage
        train_subset = df[df["split"] == 0][features]
        scaler.fit(train_subset)
    
    # Apply transform to the entire dataframe in-place
    # (TorchStandardScaler returns numpy array, safe to assign back)
    df[features] = scaler.transform(df[features])
    # ---------------------------------------

    train_X, train_Y, train_T = [], [], []
    val_X, val_Y, val_T = [], [], []

    for t in tickers:
        g = df[df["ticker"] == t].reset_index(drop=True)
        if len(g) < window + max(horizons):
            continue

        X_arr = g[features].values.astype(np.float32)
        y_arr = g[target_col].values.astype(np.float32)
        split_arr = g["split"].values
        
        max_h = max(horizons)
        valid_indices = np.arange(window, len(g) - max_h)
        if len(valid_indices) == 0: continue

        X_windows = np.lib.stride_tricks.sliding_window_view(X_arr, window, axis=0)
        X_windows = X_windows[valid_indices - window] 
        
        y_targets = np.stack([y_arr[valid_indices + h - 1] for h in horizons], axis=1)
        t_ids = np.full(len(valid_indices), ticker_to_id[t], dtype=np.int64)
        
        target_indices = valid_indices + horizons[0] - 1
        is_val = split_arr[target_indices] == 1
        
        if np.sum(~is_val) > 0:
            train_X.append(X_windows[~is_val])
            train_Y.append(y_targets[~is_val])
            train_T.append(t_ids[~is_val])
        if np.sum(is_val) > 0:
            val_X.append(X_windows[is_val])
            val_Y.append(y_targets[is_val])
            val_T.append(t_ids[is_val])

    def to_tensor_dataset(X_list, Y_list, T_list):
        if not X_list: return None
        X = torch.tensor(np.concatenate(X_list))
        # Transpose [B, F, T] -> [B, T, F]
        if X.ndim == 3 and X.shape[2] == window:
             X = X.permute(0, 2, 1)
        Y = torch.tensor(np.concatenate(Y_list))
        T = torch.tensor(np.concatenate(T_list), dtype=torch.long)
        return TensorDataset(X, T, Y)

    train_ds = to_tensor_dataset(train_X, train_Y, train_T)
    val_ds = to_tensor_dataset(val_X, val_Y, val_T)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    # Return scaler as the last element
    return ticker_to_id, train_loader, val_loader, train_ds, val_ds, scaler


# --------------------------------------------------------------------
# 📊 Vectorized Evaluation
# --------------------------------------------------------------------
def evaluate_volnetx(model, loader, config: VolNetXConfig):
    model.eval()
    all_x, all_tidx, all_y = [], [], []

    for x, tidx, y in loader:
        all_x.append(x)
        all_tidx.append(tidx)
        all_y.append(y)

    x_full = torch.cat(all_x).to(config.device)
    tidx_full = torch.cat(all_tidx).to(config.device)
    y_full = torch.cat(all_y)

    with torch.no_grad():
        preds_full = model(tidx_full, x_full).cpu()

    return preds_full, y_full
