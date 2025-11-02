import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


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
    batch_size: int = 64
    epochs: int = 20


# --------------------------------------------------------------------
# 🧠 Feature-Wise Attention (Optional)
# --------------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, T, F]
        weights = self.softmax(self.attn(x))
        return x * weights


# --------------------------------------------------------------------
# 🔮 VolNetX Model
# --------------------------------------------------------------------
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

        self.input_size = input_size
        if use_ticker_embedding:
            self.emb = nn.Embedding(n_tickers, emb_dim)
            input_size += emb_dim

        if use_feature_attention:
            self.feature_attn = FeatureAttention(input_size)

        self.encoder = nn.LSTM(input_size, hidden_dim, batch_first=True,
                               num_layers=num_layers, dropout=dropout)

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)

        # Output heads
        out_dim = len(horizons)
        if separate_heads:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in horizons])
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )

    def forward(self, tidx, x):
        if hasattr(self, "emb"):
            emb = self.emb(tidx)
            emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, emb], dim=-1)

        if hasattr(self, "feature_attn"):
            x = self.feature_attn(x)

        x, _ = self.encoder(x)

        if hasattr(self, "transformer"):
            x = self.transformer(x)

        x = x[:, -1]
        if hasattr(self, "norm"):
            x = self.norm(x)

        if self.separate_heads:
            return torch.cat([h(x) for h in self.heads], dim=-1)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
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
            loss = sum(w * loss_fn(preds[:, i], y[:, i]) for i, w in enumerate(config.loss_horizon_weights))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

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
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
