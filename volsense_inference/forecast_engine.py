import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from volsense_inference.model_loader import load_model
from volsense_inference.predictor import predict_batch, attach_realized
from volsense_core.data_fetching.multi_fetch import fetch_multi_ohlcv
from volsense_core.data_fetching.fetch_yf import compute_returns_vol


def build_features(df_all: pd.DataFrame, eps=1e-6):
    """
    Recreate the feature pipeline used in training.
    Works with combined multi-ticker data.
    """
    df_all = df_all.copy()
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df_all.groupby("ticker", group_keys=False)

    # Rolling features
    df_all["vol_3d"]    = g["realized_vol"].apply(lambda s: s.rolling(3,  min_periods=1).mean())
    df_all["vol_10d"]   = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=1).mean())
    df_all["vol_ratio"] = df_all["vol_3d"] / (df_all["vol_10d"] + eps)
    df_all["vol_chg"]   = df_all["vol_3d"] - df_all["vol_10d"]
    df_all["vol_vol"]   = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=2).std())

    # Market-level and calendar features
    df_all["market_stress"] = df_all.groupby("date")["return"].transform(lambda x: x.std())
    df_all["market_stress_1d_lag"] = g["market_stress"].shift(1)

    def _skew5(x):
        if len(x) < 3:
            return np.nan
        m, sd = np.mean(x), np.std(x)
        sd = sd if sd > 0 else eps
        return np.mean(((x - m) / sd) ** 3)

    df_all["skew_5d"] = g["return"].apply(lambda s: s.rolling(5, min_periods=3).apply(_skew5, raw=True))
    df_all["day_of_week"] = df_all["date"].dt.dayofweek / 6.0
    df_all["month_sin"] = np.sin(2 * np.pi * df_all["date"].dt.month / 12)
    df_all["month_cos"] = np.cos(2 * np.pi * df_all["date"].dt.month / 12)
    df_all["abs_return"] = df_all["return"].abs()
    df_all["ret_sq"] = df_all["return"] ** 2

    return df_all



class Forecast:
    """
    High-level runtime interface for volatility forecasting and visualization.
    """

    def __init__(
        self,
        model_version: str = "v3",
        checkpoints_dir: str = "models",
        start: str = "2010-01-01",
    ):
        print(f"🚀 Initializing VolSense.Forecast (model={model_version})")
        self.model_version = model_version
        self.checkpoints_dir = checkpoints_dir
        self.start = start

        # Load model and assets
        self.model, self.meta, self.scalers, self.ticker_to_id, self.features = load_model(
            model_version=model_version, checkpoints_dir=checkpoints_dir
        )

        self.window = self.meta.get("window", 30)
        self.horizons = self.meta.get("horizons", [1])
        print(f"✔ Window={self.window}, Horizons={self.horizons}")

        self.predictions = None

    # ------------------------------------------------------------------
    # Data & Forecasting
    # ------------------------------------------------------------------
    def _prepare_data(self, tickers):
        data_dict = fetch_multi_ohlcv(tickers, start=self.start)
        frames = []
        for tkr, df in data_dict.items():
            feat = compute_returns_vol(df, window=self.window, ticker=tkr)
            feat["ticker"] = tkr
            frames.append(feat)

        df_recent = pd.concat(frames, ignore_index=True)
        df_recent = build_features(df_recent)   # <-- New step
        self.df_recent = df_recent

        return df_recent

    def run(self, tickers):
        tickers = [tickers] if isinstance(tickers, str) else tickers
        print(f"\n🌍 Running forecasts for {len(tickers)} tickers...\n")

        df_recent = self._prepare_data(tickers)
        preds = predict_batch(
            self.model, self.meta, df_recent, tickers,
            scalers=self.scalers, ticker_to_id=self.ticker_to_id, features=self.features
        )
        preds = attach_realized(preds, df_recent)
        self.predictions = preds
        print("✅ Forecast complete.")
        return preds

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(self, ticker: str, show_vix: bool = False, vix_df: pd.DataFrame = None):
        if self.predictions is None:
            raise RuntimeError("No forecasts computed yet. Run .run(ticker) first.")

        preds = self.predictions[self.predictions["ticker"] == ticker]
        if preds.empty:
            raise ValueError(f"No forecast results for {ticker}")

        df_t = self.df_recent[self.df_recent["ticker"] == ticker].copy()
        df_t = df_t.sort_values("date").tail(180)  # show recent ~6 months

        plt.figure(figsize=(10, 5))
        plt.plot(df_t["date"], df_t["realized_vol"], label="Realized Volatility", color="tab:blue")
        plt.axhline(preds["pred_vol_1"].values[0], color="tab:orange", linestyle="--", label="Predicted 1d Vol")
        if len(self.horizons) > 1:
            for i, h in enumerate(self.horizons[1:], start=2):
                col = f"pred_vol_{h}"
                if col in preds.columns:
                    plt.axhline(preds[col].values[0], linestyle="--", alpha=0.7, label=f"Pred {h}d")

        if show_vix and vix_df is not None:
            vix_df = vix_df.copy()
            vix_df["date"] = pd.to_datetime(vix_df["date"])
            plt.plot(vix_df["date"], vix_df["close"], color="tab:red", alpha=0.5, label="VIX Index")

        plt.title(f"{ticker} — Forecast vs Realized Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()