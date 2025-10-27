import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from volsense_inference.model_loader import load_model
from volsense_inference.predictor import predict_batch, attach_realized
from volsense_core.data_fetching.multi_fetch import fetch_multi_ohlcv
from volsense_core.data_fetching.fetch_yf import compute_returns_vol
from volsense_inference.analytics import Analytics



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
    df_all["ewma_vol_10d"] = g["realized_vol"].apply(lambda s: s.ewm(span=10, adjust=False).mean())

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
        model_version: str = "v109",
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

        self.window = self.meta.get("window", 40)
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

        # Attach analytics object
        self.signals = Analytics(preds)
        self.signals.compute()

        return preds

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------


    def plot(self, ticker: str, show_vix: bool = False, vix_df: pd.DataFrame = None, show: bool = True):
        """Plots realized vs predicted volatility for a given ticker.

        Args:
            ticker: Ticker symbol to plot.
            show_vix: Overlay VIX if True and vix_df provided.
            vix_df: DataFrame with columns ['date','close'] for VIX.
            show: If True, call plt.show() inside and close the figure (return None).
                  If False, return the Figure and do not show.
        """
        if self.predictions is None:
            raise RuntimeError("No forecasts computed yet. Run .run(ticker) first.")

        preds = self.predictions[self.predictions["ticker"] == ticker]
        if preds.empty:
            raise ValueError(f"No forecast results for {ticker}")

        df_t = self.df_recent[self.df_recent["ticker"] == ticker].copy()
        df_t = df_t.sort_values("date").tail(180)  # last ~6 months

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_t["date"], df_t["realized_vol"], label="Realized Volatility", color="tab:blue")

        # Collect horizons we actually have predictions for
        present_horizons = [h for h in sorted(set(getattr(self, "horizons", [])))
                            if f"pred_vol_{h}" in preds.columns]

        # Unique colors per horizon (skip index 0 in tab10 to avoid blue clash)
        palette = plt.get_cmap("tab10").colors
        start_idx = 1  # 0 is blue; we already used that for realized vol
        color_map = {h: palette[(start_idx + i) % len(palette)] for i, h in enumerate(present_horizons)}

        # Plot all horizons with distinct colors
        for h in present_horizons:
            col = f"pred_vol_{h}"
            y = preds[col].values[0]
            ax.axhline(y, color=color_map[h], linestyle="--", alpha=0.9, label=f"Pred {h}d")

        if show_vix and vix_df is not None:
            vix_df = vix_df.copy()
            vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")
            ax.plot(vix_df["date"], vix_df["close"], color="tab:red", alpha=0.5, label="VIX Index")

        ax.set_title(f"{ticker} — Forecast vs Realized Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")

        # Deduplicate legend entries (defensive)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if show:
            plt.show()
            plt.close(fig)  # avoid a second capture by caller
            return None

        return fig

