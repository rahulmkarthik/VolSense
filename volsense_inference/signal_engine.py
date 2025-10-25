"""
VolSense Signal Engine
----------------------

Converts standardized volatility forecasts into time-series
and cross-sectional signals for quantitative trading,
risk management, and macro regime detection.

This module complements `analytics.py`, which focuses on
snapshot model diagnostics and evaluation metrics.

Usage Example
-------------
>>> from volsense_inference.signal_engine import SignalEngine
>>> engine = SignalEngine(df_forecasts)
>>> df_signals = engine.compute_signals(z_window=60)
>>> top_long = engine.top_signals(horizon=5, direction="long")
"""

import numpy as np
import pandas as pd
from typing import Optional, List


# ============================================================
# 🧠 Signal Engine
# ============================================================
class SignalEngine:
    def __init__(self, df_forecasts: pd.DataFrame):
        """
        Parameters
        ----------
        df_forecasts : pd.DataFrame
            DataFrame containing standardized model outputs:
            ['date','ticker','horizon','forecast_vol','realized_vol'].
        """
        required_cols = {"date", "ticker", "horizon", "forecast_vol"}
        missing = required_cols - set(df_forecasts.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df_forecasts.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.signals: Optional[pd.DataFrame] = None

    # ============================================================
    # 🔹 Core Signal Computation
    # ============================================================
    def compute_signals(
        self,
        z_window: int = 60,
        cross_sectional: bool = True,
        regime_thresholds: tuple = (1.5, -1.0),
    ) -> pd.DataFrame:
        """
        Compute standardized volatility-based trading signals.

        Parameters
        ----------
        z_window : int
            Rolling window (days) for z-score computation.
        cross_sectional : bool
            Whether to compute percentile ranks across tickers per day/horizon.
        regime_thresholds : (float,float)
            (high_z, low_z) thresholds for spike/calm regimes.

        Returns
        -------
        pd.DataFrame
            Original data + columns:
            ['vol_zscore','vol_momentum','vol_spread',
             'regime_flag','xsec_rank','signal_strength']
        """
        df = self.df.sort_values(["ticker", "horizon", "date"]).copy()

        # --- Rolling z-score (time-series)
        df["vol_zscore"] = (
            df.groupby(["ticker", "horizon"])["forecast_vol"]
            .transform(lambda x: (x - x.rolling(z_window, min_periods=10).mean())
                                / (x.rolling(z_window, min_periods=10).std() + 1e-8))
        )

        # --- Momentum: directional change over 5 days
        df["vol_momentum"] = (
            df.groupby(["ticker", "horizon"])["forecast_vol"]
            .diff(5)
            .apply(np.sign)
            .fillna(0)
        )

        # --- Forecast/realized spread
        if "realized_vol" in df.columns:
            df["vol_spread"] = (df["forecast_vol"] / (df["realized_vol"] + 1e-8)) - 1
        else:
            df["vol_spread"] = np.nan

        # --- Regime classification
        high_thr, low_thr = regime_thresholds

        def _regime(v):
            if v > high_thr:
                return "spike"
            elif v < low_thr:
                return "calm"
            else:
                return "normal"

        df["regime_flag"] = df["vol_zscore"].apply(_regime)

        # --- Cross-sectional normalization
        if cross_sectional:
            df["xsec_rank"] = df.groupby(["date", "horizon"])["vol_zscore"].rank(pct=True)
            df["signal_strength"] = (df["xsec_rank"] - 0.5) * 2  # scale [-1,1]
        else:
            df["signal_strength"] = df["vol_zscore"].clip(-3, 3) / 3

        self.signals = df
        return df

    # ============================================================
    # 🔹 Query Utilities
    # ============================================================
    def top_signals(
        self,
        horizon: int = 5,
        n: int = 10,
        direction: str = "long",
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Retrieve strongest volatility signals for a given horizon.

        Parameters
        ----------
        horizon : int
            Forecast horizon to filter (e.g., 1, 5, 10).
        n : int
            Number of top signals to return.
        direction : {"long","short"}
            Long = expecting volatility rise (buy vol)
            Short = expecting volatility fall (sell vol)
        date : Timestamp, optional
            Specific date to filter. Defaults to last date available.
        """
        if self.signals is None:
            self.compute_signals()

        dfh = self.signals[self.signals["horizon"] == horizon].copy()

        # Pick last available day or user-specified date
        if date is None:
            latest_date = dfh["date"].max()
            dfh = dfh[dfh["date"] == latest_date]
        else:
            dfh = dfh[dfh["date"] == pd.to_datetime(date)]

        # Rank by signal direction
        if direction == "long":
            out = dfh.sort_values("signal_strength", ascending=False).head(n)
        else:
            out = dfh.sort_values("signal_strength", ascending=True).head(n)

        return out[
            ["date", "ticker", "horizon", "forecast_vol", "vol_zscore",
             "vol_momentum", "vol_spread", "signal_strength", "regime_flag"]
        ]

    def export(self, path: str):
        """Save computed signals to CSV."""
        if self.signals is None:
            raise RuntimeError("Compute signals first using .compute_signals()")
        self.signals.to_csv(path, index=False)

    # ============================================================
    # 🔹 Visualization Hooks (Optional)
    # ============================================================
    def plot_heatmap(self, date: Optional[pd.Timestamp] = None):
        """
        Plot cross-sectional volatility signal heatmap for a given date.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if self.signals is None:
            raise RuntimeError("Compute signals first.")
        dfh = self.signals.copy()
        if date is None:
            date = dfh["date"].max()
        dfh = dfh[dfh["date"] == pd.to_datetime(date)]

        pivot = dfh.pivot_table(
            index="ticker", columns="horizon", values="signal_strength"
        )
        plt.figure(figsize=(10, max(6, len(pivot) / 10)))
        sns.heatmap(
            pivot, cmap="coolwarm", center=0, annot=False,
            cbar_kws={"label": "Signal Strength"}
        )
        plt.title(f"Volatility Signal Heatmap | {pd.to_datetime(date).date()}")
        plt.xlabel("Horizon (days)")
        plt.ylabel("Ticker")
        plt.tight_layout()
        plt.show()
