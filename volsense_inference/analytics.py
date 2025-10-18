"""
VolSense Analytics — Object-Oriented Quant Signal Toolkit
---------------------------------------------------------
Used internally by Forecast() to compute and visualize volatility signals.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt


class VolAnalytics:
    """
    A modular analytics layer for VolSense forecasts.

    Handles computation of derived signals, rolling stats, and visualization.
    """

    def __init__(
        self,
        preds: pd.DataFrame,
        realized_window: int = 20,
        vol_regime_quantiles: List[float] = [0.2, 0.8],
    ):
        if preds is None or preds.empty:
            raise ValueError("Empty forecast DataFrame provided to VolAnalytics.")
        self.raw = preds.copy()
        self.realized_window = realized_window
        self.vol_regime_quantiles = vol_regime_quantiles
        self.processed = None

    # ===============================================================
    # 🔹 Compute Signals
    # ===============================================================

    def compute(self) -> pd.DataFrame:
        """
        Compute derived volatility signals and tag regimes.
        """
        df = self.raw.copy()
        horizon_cols = [c for c in df.columns if c.startswith("pred_vol_")]
        out_frames = []

        for tkr, g in df.groupby("ticker"):
            g = g.copy()
            g["realized_mean"] = (
                g["realized_vol"]
                .rolling(self.realized_window, min_periods=5)
                .mean()
            )
            g["realized_std"] = (
                g["realized_vol"]
                .rolling(self.realized_window, min_periods=5)
                .std()
            )

            for col in horizon_cols:
                g[f"{col}_f_r_ratio"] = g[col] / (g["realized_vol"] + 1e-8)
                g[f"{col}_zscore"] = (
                    g[col] - g["realized_mean"]
                ) / (g["realized_std"] + 1e-8)
                g[f"{col}_signal_strength"] = g[f"{col}_zscore"].clip(-3, 3)

            out_frames.append(g)

        df = pd.concat(out_frames, ignore_index=True)
        df["vol_regime"] = self._assign_vol_regimes(df)
        self.processed = df
        return df

    # ===============================================================
    # 🔹 Private helper: regime tagging
    # ===============================================================

    def _assign_vol_regimes(self, df: pd.DataFrame) -> pd.Series:
        low_q, high_q = self.vol_regime_quantiles
        low_thr = df["realized_vol"].quantile(low_q)
        high_thr = df["realized_vol"].quantile(high_q)

        def _regime(x):
            if x < low_thr:
                return "Low Vol"
            elif x > high_thr:
                return "High Vol"
            else:
                return "Normal"

        return df["realized_vol"].apply(_regime)

    # ===============================================================
    # 🔹 Trader-facing summaries
    # ===============================================================

    def summary(self, horizon: str = "pred_vol_1") -> pd.DataFrame:
        """
        Return the latest per-ticker forecast summary.
        """
        if self.processed is None:
            self.compute()
        df = self.processed
        cols = [
            "ticker",
            "realized_vol",
            horizon,
            f"{horizon}_f_r_ratio",
            "vol_regime",
            f"{horizon}_signal_strength",
        ]
        cols = [c for c in cols if c in df.columns]
        latest = df.groupby("ticker").tail(1)[cols].reset_index(drop=True)
        latest.rename(
            columns={
                horizon: "forecast_vol",
                f"{horizon}_f_r_ratio": "f/r_ratio",
                f"{horizon}_signal_strength": "signal_strength",
            },
            inplace=True,
        )
        return latest.sort_values("signal_strength", ascending=False)

    def overview(self) -> pd.DataFrame:
        """
        Compute RMSE, bias, and correlation summary.
        """
        if self.processed is None:
            self.compute()

        df = self.processed
        horizons = [c for c in df.columns if c.startswith("pred_vol_")]
        summary = []

        for tkr, g in df.groupby("ticker"):
            entry = {"ticker": tkr}
            for col in horizons:
                valid = g.dropna(subset=["realized_vol", col])
                if len(valid) < 5:
                    continue
                err = valid[col] - valid["realized_vol"]
                entry[f"{col}_rmse"] = np.sqrt(np.mean(err**2))
                entry[f"{col}_bias"] = np.mean(err)
                entry[f"{col}_corr"] = np.corrcoef(
                    valid["realized_vol"], valid[col]
                )[0, 1]
            summary.append(entry)
        return pd.DataFrame(summary).round(4)

    # ===============================================================
    # 🔹 Visualization
    # ===============================================================

    def plot(self, ticker: str):
        """
        Plot realized volatility colored by regime and forecast lines.
        """
        if self.processed is None:
            self.compute()

        df = self.processed
        g = df[df["ticker"] == ticker].copy()
        if g.empty:
            raise ValueError(f"No data for {ticker}")

        plt.figure(figsize=(10, 4))
        plt.plot(g["date"], g["realized_vol"], color="gray", lw=1.5, label="Realized")
        plt.scatter(
            g["date"],
            g["realized_vol"],
            c=g["vol_regime"].map(
                {"Low Vol": "green", "Normal": "orange", "High Vol": "red"}
            ),
            label="Vol Regime",
            s=30,
            alpha=0.8,
        )

        for col in [c for c in g.columns if c.startswith("pred_vol_")]:
            plt.axhline(
                g[col].iloc[-1],
                linestyle="--",
                lw=1.2,
                label=f"{col.replace('pred_vol_', 'Pred ')}",
            )

        plt.title(f"{ticker} — Realized Volatility & Forecast")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
