"""
VolSense Signal Engine
----------------------

Converts model forecasts into time-series and cross-sectional signals.

Now supports initializing directly from the `preds` DataFrame returned by
Forecast.run (wide snapshot with columns like ['ticker','realized_vol','pred_vol_1',...]),
and will auto-convert to the internal long format:

['date','ticker','horizon','forecast_vol','realized_vol'].
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _coerce_to_long(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - Wide snapshot (one row per ticker): ['ticker','realized_vol','pred_vol_1',...,'date'?]
      - Already long format: ['date','ticker','horizon','forecast_vol','realized_vol']

    Returns long-format DataFrame with:
      ['date','ticker','horizon','forecast_vol','realized_vol']
    """
    cols = set(df_in.columns)

    # Already long-format
    if {"ticker", "horizon", "forecast_vol"}.issubset(cols):
        out = df_in.copy()
        if "date" not in out.columns:
            # If no date, assume today for all rows
            out["date"] = pd.Timestamp.today().normalize()
        if "realized_vol" not in out.columns:
            out["realized_vol"] = np.nan
        out["date"] = pd.to_datetime(out["date"])
        out = out[["date", "ticker", "horizon", "forecast_vol", "realized_vol"]]
        return out.sort_values(["ticker", "horizon", "date"]).reset_index(drop=True)

    # Wide snapshot from Forecast.run (pred_vol_* columns)
    pred_cols = [c for c in df_in.columns if c.startswith("pred_vol_")]
    if "ticker" in cols and pred_cols:
        df_wide = df_in.copy()
        # Date handling
        if "date" in df_wide.columns:
            df_wide["date"] = pd.to_datetime(df_wide["date"])
        else:
            df_wide["date"] = pd.Timestamp.today().normalize()

        # Melt predictions to long format
        long = df_wide.melt(
            id_vars=[c for c in ["ticker", "realized_vol", "date"] if c in df_wide.columns],
            value_vars=pred_cols,
            var_name="pred_col",
            value_name="forecast_vol",
        )
        # Extract numeric horizon from "pred_vol_{h}"
        long["horizon"] = (
            long["pred_col"].str.replace("pred_vol_", "", regex=False).astype(int)
        )
        if "realized_vol" not in long.columns:
            long["realized_vol"] = np.nan

        out = long.rename(columns={"date": "date"})  # explicit for clarity
        out = out[["date", "ticker", "horizon", "forecast_vol", "realized_vol"]]
        return out.sort_values(["ticker", "horizon", "date"]).reset_index(drop=True)

    raise ValueError(
        "Unsupported input format. Provide either a wide snapshot with columns "
        "['ticker','pred_vol_1', ...] (optionally 'realized_vol','date') or a long "
        "format with ['date','ticker','horizon','forecast_vol' (+'realized_vol')]."
    )


# ============================================================
# 🧠 Signal Engine
# ============================================================
class SignalEngine:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame, optional
            - Directly pass preds from Forecast.run (wide snapshot), or
            - A long-format DataFrame with ['date','ticker','horizon','forecast_vol','realized_vol'].
        """
        self.df: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame):
        """Set/replace the engine input data (accepts wide preds or long format)."""
        df_long = _coerce_to_long(data)
        # Normalize types
        df_long["date"] = pd.to_datetime(df_long["date"])
        df_long["ticker"] = df_long["ticker"].astype(str)
        df_long["horizon"] = df_long["horizon"].astype(int)
        self.df = df_long

    @classmethod
    def from_preds(cls, preds: pd.DataFrame) -> "SignalEngine":
        """Convenience: build from the Forecast.run output (wide snapshot)."""
        return cls(preds)

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

        Handles two cases:
        - Time series (multiple dates per ticker/horizon): rolling z-score over time.
        - Snapshot (single date): cross-sectional z-score across tickers per horizon.

        Parameters
        ----------
        z_window : int
            Rolling window (days) for z-score computation (time-series case).
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
        if self.df is None:
            raise RuntimeError("No data set. Initialize with preds or call set_data(df).")

        df = self.df.sort_values(["ticker", "horizon", "date"]).copy()

        # Detect if we have time series (more than one date per ticker/horizon)
        ts_counts = df.groupby(["ticker", "horizon"])["date"].nunique()
        has_timeseries = (ts_counts > 1).any()

        # --- Z-score
        if has_timeseries:
            df["vol_zscore"] = (
                df.groupby(["ticker", "horizon"])["forecast_vol"]
                  .transform(lambda x: (x - x.rolling(z_window, min_periods=10).mean())
                                    / (x.rolling(z_window, min_periods=10).std() + 1e-8))
            )
            # Momentum over time
            df["vol_momentum"] = (
                df.groupby(["ticker", "horizon"])["forecast_vol"]
                  .diff(5).apply(np.sign).fillna(0)
            )
        else:
            # Snapshot: cross-sectional z-score per horizon/day
            def _xsec_z(g):
                mu = g["forecast_vol"].mean()
                sd = g["forecast_vol"].std(ddof=0)
                sd = sd if sd > 0 else 1e-8
                return (g["forecast_vol"] - mu) / sd

            df["vol_zscore"] = (
                df.groupby(["date", "horizon"], group_keys=False).apply(_xsec_z)
            )
            df["vol_momentum"] = 0.0

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

        # --- Cross-sectional normalization and signal strength
        if cross_sectional:
            df["xsec_rank"] = df.groupby(["date", "horizon"])["vol_zscore"].rank(pct=True)
            df["signal_strength"] = (df["xsec_rank"] - 0.5) * 2  # scale [-1,1]
        else:
            df["signal_strength"] = df["vol_zscore"].clip(-3, 3) / 3

        self.signals = df
        return df

        # End compute_signals

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

        if self.signals is None:
            raise RuntimeError("Compute signals first.")
        dfh = self.signals.copy()
        if date is None:
            date = dfh["date"].max()
        dfh = dfh[dfh["date"] == pd.to_datetime(date)]

        pivot = dfh.pivot_table(index="ticker", columns="horizon", values="signal_strength")
        plt.figure(figsize=(10, max(6, len(pivot) / 10)))
        sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False, cbar_kws={"label": "Signal Strength"})
        plt.title(f"Volatility Signal Heatmap | {pd.to_datetime(date).date()}")
        plt.xlabel("Horizon (days)")
        plt.ylabel("Ticker")
        plt.tight_layout()
        plt.show()