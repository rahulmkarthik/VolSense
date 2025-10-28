"""
VolSense Signal Engine (Snapshot Edition)
-----------------------------------------
Transforms snapshot forecasts from ForecastEngine or ForecasterAPI
into actionable volatility signals and sector-level intelligence.

Key changes vs. prior version:
  • Removed metrics needing time-series history (EMA, momentum, spillover)
  • Treats realized_vol as today's realized volatility
  • Renamed realized_vol → today_vol for interpretability
  • Cleaned & vectorized computations for speed
  • Simplified sector-relative rollups and cross-sectional scoring
  • Added cross-sectional sector z-score computation for single-day analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.colors as mcolors

from volsense_inference.sector_mapping import get_sector_map, get_color


# ============================================================
# 🔹 Utility: Coerce model outputs into standard format
# ============================================================
def _coerce_to_long(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wide or standardized forecast DataFrame to canonical format:
        ['date', 'ticker', 'horizon', 'forecast_vol', 'today_vol']
    """
    df = df_in.copy()
    df.columns = [c.lower() for c in df.columns]

    # Case 1: Already in long standardized format
    if {"ticker", "horizon", "forecast_vol"}.issubset(df.columns):
        if "today_vol" not in df.columns and "realized_vol" in df.columns:
            df["today_vol"] = df["realized_vol"]
        elif "today_vol" not in df.columns:
            df["today_vol"] = np.nan
        df["date"] = pd.to_datetime(df.get("date", pd.Timestamp.today().normalize()))
        return df[["date", "ticker", "horizon", "forecast_vol", "today_vol"]]

    # Case 2: ForecastEngine wide format (pred_vol_1, pred_vol_5, ...)
    pred_cols = [c for c in df.columns if c.startswith("pred_vol_")]
    if "ticker" in df.columns and pred_cols:
        df["date"] = pd.Timestamp.today().normalize()
        long = df.melt(
            id_vars=["ticker", "date"],
            value_vars=pred_cols,
            var_name="horizon_col",
            value_name="forecast_vol",
        )
        long["horizon"] = long["horizon_col"].str.extract(r"(\d+)").astype(int)
        long["today_vol"] = df["realized_vol"].reindex(long.index, fill_value=np.nan)
        return long[["date", "ticker", "horizon", "forecast_vol", "today_vol"]]

    raise ValueError("Input DataFrame format not recognized for SignalEngine.")


# ============================================================
# ⚙️ SignalEngine: Cross-Sectional and Sector Intelligence
# ============================================================
class SignalEngine:
    """
    Converts forecast snapshots into standardized volatility signals.
    Works on a single as-of date across multiple tickers and horizons.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, model_version: str = "v109"):
        self.df: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.model_version = model_version
        self.sector_map = get_sector_map(model_version)

        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame):
        """Standardize input."""
        df_long = _coerce_to_long(data)
        df_long["ticker"] = df_long["ticker"].astype(str)
        df_long["date"] = pd.to_datetime(df_long["date"])
        self.df = df_long
    def ticker_summary(
        self,
        ticker: str,
        horizons: list[int] | None = None,
        date: pd.Timestamp | None = None,
        decimals: int = 4,
    ) -> str:
        """
        Returns a human-readable summary string for a single ticker across selected horizons.

        Example:
            print(engine.ticker_summary("AAPL", horizons=[1,5,10]))
        """
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")

        df = self.signals.copy()
        if date is None:
            date = df["date"].max()
        dsub = df[(df["date"] == pd.to_datetime(date)) & (df["ticker"] == str(ticker))].copy()

        if dsub.empty:
            return f"{ticker} — no data for {pd.to_datetime(date).date()}"

        if horizons is None:
            horizons = sorted(dsub["horizon"].dropna().unique().tolist())

        lines = [f"{ticker} — {pd.to_datetime(date).date()}"]
        # Use the first row for static attributes (sector)
        sector = dsub["sector"].iloc[0] if "sector" in dsub.columns else "Unknown"

        for h in horizons:
            row = dsub[dsub["horizon"] == h]
            if row.empty:
                lines.append(f"  {h}d: no data")
                continue
            r = row.iloc[0]

            fvol = float(r.get("forecast_vol", np.nan))
            tvol = float(r.get("today_vol", np.nan))
            spread = float(r.get("vol_spread", np.nan)) * 100 if pd.notna(r.get("vol_spread", np.nan)) else np.nan
            z = float(r.get("vol_zscore", np.nan))
            pos = str(r.get("position", "")) if pd.notna(r.get("position", np.nan)) else ""
            reg = str(r.get("regime_flag", "")) if pd.notna(r.get("regime_flag", np.nan)) else ""
            sect_z = float(r.get("sector_z", np.nan)) if "sector_z" in r else np.nan
            rank_u = float(r.get("rank_universe", np.nan)) if "rank_universe" in r else np.nan

            fvol_str = f"{fvol:.{decimals}f}" if np.isfinite(fvol) else "NaN"
            tvol_str = f"{tvol:.{decimals}f}" if np.isfinite(tvol) else "NaN"
            spr_str = f"{spread:+.1f}%" if np.isfinite(spread) else "NaN"
            z_str = f"{z:+.2f}" if np.isfinite(z) else "NaN"
            sectz_str = f"{sect_z:+.2f}" if np.isfinite(sect_z) else "NaN"
            ranku_str = f"{rank_u:.0%}" if np.isfinite(rank_u) else "NaN"

            lines.append(
                f"  {h}d: fc={fvol_str}, today={tvol_str if tvol_str != 'NaN' else 'N/A'} ({spr_str}), z={z_str}, "
                f"pos={pos or '-'}, regime={reg or '-'}, sector={sector} (z={sectz_str}, uni-rank={ranku_str})"
            )

        return "\n".join(lines)

    # ============================================================
    # 🔹 Core Signal Computation
    # ============================================================
    def compute_signals(self, enrich_with_sectors: bool = True) -> pd.DataFrame:
        """
        Computes cross-sectional volatility signals and optional sector enrichment.
        """
        if self.df is None:
            raise RuntimeError("No data loaded into SignalEngine.")

        df = self.df.copy().sort_values(["ticker", "horizon"])
        print(f"⚙️ Computing cross-sectional signals for {df['ticker'].nunique()} tickers...")

        # --- 1. Z-score (cross-sectional within each horizon)
        df["vol_zscore"] = df.groupby("horizon")["forecast_vol"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

        # --- 2. Spread vs today's realized vol (interpretable)
        if "today_vol" in df.columns and df["today_vol"].notna().any():
            df["vol_spread"] = (df["forecast_vol"] / (df["today_vol"] + 1e-8)) - 1
        else:
            df["vol_spread"] = np.nan

        # --- 3. Cross-sectional ranks (strength & direction)
        df["xsec_rank"] = df.groupby("horizon")["vol_zscore"].rank(pct=True)
        df["signal_strength"] = (df["xsec_rank"] - 0.5) * 2

        # --- 4. Regime flags based on z-score thresholds
        df["regime_flag"] = pd.cut(
            df["vol_zscore"],
            bins=[-np.inf, -1.0, 1.0, np.inf],
            labels=["calm", "normal", "spike"],
        )

        # --- 5. Sector enrichments
        if enrich_with_sectors:
            df = self._attach_sector(df)
            df = self._sector_rollups(df)

        # --- 6. Position Classification
        df["position"] = pd.cut(
            df["vol_zscore"],
            bins=[-np.inf, -1.0, 1.0, np.inf],
            labels=["long", "neutral", "short"],
        )

        self.signals = df
        return df

    # ============================================================
    # 🔹 Sector Utilities
    # ============================================================
    def _attach_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach sector tags using model-specific sector map."""
        df["sector"] = df["ticker"].map(self.sector_map).fillna("Unknown")
        return df

    def _sector_rollups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sector-level mean, std, and z-score (cross-sectional snapshot)."""
        grp = df.groupby(["horizon", "sector"])["forecast_vol"]
        agg = grp.agg(["mean", "std", "median"]).reset_index().rename(
            columns={
                "mean": "sector_mean",
                "std": "sector_std",
                "median": "sector_median",
            }
        )
        df = df.merge(agg, on=["horizon", "sector"], how="left")

        # --- Cross-sectional sector z-score (snapshot mode) ---
        # Measures how each sector's average forecast volatility compares
        # to all other sectors *on the same day*, not over time.
        # --- Cross-sectional sector z-score per horizon (snapshot mode) ---
        # Measures how each sector's average forecast volatility compares
        # to all other sectors *within the same horizon* on this day.
        sector_summary = (
            df.groupby(["horizon", "sector"])["sector_mean"]
            .mean()
            .reset_index()
        )
        sector_summary["sector_z"] = sector_summary.groupby("horizon")["sector_mean"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
        df = df.merge(sector_summary[["horizon", "sector", "sector_z"]], on=["horizon", "sector"], how="left")


        # Within-sector & universe ranks
        df["rank_sector"] = df.groupby(["horizon", "sector"])["forecast_vol"].rank(pct=True)
        df["rank_universe"] = df.groupby(["horizon"])["forecast_vol"].rank(pct=True)
        return df

    # ============================================================
    # 🔹 Visualization
    # ============================================================
    def plot_sector_heatmap(self, date: Optional[pd.Timestamp] = None):
        """Visualizes sector z-scores as a heatmap by horizon."""
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")
        df = self.signals.copy()
        if date is None:
            date = df["date"].max()
        dsub = df[df["date"] == pd.to_datetime(date)]

        pivot = (
            dsub.pivot_table(
                index="sector", columns="horizon", values="sector_z", aggfunc="mean"
            )
            .fillna(0)
            .sort_index()
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Sector Z-score"},
        )
        plt.title(f"Sector Volatility Heatmap ({pd.to_datetime(date).date()})")
        plt.tight_layout()
        plt.show()

    def plot_top_sectors(
        self, horizon: int = 5, date: Optional[pd.Timestamp] = None, top_n: int = 8
    ):
        """Displays top sectors by mean sector z-score."""
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")
        df = self.signals.copy()
        if date is None:
            date = df["date"].max()
        dsub = df[df["date"] == pd.to_datetime(date)]
        sector_stats = dsub.groupby("sector")["sector_z"].mean().sort_values(ascending=False)
        top = sector_stats.head(top_n)
        colors = [get_color(s) for s in top.index]
        plt.figure(figsize=(8, 4))
        plt.barh(top.index, top.values, color=colors)
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_n} Sectors by Mean Z-score ({horizon}d)")
        plt.xlabel("Mean Sector Z-score")
        plt.tight_layout()
        plt.show()

    def sector_summary(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Tabular summary of per-sector volatility conditions."""
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")
        df = self.signals.copy()
        if date is None:
            date = df["date"].max()
        dsub = df[df["date"] == pd.to_datetime(date)]
        summary = (
            dsub.groupby("sector")
            .agg(
                sector_mean=("sector_mean", "mean"),
                sector_std=("sector_std", "mean"),
                sector_z=("sector_z", "mean"),
            )
            .reset_index()
        )
        summary["sector_color"] = summary["sector"].map(get_color)
        summary = summary.sort_values("sector_z", ascending=False)
        print(f"📊 Sector Summary for {pd.to_datetime(date).date()}")
        display(summary.style.background_gradient(subset=["sector_z"], cmap="coolwarm"))
        return summary

    def plot_ticker_heatmap(self, horizon: int = 5, sector: str | None = None):
        """Displays a cross-sectional heatmap of tickers' z-scores and positions for a given horizon."""
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")

        df = self.signals.copy()
        df = df[df["horizon"] == horizon]
        if sector:
            df = df[df["sector"] == sector]

        pivot = df.pivot_table(index="ticker", values="vol_zscore", aggfunc="mean").sort_values("vol_zscore")

        plt.figure(figsize=(8, max(6, len(pivot) * 0.25)))
        sns.heatmap(
            pivot,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Volatility Z-score"},
        )

        title = f"{'Sector: ' + sector if sector else 'All Sectors'} — Ticker Z-scores ({horizon}d)"
        plt.title(title)
        plt.xlabel("")
        plt.ylabel("Ticker")
        plt.tight_layout()
        plt.show()

    def plot_position_counts(self, horizon: int = 5):
        """Plots count of long/neutral/short signals across all tickers."""
        if self.signals is None:
            raise RuntimeError("Run .compute_signals() first.")

        df = self.signals[self.signals["horizon"] == horizon]
        counts = df["position"].value_counts().reindex(["long", "neutral", "short"]).fillna(0)

        plt.figure(figsize=(6, 3))
        sns.barplot(x=counts.index, y=counts.values, palette=["green", "gray", "red"])
        plt.title(f"Signal Position Counts ({horizon}d)")
        plt.ylabel("Number of Tickers")
        plt.tight_layout()
        plt.show()
