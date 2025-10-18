"""
Feature engineering utilities for VolSense.
Encapsulates all rolling, cross-sectional, and calendar-based features
used in GlobalVolForecaster training and inference.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# 🔧 Core Feature Computation
# -------------------------------------------------------------------

def compute_base_features(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Computes base return and realized volatility."""
    df = df.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["realized_vol"] = df["return"].rolling(10, min_periods=3).std() * np.sqrt(252)
    df["realized_vol_log"] = np.log(df["realized_vol"] + eps)
    return df


def add_rolling_features(df: pd.DataFrame, group_col="ticker", eps=1e-6) -> pd.DataFrame:
    """Adds rolling volatility and ratio-based features."""
    g = df.groupby(group_col, group_keys=False)

    df["vol_3d"] = g["realized_vol"].apply(lambda s: s.rolling(3,  min_periods=1).mean())
    df["vol_10d"] = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=1).mean())
    df["vol_ratio"] = df["vol_3d"] / (df["vol_10d"] + eps)
    df["vol_chg"] = df["vol_3d"] - df["vol_10d"]
    df["vol_vol"] = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=2).std())
    df["ewma_vol_10d"] = g["realized_vol"].apply(lambda s: s.ewm(span=10, adjust=False).mean())
    return df


def add_market_features(df: pd.DataFrame, group_col="ticker", eps=1e-6) -> pd.DataFrame:
    """Adds cross-sectional market stress and skewness."""
    df["market_stress"] = df.groupby("date")["return"].transform(lambda x: x.std())
    df["market_stress_1d_lag"] = df.groupby(group_col)["market_stress"].shift(1)

    def _skew5(x):
        if len(x) < 3: return np.nan
        m, sd = np.mean(x), np.std(x)
        sd = sd if sd > 0 else eps
        return np.mean(((x - m)/sd)**3)

    df["skew_5d"] = df.groupby(group_col)["return"].apply(lambda s: s.rolling(5, min_periods=3).apply(_skew5, raw=True))
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds day-of-week and month cyclic encodings."""
    df["day_of_week"] = df["date"].dt.dayofweek / 6.0
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds absolute and squared returns."""
    df["abs_return"] = df["return"].abs()
    df["ret_sq"] = df["return"] ** 2
    return df


# -------------------------------------------------------------------
# 🎛️ Unified Builder
# -------------------------------------------------------------------

def build_features(df: pd.DataFrame, include: list = None, exclude: list = None) -> pd.DataFrame:
    """
    Builds all features and allows inclusion/exclusion control.

    Example:
        df = build_features(raw_df, exclude=["skew_5d", "ret_sq"])
    """
    include = include or [
        "vol_3d", "vol_10d", "vol_ratio", "vol_chg", "vol_vol",
        "ewma_vol_10d", "market_stress", "market_stress_1d_lag",
        "skew_5d", "day_of_week", "month_sin", "month_cos", "abs_return"
    ]
    df = compute_base_features(df)
    df = add_rolling_features(df)
    df = add_market_features(df)
    df = add_calendar_features(df)
    df = add_return_features(df)

    # Filter included/excluded features
    feats = [f for f in include if f not in (exclude or [])]
    keep = ["date", "ticker", "return", "realized_vol", "realized_vol_log"] + feats
    return df[keep].dropna().reset_index(drop=True)