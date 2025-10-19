import numpy as np
import pandas as pd

EPS = 1e-6

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """If columns are MultiIndex (common from yfinance), flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns.name = None
    return df

def compute_base_features(df: pd.DataFrame, eps: float = EPS, vol_lookback: int = 10) -> pd.DataFrame:
    """
    Ensure the core columns exist:
      - date (datetime), ticker (if missing, fill with 'UNK')
      - return (pct change)  -> computed from Adj Close or Close if missing
      - realized_vol (rolling std of return) -> computed if missing
      - realized_vol_log = log(realized_vol + eps)

    Accepts either:
      (A) raw OHLCV with 'Adj Close' or 'Close'
      (B) already-processed long table with 'return' and maybe 'realized_vol'
    """
    df = _flatten_cols(df.copy())

    # date column
    if "date" not in df.columns:
        # yfinance frames typically have DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        else:
            raise ValueError("Input DataFrame must contain a 'date' column or a DatetimeIndex.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ticker column
    if "ticker" not in df.columns:
        # single-ticker fallback
        df["ticker"] = "UNK"

    # returns
    if "return" not in df.columns:
        price_col = None
        if "Adj Close" in df.columns:
            price_col = "Adj Close"
        elif "Close" in df.columns:
            price_col = "Close"

        if price_col is None:
            raise KeyError(
                "Cannot compute 'return': expected 'Adj Close' or 'Close' when 'return' is not present."
            )

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["return"] = df.groupby("ticker")[price_col].pct_change()

    # realized volatility
    if "realized_vol" not in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["realized_vol"] = (
            df.groupby("ticker")["return"]
              .rolling(vol_lookback, min_periods=3)
              .std()
              .reset_index(level=0, drop=True)
            * np.sqrt(252)
        )

    # log target
    df["realized_vol_log"] = np.log(df["realized_vol"].astype(float) + eps)

    # tidy
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """Add short/long vol, ratios, changes, and vol-of-vol."""
    g = df.groupby("ticker", group_keys=False)
    df["vol_3d"]    = g["realized_vol"].apply(lambda s: s.rolling(3,  min_periods=1).mean())
    df["vol_10d"]   = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=1).mean())
    df["vol_ratio"] = df["vol_3d"] / (df["vol_10d"] + eps)
    df["vol_chg"]   = df["vol_3d"] - df["vol_10d"]
    df["vol_vol"]   = g["realized_vol"].apply(lambda s: s.rolling(10, min_periods=2).std())
    # long-term EWMA proxy (used by later models)
    df["ewma_vol_10d"] = g["realized_vol"].apply(lambda s: s.ewm(span=10, adjust=False).mean())
    return df


def add_market_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """Add cross-sectional market stress and lag, plus 5d skew of returns."""
    df["market_stress"] = df.groupby("date")["return"].transform(lambda x: x.std())
    g = df.groupby("ticker", group_keys=False)
    df["market_stress_1d_lag"] = g["market_stress"].shift(1)

    def _skew5(x):
        if len(x) < 3:
            return np.nan
        m, sd = np.mean(x), np.std(x)
        sd = sd if sd > 0 else eps
        return np.mean(((x - m) / sd) ** 3)

    df["skew_5d"] = g["return"].apply(lambda s: s.rolling(5, min_periods=3).apply(_skew5, raw=True))
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar encodings and absolute return moments."""
    df["day_of_week"] = df["date"].dt.dayofweek / 6.0
    df["month_sin"]   = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["abs_return"]  = df["return"].abs()
    df["ret_sq"]      = df["return"] ** 2
    return df


def build_features(df: pd.DataFrame, include=None, exclude=None) -> pd.DataFrame:
    """
    Build all features with inclusion/exclusion control.

    Works with:
      - raw OHLCV (has 'Adj Close'/'Close'), or
      - processed long table from build_multi_dataset (has 'return' and 'realized_vol').

    Example:
        df = build_features(raw_df, exclude=["skew_5d", "ret_sq"])
    """
    include = include or [
        "vol_3d","vol_10d","vol_ratio","vol_chg","vol_vol","ewma_vol_10d",
        "market_stress","market_stress_1d_lag","skew_5d",
        "day_of_week","month_sin","month_cos","abs_return","ret_sq",
    ]
    exclude = set(exclude or [])

    df = compute_base_features(df)          # <- safe on both raw or processed
    df = add_rolling_features(df)
    df = add_market_features(df)
    df = add_calendar_features(df)

    # Filter to requested engineered features + core columns
    keep = set(include) - exclude
    core = {"date", "ticker", "return", "realized_vol", "realized_vol_log"}
    cols = [c for c in list(core) + list(keep) if c in df.columns]
    return df[cols].sort_values(["ticker", "date"]).reset_index(drop=True)