# volsense_pkg/data_fetching/data_utils.py
import os
import pandas as pd
from pathlib import Path
from .fetch_yf import fetch_ohlcv, compute_returns_vol
from .multi_fetch import fetch_multi_ohlcv, build_multi_dataset

DATA_CACHE = Path(os.getenv("VOLSENSE_DATA", "./.volsense_cache"))
DATA_CACHE.mkdir(parents=True, exist_ok=True)

def save_to_cache(df: pd.DataFrame, name: str):
    """
    Save a DataFrame to local parquet cache.
    """
    path = DATA_CACHE / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path

def load_from_cache(name: str) -> pd.DataFrame:
    """
    Load a DataFrame from parquet cache if it exists.
    """
    path = DATA_CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"No cache found for {name}")

def get_or_fetch_single(ticker: str, start="2000-01-01", end=None, use_cache=True):
    """
    Fetch OHLCV + realized vol for a single ticker with optional caching.
    """
    cache_name = f"{ticker}_{start}_{end}".replace(":", "-")
    if use_cache:
        try:
            return load_from_cache(cache_name)
        except FileNotFoundError:
            pass
    
    df = fetch_ohlcv(ticker, start=start, end=end)
    df = compute_returns_vol(df, ticker=ticker)
    if use_cache:
        save_to_cache(df, cache_name)
    return df

def get_or_fetch_multi(tickers, start="2000-01-01", end=None, lookback=21, use_cache=True):
    """
    Fetch OHLCV + realized vol for multiple tickers with optional caching.
    """
    cache_name = f"{'_'.join(tickers)}_{start}_{end}_{lookback}".replace(":", "-")
    if use_cache:
        try:
            return load_from_cache(cache_name)
        except FileNotFoundError:
            pass
    
    raw_dict = fetch_multi_ohlcv(tickers, start=start, end=end)
    df = build_multi_dataset(raw_dict, lookback=lookback)
    if use_cache:
        save_to_cache(df, cache_name)
    return df

# ============================================================
# 🔁 make_rolling_windows: Generate rolling subwindows
# ============================================================


def make_rolling_windows(df: pd.DataFrame, window: int = 30, stride: int = 5):
    """
    Generate rolling subwindows of a DataFrame for evaluation or backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series (must include a 'date' column sorted ascending).
    window : int, default=30
        Length of each rolling window.
    stride : int, default=5
        Step size between window start indices.

    Returns
    -------
    list[pd.DataFrame]
        List of rolling DataFrame segments, each of length `window`.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must include a 'date' column for rolling windows.")

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    windows = []

    for start in range(0, n - window + 1, stride):
        sub = df.iloc[start:start + window].copy()
        windows.append(sub)

    return windows

