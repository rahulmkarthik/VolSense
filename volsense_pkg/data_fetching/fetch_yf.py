# volsense_pkg/data/fetch_yf.py
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_ohlcv(ticker: str, start: str = "2005-01-01", end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker (str): e.g. "SPY", "AAPL", "BTC-USD"
        start (str): start date (YYYY-MM-DD)
        end (str): end date (YYYY-MM-DD). Defaults to today if None.
        interval (str): data interval ("1d", "1h", etc.)

    Returns:
        pd.DataFrame with datetime index and OHLCV columns.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=str.lower)  # lowercase col names
    return df


def compute_returns_vol(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute log returns and realized volatility.

    Args:
        df (pd.DataFrame): OHLCV data with 'close'
        window (int): rolling window size (days) for realized volatility

    Returns:
        pd.DataFrame with returns + realized volatility
    """
    out = df.copy()
    out["return"] = np.log(out["close"] / out["close"].shift(1))
    out["vol_realized"] = out["return"].rolling(window).std() * np.sqrt(252)  # annualized
    return out.dropna()
