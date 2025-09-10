import yfinance as yf
import pandas as pd
import numpy as np


def fetch_ohlcv(ticker: str, start="2000-01-01", end=None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker.
    Always returns flat columns: date, open, high, low, close, adj_close, volume.
    """
    df = yf.download(ticker, start=start, end=end)

    # Reset index to get Date as a column
    df = df.reset_index()

    # Flatten column names if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Standardize "adj close" naming
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    elif "close" in df.columns:
        # fallback if adj close not available
        df = df.rename(columns={"close": "adj_close"})

    # Standardize "date" naming
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    elif "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})

    # Ensure only expected columns exist
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    available = [c for c in expected if c in df.columns]

    return df[available]




def compute_returns_vol(df, window=21, ticker=None):
    out = df.copy()

    # detect adjusted close column
    if "adj_close" in out.columns:       # lowercase
        price_col = "adj_close"
    elif "Adj Close" in out.columns:     # original Yahoo naming
        out = out.rename(columns={"Adj Close": "adj_close"})
        price_col = "adj_close"
    elif "Close" in out.columns:
        out = out.rename(columns={"Close": "close"})
        price_col = "close"
    elif "close" in out.columns:         # lowercase fallback
        price_col = "close"
    else:
        raise KeyError(f"No valid price column found in df: {out.columns}")

    # compute returns
    out["return"] = out[price_col].pct_change()

    # realized volatility
    out["vol_realized"] = out["return"].rolling(window).std() * np.sqrt(252)

    # optional ticker
    if ticker:
        out["ticker"] = ticker
        return out[["date", "return", "vol_realized", "ticker"]].dropna()
    else:
        return out[["date", "return", "vol_realized"]].dropna()