import yfinance as yf
import pandas as pd
import numpy as np
import re


def fetch_ohlcv(ticker: str, start="2000-01-01", end=None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker from Yahoo Finance.

    Handles all yfinance quirks:
    - MultiIndex or ticker-suffixed columns like 'adj close spy' or 'close_spy'
    - Auto-adjust and capitalization issues
    - Missing Adj Close fallback
    Always returns columns:
        ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.reset_index()

    # --- Normalize column names ---
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Remove any extra symbols and clean spaces
    df.columns = [re.sub(r"[^a-z0-9_ ]+", "", c).strip() for c in df.columns]

    # --- Remove ticker suffixes like ' close spy ' or 'close_spy' ---
    pattern_space = rf"( {ticker.lower()})$"
    pattern_underscore = rf"(_{ticker.lower()})$"
    df.columns = [re.sub(pattern_space, "", c).strip() for c in df.columns]
    df.columns = [re.sub(pattern_underscore, "", c).strip() for c in df.columns]

    # --- Standardize common names ---
    rename_map = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "date_": "date",
        "datetime": "date",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # --- Fallbacks ---
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    if "adj_close" not in df.columns:
        raise KeyError(f"No valid price column found in df: {df.columns}")

    if "date" not in df.columns:
        df.insert(0, "date", df.index)

    df["date"] = pd.to_datetime(df["date"])

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