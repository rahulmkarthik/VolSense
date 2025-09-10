# volsense_pkg/data_fetching/multi_fetch.py
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_multi_ohlcv(tickers, start="2000-01-01", end=None):
    """
    Fetch OHLCV data for multiple tickers into a dict of DataFrames.
    Ensures Adj Close is always present.
    """
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False)  # keep Adj Close
            if "Adj Close" not in df.columns:
                print(f"⚠️ {ticker}: 'Adj Close' missing, using 'Close' instead.")
                df["Adj Close"] = df["Close"]
            data_dict[ticker] = df
        except Exception as e:
            print(f"⚠️ Failed {ticker}: {e}")
    return data_dict


# volsense_pkg/data_fetching/multi_fetch.py

def build_multi_dataset(data_dict, lookback=21):
    """
    Combine multiple tickers into a long DF with columns:
    ['date', 'return', 'realized_vol', 'ticker']
    """
    import numpy as np
    import pandas as pd

    frames = []
    for ticker, df in data_dict.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Ensure Adj Close exists, else fallback to Close
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Features/target
        df["return"] = df["Adj Close"].pct_change()
        df["realized_vol"] = df["return"].rolling(lookback).std() * np.sqrt(252)

        # Keep only what we need and drop NaNs
        temp = df[["return", "realized_vol"]].dropna().copy()
        temp["ticker"] = ticker

        # Make the index a column and standardize its name to 'date'
        temp = temp.reset_index()
        if "Date" in temp.columns:
            temp = temp.rename(columns={"Date": "date"})
        elif "index" in temp.columns:  # safety for unnamed index
            temp = temp.rename(columns={"index": "date"})
        temp["date"] = pd.to_datetime(temp["date"])

        # Clean up column-index name (that "Price" label you saw)
        temp.columns.name = None

        frames.append(temp[["date", "return", "realized_vol", "ticker"]])

    if not frames:
        raise ValueError("No valid data fetched for any ticker!")

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["ticker", "date"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out