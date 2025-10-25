import yfinance as yf
import pandas as pd
import numpy as np


def fetch_multi_ohlcv(tickers, start="2000-01-01", end=None):
    """
    Fetch OHLCV data for multiple tickers into a dict of normalized DataFrames.
    Ensures:
    ✅ Flat columns (no MultiIndex)
    ✅ 'date' column always present
    ✅ 'Adj Close' available (fallbacks to 'Close')
    ✅ Backward compatible with all downstream notebooks
    """
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False)
            if df is None or df.empty:
                print(f"⚠️ {ticker}: no data returned.")
                continue

            # --- Flatten MultiIndex columns if any ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            # --- Ensure 'Adj Close' column exists ---
            if "Adj Close" not in df.columns:
                if "Close" in df.columns:
                    df["Adj Close"] = df["Close"]
                else:
                    raise KeyError(f"{ticker}: neither 'Adj Close' nor 'Close' found.")

            # --- Standardize datetime index/column ---
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")

            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # --- Clean up names ---
            df.columns.name = None
            df.index.name = None

            data_dict[ticker] = df

        except Exception as e:
            print(f"⚠️ Failed {ticker}: {e}")

    return data_dict


def build_multi_dataset(data_dict, window=21):
    """
    Combine multiple normalized ticker DataFrames into a single long-form DataFrame.
    Output columns: ['date', 'return', 'realized_vol', 'ticker']

    Backward-compatible and safe for both training & inference.
    """
    frames = []
    for ticker, df in data_dict.items():
        df = df.copy()
        if "date" not in df.columns:
            # Fallback for any unnormalized input
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Ensure Adj Close
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Compute returns + realized volatility
        df["return"] = df["Adj Close"].pct_change()
        df["realized_vol"] = df["return"].rolling(window).std() * np.sqrt(252)

        temp = df[["date", "return", "realized_vol"]].dropna().copy()
        temp["ticker"] = ticker
        frames.append(temp)

    if not frames:
        raise ValueError("No valid data fetched for any ticker!")

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["ticker", "date"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.columns.name = None
    out.index.name = None
    return out
