import yfinance as yf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Union, List, Dict


# ============================================================
# 🌍 Unified Fetch Function
# ============================================================
def fetch_ohlcv(
    tickers: Union[str, List[str]],
    start: str = "2000-01-01",
    end: str | None = None,
    show_progress: bool = True,
    cache_dir: str | None = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for one or more tickers from Yahoo Finance.

    :param tickers: Single ticker symbol or list of tickers.
    :type tickers: str or list[str]
    :param start: Start date for the download (YYYY-MM-DD).
    :type start: str
    :param end: End date for the download (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param show_progress: Whether to display a tqdm progress bar for multi-ticker requests.
    :type show_progress: bool
    :param cache_dir: Optional directory to cache each ticker’s data as CSV. If provided and a cache exists, it will be reused.
    :type cache_dir: str, optional
    :return: For a single ticker, a DataFrame with columns ['date','open','high','low','close','adj_close','volume']. For multiple tickers, a dict mapping ticker to its DataFrame.
    :rtype: pandas.DataFrame or dict[str, pandas.DataFrame]
    """
    # Normalize tickers
    single_mode = isinstance(tickers, str)
    tickers = (
        [tickers]
        if single_mode
        else list(dict.fromkeys([t.upper().strip() for t in tickers]))
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    iterator = (
        tqdm(tickers, desc="🌍 Fetching market data", unit="ticker")
        if show_progress and len(tickers) > 1
        else tickers
    )
    out_dict = {}

    for tkr in iterator:
        try:
            cache_path = os.path.join(cache_dir, f"{tkr}.csv") if cache_dir else None
            if cache_path and os.path.exists(cache_path):
                df = pd.read_csv(cache_path, parse_dates=["date"])
                out_dict[tkr] = df
                continue

            df = yf.download(
                tkr,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                print(f"⚠️ {tkr}: no data returned.")
                continue

            # --- Normalize columns ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index().rename(
                columns={"Date": "date", "Adj Close": "adj_close", "Close": "close"}
            )
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns and col.capitalize() in df.columns:
                    df.rename(columns={col.capitalize(): col}, inplace=True)

            if "adj_close" not in df.columns:
                df["adj_close"] = df.get("close", np.nan)

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

            # Cache save
            if cache_path:
                df.to_csv(cache_path, index=False)

            out_dict[tkr] = df

        except Exception as e:
            print(f"⚠️ Failed {tkr}: {e}")
            continue

    if single_mode:
        # Return single ticker DataFrame directly
        return next(iter(out_dict.values())) if out_dict else pd.DataFrame()
    return out_dict


# ============================================================
# 📈 Unified Dataset Builder
# ============================================================
def build_dataset(
    tickers: Union[str, List[str]],
    start: str = "2000-01-01",
    end: str | None = None,
    window: int = 15,
    cache_dir: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch raw OHLCV data and build a standardized volatility dataset.

    Computes daily returns from adjusted close (if available, else close) and realized volatility
    as a rolling standard deviation over the given window, annualized by sqrt(252).

    :param tickers: Single ticker symbol or list of tickers to include.
    :type tickers: str or list[str]
    :param start: Start date for the download (YYYY-MM-DD).
    :type start: str
    :param end: End date for the download (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param window: Rolling window length (in trading days) used to compute realized volatility.
    :type window: int
    :param cache_dir: Optional directory for CSV caching during fetch.
    :type cache_dir: str, optional
    :param show_progress: Whether to display a tqdm progress bar for multi-ticker fetches.
    :type show_progress: bool
    :raises ValueError: If no valid data is returned for any ticker.
    :return: Tidy DataFrame sorted by ['ticker','date'] with columns ['date','ticker','return','realized_vol'].
    :rtype: pandas.DataFrame
    """
    data = fetch_ohlcv(
        tickers=tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        show_progress=show_progress,
    )

    frames = []
    if isinstance(data, pd.DataFrame):  # single ticker
        data_dict = {tickers: data}
    else:
        data_dict = data

    for tkr, df in data_dict.items():
        df = df.copy()
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        df["return"] = df[price_col].pct_change()
        df["realized_vol"] = df["return"].rolling(window).std() * np.sqrt(252)
        temp = df[["date", "return", "realized_vol"]].dropna().copy()
        temp["ticker"] = tkr
        frames.append(temp)

    if not frames:
        raise ValueError("No valid data returned for any ticker.")

    dataset = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return dataset
