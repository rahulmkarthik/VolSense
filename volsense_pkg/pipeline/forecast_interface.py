# volsense_pkg/pipeline/forecast_interface.py
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict

from volsense_pkg.data_fetching.fetch_yf import fetch_ohlcv, compute_returns_vol
from volsense_pkg.forecasters.forecaster_api import VolSenseForecaster


def forecast_ticker(
    tickers: Union[str, List[str]],
    method: str = "lstm",
    start: str = "2020-01-01",
    end: str = None,
    lookback: int = 21,
    window: int = 30,
    horizon: int = 1,
    epochs: int = 5,
    batch_size: int = 32,
    **kwargs
) -> List[Dict]:
    """
    Unified forecasting interface for one or more tickers.
    Supports both LSTM and GARCH forecasters.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    results = []

    for ticker in tickers:
        print(f"\n📈 Processing {ticker} ({method.upper()})...")

        try:
            # --- Step 1: Fetch data ---
            df = fetch_ohlcv(ticker, start=start, end=end)
            df = compute_returns_vol(df, window=lookback, ticker=ticker)
            df = df.rename(columns={"vol_realized": "realized_vol"})

            if df.empty:
                print(f"⚠️ Skipping {ticker}: no data returned.")
                continue

            # --- Step 2: Initialize and fit model ---
            model = VolSenseForecaster(method=method, window=window, horizon=horizon, **kwargs)
            if method == "lstm":
                model.fit(df, epochs=epochs, batch_size=batch_size)
            elif method == "garch":
                returns_series = df["return"].dropna()
                model.fit(returns_series)
            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'lstm' or 'garch'.")

            # --- Step 3: Forecast next-day volatility ---
            if method == "lstm":
                preds, actuals = model.predict()
                forecast_vol = float(preds[-1]) if len(preds) > 0 else np.nan
            else:
                forecast_vol = float(model.model.predict(horizon=horizon)[-1])

            latest_realized = df["realized_vol"].iloc[-1]

            results.append({
                "ticker": ticker,
                "realized_vol": latest_realized,
                "forecast_vol": forecast_vol,
                "model": model
            })

        except Exception as e:
            print(f"❌ {ticker}: failed due to {e}")

    return results


# ---------------------------------------------------------------------
# 🧵 Parallelized version
# ---------------------------------------------------------------------
def forecast_multi_parallel(
    tickers: List[str],
    method: str = "lstm",
    max_workers: int = 4,
    **kwargs
) -> List[Dict]:
    """
    Parallelized forecasting across multiple tickers using threads.
    Shares same args as forecast_ticker().

    Example:
        forecast_multi_parallel(["SPY", "AAPL", "GOOG"], method="garch", max_workers=3)
    """
    results = []
    print(f"\n🚀 Running parallel forecasts with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(forecast_ticker, t, method=method, **kwargs): t
            for t in tickers
        }

        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.extend(res)
            except Exception as e:
                print(f"❌ {ticker}: failed due to {e}")

    return results


# ---------------------------------------------------------------------
# 📊 Summarizer
# ---------------------------------------------------------------------
def summarize_forecasts(results: List[Dict]) -> pd.DataFrame:
    """
    Convert the output of forecast_ticker() or forecast_multi_parallel() into a DataFrame.
    """
    df = pd.DataFrame([
        {
            "ticker": r["ticker"],
            "realized_vol": r["realized_vol"],
            "forecast_vol": r["forecast_vol"]
        }
        for r in results
    ])
    df["vol_diff"] = df["forecast_vol"] - df["realized_vol"]
    df["vol_direction"] = np.sign(df["vol_diff"])
    return df
