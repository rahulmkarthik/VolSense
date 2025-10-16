import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from volsense_pkg.data_fetching.multi_fetch import fetch_multi_ohlcv, build_multi_dataset
from volsense_pkg.data_fetching.fetch_yf import fetch_ohlcv, compute_returns_vol
from volsense_pkg.utils.evaluation import Backtester
from volsense_pkg.forecasters.forecaster_api import VolSenseForecaster


# ---------------------------------------------------------------------------
# Core single-ticker forecasting routine
# ---------------------------------------------------------------------------
def forecast_single_ticker(
    ticker,
    method="garch",
    start="2020-01-01",
    lookback=15,
    horizon=1,
    df=None,  # optional pre-fetched DataFrame
    **kwargs
):
    print(f"\n📈 Processing {ticker} ({method.upper()})...")

    try:
        # === Global model handling ===
        if method.lower() == "global_lstm":
            # fetch all required data for multiple tickers together
            raise ValueError(
                "Use forecast_multi_global() for global_lstm (multi-ticker training)."
            )

        # === Fetch data only if not provided ===
        if df is None:
            df = fetch_ohlcv(ticker, start=start)
        if df is None or len(df) < lookback:
            raise ValueError(f"No data for {ticker}")

        feat = compute_returns_vol(df, window=lookback, ticker=ticker)

        # --- normalize column naming ---
        if "vol_realized" in feat.columns and "realized_vol" not in feat.columns:
            feat = feat.rename(columns={"vol_realized": "realized_vol"})

        # --- Ensure required columns exist ---
        required = ["date", "return", "realized_vol"]
        for col in required:
            if col not in feat.columns:
                raise KeyError(f"Expected column '{col}' missing in {ticker}")

        feat["ticker"] = ticker

        # === Initialize model ===
        model = VolSenseForecaster(method=method, **kwargs)

        # === Fit and forecast ===
        if method.lower() in ["garch", "egarch", "gjr"]:
            returns_series = feat["return"].dropna()
            model.fit(returns_series)
            pred = model.predict(horizon=horizon)
        elif method.lower() == "lstm":
            model.fit(feat)
            pred, _ = model.predict(horizon=horizon)
        else:
            raise ValueError(f"Unsupported method '{method}'.")

        realized_vol = float(feat["realized_vol"].iloc[-1])
        forecast_vol = float(np.asarray(pred).flatten()[-1])

        return {
            "ticker": ticker,
            "realized_vol": realized_vol,
            "forecast_vol": forecast_vol,
        }

    except Exception as e:
        print(f"❌ {ticker}: failed due to {e}")
        return None


# ---------------------------------------------------------------------------
# Multi-ticker parallelized version (for per-ticker models)
# ---------------------------------------------------------------------------
def forecast_multi_parallel(
    tickers,
    method="garch",
    start="2020-01-01",
    lookback=15,
    horizon=1,
    max_workers=4,
    **kwargs
):
    from volsense_pkg.pipeline.forecast_interface import forecast_single_ticker

    print(f"\n🚀 Running parallel forecasts with {max_workers} workers...\n")

    data_dict = fetch_multi_ohlcv(tickers, start=start)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for ticker, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                print(f"⚠️ {ticker}: invalid data (not a DataFrame). Skipping.")
                continue

            df = df.copy()
            if "date" not in df.columns:
                df = df.reset_index()
                if "Date" in df.columns:
                    df.rename(columns={"Date": "date"}, inplace=True)

            futures[executor.submit(
                forecast_single_ticker,
                ticker=ticker,
                method=method,
                start=start,
                lookback=lookback,
                horizon=horizon,
                df=df,
                **kwargs
            )] = ticker

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"❌ {ticker}: failed due to {e}")

    if not results:
        print("⚠️ No forecasts to summarize.")
        return pd.DataFrame(columns=["ticker", "realized_vol", "forecast_vol", "vol_diff", "vol_direction"])

    df = pd.DataFrame(results)
    df["vol_diff"] = df["forecast_vol"] - df["realized_vol"]
    df["vol_direction"] = np.sign(df["vol_diff"])

    print("\n=== Multi Ticker (%s) Parallel ===" % method.upper())
    return df


# ---------------------------------------------------------------------------
# Multi-ticker sequential version
# ---------------------------------------------------------------------------
def forecast_ticker(tickers, method="garch", start="2020-01-01", lookback=15,
                    horizon=1, **kwargs):
    results = []
    for t in tickers:
        res = forecast_single_ticker(
            t, method=method, start=start, lookback=lookback,
            horizon=horizon, **kwargs
        )
        if res:
            results.append(res)
    return results


# ---------------------------------------------------------------------------
# NEW: Global LSTM (multi-ticker) unified forecast
# ---------------------------------------------------------------------------
def forecast_multi_global(
    tickers,
    start="2010-01-01",
    val_start="2025-01-01",
    lookback=15,
    window=30,
    horizon=1,
    ckpt_path="checkpoints/global_vol_lstm.pt",
    **kwargs
):
    """
    Unified routine for multi-ticker global volatility forecasting.
    Automatically loads pretrained checkpoint if available.
    Trains a new GlobalVolForecaster if not found.
    """
    print("\n🌍 Running GLOBAL LSTM multi-ticker forecast...\n")

    # Fetch and combine data
    data_dict = fetch_multi_ohlcv(tickers, start=start)
    multi = build_multi_dataset(data_dict, lookback=lookback)

    # Initialize the unified forecaster
    forecaster = VolSenseForecaster(method="global_lstm", global_ckpt_path=ckpt_path, **kwargs)

    # Try loading checkpoint first
    if os.path.exists(ckpt_path):
        print(f"📦 Found existing checkpoint at {ckpt_path}, loading instead of retraining...")
        forecaster.load_global(ckpt_path)
    else:
        print(f"⚙️ No checkpoint found — training new global model...")
        forecaster.fit(multi, val_start=val_start, window=window, horizons=horizon)

    # Generate predictions
    preds = forecaster.predict(data=multi)

    # Standardize result format
    preds["realized_vol"] = preds["ticker"].map(
        lambda t: float(multi.loc[multi["ticker"] == t, "realized_vol"].iloc[-1])
        if (multi["ticker"] == t).any() else np.nan
    )
    preds["vol_diff"] = preds["pred_vol_1"] - preds["realized_vol"]
    preds["vol_direction"] = np.sign(preds["vol_diff"])
    preds.rename(columns={"pred_vol_1": "forecast_vol"}, inplace=True)

    print("\n=== Multi Ticker (GLOBAL LSTM) ===")
    return preds[["ticker", "realized_vol", "forecast_vol", "vol_diff", "vol_direction"]]


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def summarize_forecasts(results):
    if results is None:
        print("⚠️ No forecasts to summarize.")
        return pd.DataFrame(columns=["ticker", "realized_vol", "forecast_vol", "vol_diff", "vol_direction"])

    if isinstance(results, pd.DataFrame):
        if results.empty:
            print("⚠️ No forecasts to summarize.")
            return results
        df = results.copy()
    elif isinstance(results, (list, tuple)):
        if not results:
            print("⚠️ No forecasts to summarize.")
            return pd.DataFrame(columns=["ticker", "realized_vol", "forecast_vol", "vol_diff", "vol_direction"])
        df = pd.DataFrame(results)
    else:
        print(f"⚠️ Unexpected type for results: {type(results)}")
        return pd.DataFrame(columns=["ticker", "realized_vol", "forecast_vol", "vol_diff", "vol_direction"])

    df["vol_diff"] = df["forecast_vol"] - df["realized_vol"]
    df["vol_direction"] = np.sign(df["vol_diff"])
    return df


# ---------------------------------------------------------------------------
# Unified interface (single OR multiple tickers)
# ---------------------------------------------------------------------------
def forecast_interface(tickers, method="garch", parallel=False,
                       start="2020-01-01", lookback=15, horizon=1,
                       max_workers=4, **kwargs):
    """
    Generic interface for single or multi-ticker forecasting.
    Handles GARCH, LSTM, and Global LSTM models.
    """
    tickers = [tickers] if isinstance(tickers, str) else tickers

    # --- Global LSTM special case ---
    if method.lower() == "global_lstm":
        return forecast_multi_global(
            tickers, start=start, lookback=lookback, horizon=horizon, **kwargs
        )

    # --- Standard per-ticker models ---
    if len(tickers) == 1 and not parallel:
        res = forecast_ticker(
            tickers, method=method, start=start,
            lookback=lookback, horizon=horizon, **kwargs
        )
    elif parallel:
        res = forecast_multi_parallel(
            tickers, method=method, start=start,
            lookback=lookback, horizon=horizon,
            max_workers=max_workers, **kwargs
        )
    else:
        res = forecast_ticker(
            tickers, method=method, start=start,
            lookback=lookback, horizon=horizon, **kwargs
        )

    return summarize_forecasts(res)


# ---------------------------------------------------------------------------
# Backtesting Integration
# ---------------------------------------------------------------------------
def run_backtest_for_ticker(ticker, method="lstm", start="2020-01-01",
                            lookback=15, horizon=1, plot=True, **kwargs):
    """
    Runs a backtest for a single ticker using the specified forecasting method.
    """
    backtester = Backtester(method=method, lookback=lookback, horizon=horizon, **kwargs)
    feat = backtester.prepare_data(ticker, start=start)
    df_eval = backtester.run(feat)
    metrics = backtester.evaluate(df_eval)
    if plot:
        backtester.plot(df_eval, title=f"{ticker} ({method.upper()}) Backtest")
    return metrics