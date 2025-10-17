# ============================================================
# volsense_pkg/utils/evaluation.py
# Unified backtesting framework for all VolSense models
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from volsense_core.utils.metrics import evaluate_forecasts
from volsense_core.forecasters.forecaster_api import VolSenseForecaster
from volsense_core.data_fetching.fetch_yf import fetch_ohlcv, compute_returns_vol


# ============================================================
# 🧠 Base Backtester
# ============================================================
class Backtester:
    """
    Generic backtester for any supported VolSense model:
    GARCH, LSTM (single ticker), or Global LSTM (multi-horizon).
    """

    def __init__(self, method="lstm", lookback=15, horizon=1, step=1, device="cpu", **kwargs):
        self.method = method.lower()
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.device = device
        self.kwargs = kwargs

    # -----------------------------
    # Data Preparation
    # -----------------------------
    def prepare_data(self, ticker, start="2020-01-01", end=None):
        df = fetch_ohlcv(ticker, start=start)
        feat = compute_returns_vol(df, window=self.lookback, ticker=ticker)
        if "vol_realized" in feat.columns:
            feat = feat.rename(columns={"vol_realized": "realized_vol"})
        if end:
            feat = feat[feat["date"] <= end]
        return feat.dropna()

    # -----------------------------
    # Single Evaluation Run
    # -----------------------------
    def run_once(self, feat):
        """Fit once and evaluate."""
        f = VolSenseForecaster(method=self.method, window=self.lookback,
                               horizon=self.horizon, device=self.device, **self.kwargs)

        if self.method in ["garch", "egarch", "gjr"]:
            returns_series = feat["return"].dropna()
            f.fit(returns_series)
            preds = f.predict(horizon=self.horizon)
            y_true = feat["realized_vol"].iloc[-len(preds):].values
        else:
            f.fit(feat)
            preds, actuals = f.predict()
            preds = np.array(preds).flatten()
            y_true = np.array(actuals).flatten()

        df_eval = pd.DataFrame({
            "realized_vol": y_true,
            "forecast_vol": preds[-len(y_true):]
        })
        return df_eval, evaluate_forecasts(df_eval)

    # -----------------------------
    # Rolling / Walk-Forward Evaluation
    # -----------------------------
    def run_rolling(self, ticker, start="2020-01-01", end=None):
        """Walk-forward rolling evaluation for one ticker."""
        feat = self.prepare_data(ticker, start=start, end=end)
        preds, actuals, dates = [], [], []

        for i in range(self.lookback, len(feat) - self.horizon, self.step):
            train_df = feat.iloc[:i].copy()
            test_df = feat.iloc[i:i + self.horizon].copy()
            try:
                if self.method in ["garch", "egarch", "gjr"]:
                    f = VolSenseForecaster(method=self.method, **self.kwargs)
                    f.fit(train_df["return"].dropna())
                    pred = f.predict(horizon=self.horizon)[-1]
                elif self.method == "global_lstm":
                    # Global model predicts all tickers at once
                    f = VolSenseForecaster(method=self.method, device=self.device, **self.kwargs)
                    f.load_global(self.kwargs.get("global_ckpt_path"))
                    pred_df = f.predict_global(train_df)
                    pred = pred_df["forecast_vol_scaled"].iloc[0]
                else:
                    f = VolSenseForecaster(method=self.method, window=self.lookback,
                                           horizon=self.horizon, device=self.device, **self.kwargs)
                    f.fit(train_df)
                    pred, _ = f.predict()
                    pred = np.asarray(pred).flatten()[-1]

                preds.append(pred)
                actuals.append(test_df["realized_vol"].iloc[-1])
                dates.append(test_df["date"].iloc[-1])

            except Exception as e:
                print(f"⚠️ {ticker}: Rolling step {i} failed: {e}")
                continue

        df_bt = pd.DataFrame({
            "date": dates,
            "realized_vol": actuals,
            "forecast_vol": preds
        })
        metrics = evaluate_forecasts(df_bt)
        return df_bt, metrics

    # -----------------------------
    # Multi-Ticker Parallel Backtesting
    # -----------------------------
    def run_parallel(self, tickers, start="2020-01-01", end=None, max_workers=4):
        """
        Run rolling backtests for multiple tickers in parallel threads.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_rolling, t, start, end): t for t in tickers
            }
            for future in as_completed(futures):
                t = futures[future]
                try:
                    df_bt, metrics = future.result()
                    results[t] = {"data": df_bt, "metrics": metrics}
                    print(f"✅ {t}: backtest completed.")
                except Exception as e:
                    print(f"❌ {t}: backtest failed due to {e}")
                    results[t] = None
        return results

    # -----------------------------
    # Plotting
    # -----------------------------
    @staticmethod
    def plot(df_eval, title=None):
        plt.figure(figsize=(12, 5))
        plt.plot(df_eval["realized_vol"].values, label="Realized Volatility")
        plt.plot(df_eval["forecast_vol"].values, label="Forecast Volatility")
        plt.title(title or "Volatility Backtest")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# 🧩 Multi-Horizon Extension (Global LSTM)
# ============================================================
class MultiHorizonBacktester(Backtester):
    """
    Backtester for multi-horizon global LSTM models.
    Generates per-horizon forecasts and evaluates all at once.
    """

    def __init__(self, ckpt_path, horizons=(1, 3, 5), lookback=30, device="cpu", **kwargs):
        super().__init__(method="global_lstm", lookback=lookback, horizon=max(horizons), device=device, **kwargs)
        self.horizons = list(horizons)
        self.ckpt_path = ckpt_path

    def run(self, df):
        df = df.copy().sort_values(["ticker", "date"])
        self.model = VolSenseForecaster(method="global_lstm", device=self.device)
        self.model.load_global(self.ckpt_path)

        results = []
        for t, g in tqdm(df.groupby("ticker"), desc="Multi-horizon backtest"):
            g = g.reset_index(drop=True)
            for i in range(self.lookback, len(g) - max(self.horizons)):
                window = g.iloc[i - self.lookback:i].copy()
                actuals = {h: g.iloc[i + h - 1]["realized_vol"] for h in self.horizons}
                preds_df = self.model.predict_global(window)
                row = preds_df[preds_df["ticker"] == t].iloc[0]
                yhat_list = row["forecast_vol_scaled"]

                for j, h in enumerate(self.horizons):
                    if j < len(yhat_list):
                        results.append({
                            "ticker": t,
                            "date": g.iloc[i]["date"],
                            "horizon": h,
                            "actual": actuals[h],
                            "pred": yhat_list[j],
                        })

        df_res = pd.DataFrame(results)
        metrics = self._evaluate_multi(df_res)
        return df_res, metrics

    def _evaluate_multi(self, df):
        all_metrics = []
        for h in self.horizons:
            sub = df[df["horizon"] == h]
            m = evaluate_forecasts(sub.rename(columns={"actual": "realized_vol", "pred": "forecast_vol"}))
            m["horizon"] = h
            all_metrics.append(m)
        metrics_df = pd.DataFrame(all_metrics)
        print("\n===== Multi-Horizon Metrics =====")
        print(metrics_df[["horizon", "rmse", "mae", "mape"]].to_string(index=False))
        print("=================================")
        return metrics_df