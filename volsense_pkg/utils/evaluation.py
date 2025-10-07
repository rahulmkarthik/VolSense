# volsense_pkg/utils/evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from volsense_pkg.utils.metrics import evaluate_forecasts
from volsense_pkg.forecasters.forecaster_api import VolSenseForecaster
from volsense_pkg.data_fetching.fetch_yf import fetch_ohlcv, compute_returns_vol


class Backtester:
    """
    Runs historical forecasts and evaluates performance of a given model.
    """

    def __init__(self, method="lstm", lookback=15, horizon=1, **kwargs):
        self.method = method.lower()
        self.lookback = lookback
        self.horizon = horizon
        self.kwargs = kwargs
        self.model = None

    def prepare_data(self, ticker, start="2020-01-01"):
        df = fetch_ohlcv(ticker, start=start)
        feat = compute_returns_vol(df, window=self.lookback, ticker=ticker)
        # Normalize naming
        if "vol_realized" in feat.columns:
            feat = feat.rename(columns={"vol_realized": "realized_vol"})
        return feat.dropna()

    def run(self, feat):
        f = VolSenseForecaster(method=self.method, window=self.lookback, horizon=self.horizon, **self.kwargs)

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
        return df_eval

    def evaluate(self, df_eval):
        return evaluate_forecasts(df_eval)

    def plot(self, df_eval, title=None):
        plt.figure(figsize=(12, 5))
        plt.plot(df_eval["realized_vol"].values, label="Realized Volatility")
        plt.plot(df_eval["forecast_vol"].values, label="Forecast Volatility")
        plt.title(title or f"{self.method.upper()} Backtest")
        plt.legend()
        plt.tight_layout()
        plt.show()



class RollingBacktester(Backtester):
    """
    Performs rolling walk-forward backtesting for volatility forecasts.
    At each step, fits model on past data (up to lookback window)
    and predicts horizon steps ahead.
    """

    def __init__(self, method="lstm", lookback=15, horizon=1, step=1, **kwargs):
        super().__init__(method=method, lookback=lookback, horizon=horizon, **kwargs)
        self.step = step  # step size for rolling window

    def _run_single_ticker(self, ticker, start="2020-01-01", end=None):
        feat = self.prepare_data(ticker, start=start)
        if end:
            feat = feat[feat["date"] <= end]

        preds, actuals, dates = [], [], []

        for i in range(self.lookback, len(feat) - self.horizon, self.step):
            train_df = feat.iloc[:i].copy()
            test_df = feat.iloc[i:i + self.horizon].copy()

            try:
                if self.method in ["garch", "egarch", "gjr"]:
                    returns_series = train_df["return"].dropna()
                    f = VolSenseForecaster(method=self.method, **self.kwargs)
                    f.fit(returns_series)
                    pred = f.predict(horizon=self.horizon)
                    preds.append(pred[-1])
                else:
                    f = VolSenseForecaster(method=self.method, window=self.lookback,
                                           horizon=self.horizon, **self.kwargs)
                    f.fit(train_df)
                    pred, _ = f.predict()
                    preds.append(np.asarray(pred).flatten()[-1])

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
        return df_bt

    def run_rolling(self, ticker, start="2020-01-01", end=None):
        df_bt = self._run_single_ticker(ticker, start=start, end=end)
        metrics = self.evaluate(df_bt)
        return df_bt, metrics


class ParallelRollingBacktester(RollingBacktester):
    """
    Runs rolling backtests for multiple tickers in parallel threads.
    """

    def __init__(self, method="lstm", lookback=15, horizon=1, step=1, max_workers=4, **kwargs):
        super().__init__(method=method, lookback=lookback, horizon=horizon, step=step, **kwargs)
        self.max_workers = max_workers

    def run_parallel(self, tickers, start="2020-01-01", end=None):
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_single_ticker, t, start, end): t for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df_bt = future.result()
                    metrics = self.evaluate(df_bt)
                    results[ticker] = {"data": df_bt, "metrics": metrics}
                    print(f"✅ {ticker}: backtest completed.")
                except Exception as e:
                    print(f"❌ {ticker}: backtest failed due to {e}")
                    results[ticker] = None

        return results

