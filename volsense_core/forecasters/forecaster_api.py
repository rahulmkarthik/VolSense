# volsense_core/forecasters/forecaster_api.py

import torch
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from torch.utils.data import DataLoader
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Model Imports ---
from volsense_core.models.garch_methods import ARCHForecaster
from volsense_core.models.lstm_forecaster import (
    BaseLSTM,
    MultiVolDataset,
    train_baselstm,
    evaluate_baselstm,
)
from volsense_core.models.global_vol_forecaster import (
    GlobalVolForecaster,
    build_global_splits,
    train_global_model,
    make_last_windows,
    predict_next_day,
    save_checkpoint,
    load_checkpoint,
)
from volsense_core.data_fetching.data_utils import make_rolling_windows
from volsense_core.models.lstm_forecaster import TrainConfig as LSTMTrainConfig
from volsense_core.models.global_vol_forecaster import TrainConfig as GlobalTrainConfig

__all__ = ["VolSenseForecaster"]

# ============================================================
# Shared utility for all forecasters
# ============================================================

def make_forecast_df(
    preds: np.ndarray,
    actuals: np.ndarray | None,
    dates: list,
    ticker: str,
    horizons: list[int],
    model_name: str,
):
    """
    Create a horizon-aligned forecast DataFrame for any model.
    Each (asof_date, horizon) pair has one forecast and one realized vol.

    preds:     shape (N, H)
    actuals:   shape (N, H) or None (if unknown)
    dates:     list of as-of dates
    """
    preds = np.asarray(preds)
    actuals = np.asarray(actuals) if actuals is not None else None
    rows = []

    for i, h in enumerate(horizons):
        realized_shifted = None
        if actuals is not None and actuals.ndim == 2:
            # Align future realized vols with the as-of date
            realized_shifted = np.roll(actuals[:, i], -h)
            realized_shifted[-h:] = np.nan

        for idx, d_asof in enumerate(dates):
            y_pred = float(preds[idx, i]) if preds.ndim == 2 else float(preds[idx])
            y_true = (
                float(realized_shifted[idx])
                if realized_shifted is not None and not np.isnan(realized_shifted[idx])
                else np.nan
            )
            rows.append({
                "asof_date": pd.to_datetime(d_asof),
                "date": pd.to_datetime(d_asof) + BDay(h),
                "ticker": ticker or "Unknown",
                "horizon": h,
                "forecast_vol": y_pred,
                "realized_vol": y_true,
                "model": model_name,
            })

    return pd.DataFrame(rows).sort_values(["asof_date", "horizon"])


# ============================================================
# 🌐 Unified Multi-Model Forecasting Wrapper
# ============================================================
class VolSenseForecaster:
    """
    A unified forecasting API for BaseLSTM, Global LSTM, and GARCH family models.

    Provides:
      • fit() – train model
      • predict() – generate forecasts
      • standardize() – return unified DataFrame outputs
      • evaluate_and_plot() – visualize results

    All outputs conform to a single schema:
        ['date','ticker','horizon','forecast_vol','realized_vol','model']
    """

    def __init__(self, method="lstm", device="cpu", mode="eval", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self.device = device
        self.model = None
        self.cfg = None
        self._val_loader = None
        self.ticker = kwargs.get("ticker", None)
        self.mode = mode

        # --- Global model attributes ---
        self.global_ckpt_path = kwargs.get("global_ckpt_path", None)
        self.global_ticker_to_id = None
        self.global_scalers = None
        self.global_window = None

        # --- Initialize ARCH models ---
        if self.method in ["garch", "egarch", "gjr"]:
            self.model = ARCHForecaster(
                model=self.method,
                p=kwargs.get("p", 1),
                q=kwargs.get("q", 1),
                o=kwargs.get("o", 1 if self.method == "gjr" else 0),
                dist=kwargs.get("dist", "t"),
            )

    # ============================================================
    # 🧠 Model Fitting
    # ============================================================
    def fit(self, data, **train_kwargs):
        """
        Train the selected forecaster.
          - 'lstm': expects feature DataFrame (date,ticker,realized_vol,return,...)
          - 'global_lstm': multi-ticker DataFrame
          - 'garch': 1D returns Series
        """
        if self.method == "lstm":
            print("🧩 Training BaseLSTM Forecaster...")
            self.ticker = data["ticker"].iloc[0] if "ticker" in data.columns else "Unknown"

            cfg = LSTMTrainConfig(
                window=self.kwargs.get("window", 30),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                val_start=self.kwargs.get("val_start", "2023-01-01"),
                device=self.device,
                epochs=self.kwargs.get("epochs", 20),
                lr=self.kwargs.get("lr", 5e-4),
                dropout=self.kwargs.get("dropout", 0.2),
                hidden_dim=self.kwargs.get("hidden_dim", 128),
                num_layers=self.kwargs.get("num_layers", 3),
                output_activation="none",
            )
            self.cfg = cfg
            self.model, self.hist, loaders = train_baselstm(data, cfg)
            self._val_loader = loaders[1]

        elif self.method in ["garch", "egarch", "gjr"]:
            print(f"📈 Fitting {self.method.upper()} Forecaster...")

            required_cols = {"return", "date", "ticker"}
            missing = required_cols - set(data.columns)
            if missing:
                raise KeyError(f"multi_df must contain columns: {missing}")

            ticker = data["ticker"].iloc[0]
            self.ticker = ticker
            self.data = data.copy()

            ret_series = (
                data[["date", "return"]]
                .dropna()
                .set_index("date")["return"]
            )

            self.model = ARCHForecaster(
                model=self.method,
                p=self.kwargs.get("p", 1),
                q=self.kwargs.get("q", 1),
                o=self.kwargs.get("o", 1 if self.method == "gjr" else 0),
                dist=self.kwargs.get("dist", "t"),
            )

            self.model.fit(ret_series)
            print(f"✅ {self.method.upper()} fit complete for {ticker} ({len(ret_series)} obs).")


        elif self.method == "global_lstm":
            print("🌐 Training GlobalVolForecaster...")
            cfg = GlobalTrainConfig(
                window=self.kwargs.get("window", 30),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                val_start=self.kwargs.get("val_start", "2023-01-01"),
                device=self.device,
                epochs=self.kwargs.get("epochs", 10),
            )
            self.cfg = cfg
            model, hist, val_loader, t2i, scalers, feats = train_global_model(data, cfg)
            self.model = model
            self.global_ticker_to_id = t2i
            self.global_scalers = scalers
            self.global_window = cfg.window
            if self.global_ckpt_path:
                save_checkpoint(self.global_ckpt_path, model, t2i, scalers)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    # ============================================================
    # 🔮 Prediction Interface
    # ============================================================
    def predict(self, data=None, horizon=None, mode="eval"):
        """
        Generate forecasts using the trained model.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series, optional
            Input data for prediction (used by global_lstm or garch).
        horizon : list[int] | int, optional
            Forecast horizons.
        mode : str
            "eval"      -> generate historical forecasts for backtesting
            "inference" -> generate latest forecast(s) only

        Returns
        -------
        pd.DataFrame
            Standardized forecast DataFrame with columns:
            [asof_date, date, ticker, horizon, forecast_vol, realized_vol, model]
        """
        horizon = horizon or getattr(self.cfg, "horizons", [1])

        # ------------------------------- #
        # 1️⃣ BaseLSTM
        # ------------------------------- #
        if self.method == "lstm":
            preds, actuals = evaluate_baselstm(self.model, self._val_loader, self.cfg, device=self.device)
            preds = np.atleast_2d(preds)
            actuals = np.atleast_2d(actuals)
            dates = getattr(self._val_loader.dataset, "sample_dates", [None] * len(self._val_loader.dataset))
            ticker = self.ticker or "Unknown"

            if mode == "inference":
                preds, actuals, dates = preds[-1:], actuals[-1:], dates[-1:]

            return make_forecast_df(
                preds=preds,
                actuals=actuals,
                dates=dates,
                ticker=ticker,
                horizons=self.cfg.horizons,
                model_name="BaseLSTM",
            )

        # ------------------------------- #
        # 2️⃣ GlobalVolForecaster
        # ------------------------------- #
        elif self.method == "global_lstm":
            if data is None:
                raise ValueError("GlobalLSTM requires a DataFrame for prediction.")

            if mode == "inference":
                # ✅ current behavior (latest forecasts only)
                df_last_windows = make_last_windows(data, window=self.global_window)
                preds_df = predict_next_day(
                    self.model,
                    df_last_windows,
                    self.global_ticker_to_id,
                    self.global_scalers,
                    window=self.global_window,
                    device=self.device,
                    show_progress=False
                )
                preds = np.stack(preds_df["forecast_vol_scaled"].values)
                dates = df_last_windows["date"].values
                ticker = df_last_windows["ticker"].values[0]

                return make_forecast_df(
                    preds=np.expand_dims(preds, axis=0),
                    actuals=None,
                    dates=[dates[-1]],
                    ticker=ticker,
                    horizons=self.cfg.horizons,
                    model_name="GlobalVolForecaster",
                )

            elif mode == "eval":
                # 🚀 TODO: historical rolling prediction mode
                from tqdm import tqdm
                preds_all, dates_all = [], []
                for _, g in tqdm(data.groupby("ticker"), desc="Rolling eval forecasts"):
                    windows = make_rolling_windows(g, window=self.global_window, stride=5)
                    for w in windows:
                        df_win = make_last_windows(w, window=self.global_window)
                        preds_df = predict_next_day(
                            self.model,
                            df_win,
                            self.global_ticker_to_id,
                            self.global_scalers,
                            window=self.global_window,
                            device=self.device,
                            show_progress=False
                        )
                        preds_all.append(np.stack(preds_df["forecast_vol_scaled"].values))
                        dates_all.append(w["date"].iloc[-1])
                preds_all = np.array(preds_all)
                ticker = data["ticker"].iloc[0]
                return make_forecast_df(
                    preds=preds_all,
                    actuals=None,
                    dates=dates_all,
                    ticker=ticker,
                    horizons=self.cfg.horizons,
                    model_name="GlobalVolForecaster",
                )

        # ------------------------------- #
        # 3️⃣ GARCH / EGARCH / GJR
        # ------------------------------- #
        elif self.method in ["garch", "egarch", "gjr"]:
            if self.model is None:
                raise RuntimeError("GARCH model not fitted yet. Call .fit() first.")
            if data is None:
                data = getattr(self, "data", None)
                if data is None:
                    raise ValueError("Must provide multi_df for GARCH prediction.")

            required_cols = {"return", "date", "ticker"}
            missing = required_cols - set(data.columns)
            if missing:
                raise KeyError(f"multi_df must contain columns: {missing}")

            ticker = data["ticker"].iloc[0]
            ret_series = data[["date", "return"]].dropna().set_index("date")["return"]

            mode = mode or "inference"
            horizons = horizon if isinstance(horizon, (list, tuple)) else [horizon]

            # ====================================================
            # 📊 Evaluation Mode — rolling one-step-ahead forecasts
            # ====================================================
            if mode == "eval":
                print(f"🌀 Running rolling 1-step-ahead {self.method.upper()} evaluation on {ticker}...")

                lookback = 21
                realized_vol = (
                    ret_series.rolling(window=lookback).std() * np.sqrt(252)
                ).rename("realized_vol")

                rolling_preds = self.model.rolling_forecast(ret_series, refit_every=5)
                rolling_preds = rolling_preds.rename("forecast_vol")

                df_eval = (
                    pd.concat([rolling_preds, realized_vol], axis=1)
                    .dropna(subset=["forecast_vol"])
                    .copy()
                )

                df_eval["asof_date"] = df_eval.index
                df_eval["date"] = df_eval["asof_date"] + BDay(1)
                df_eval["ticker"] = ticker
                df_eval["horizon"] = 1
                df_eval["model"] = self.method.upper()

                # Normalize dates
                df_eval["asof_date"] = pd.to_datetime(df_eval["asof_date"]).dt.normalize()
                df_eval["date"] = pd.to_datetime(df_eval["date"]).dt.normalize()

                df_out = df_eval.reset_index(drop=True)[
                    ["asof_date", "date", "ticker", "horizon", "forecast_vol", "realized_vol", "model"]
                ]

                print(f"✅ {self.method.upper()} evaluation complete ({len(df_out)} rows).")
                return df_out

            # ====================================================
            # 🚀 Inference Mode — multi-horizon forward forecasts
            # ====================================================
            elif mode == "inference":
                last_date = pd.to_datetime(data["date"].iloc[-1])
                preds = []

                for h in horizons:
                    sigma_h = self.model.predict(horizon=h)
                    preds.append(
                        {
                            "asof_date": last_date.normalize(),
                            "date": (last_date + BDay(h)).normalize(),
                            "ticker": ticker,
                            "horizon": h,
                            "forecast_vol": float(sigma_h[-1]),
                            "realized_vol": np.nan,
                            "model": self.method.upper(),
                        }
                    )

                df_out = pd.DataFrame(preds)
                print(f"✅ {self.method.upper()} inference complete ({len(df_out)} horizons).")
                return df_out



        else:
            raise ValueError(f"Unknown method: {self.method}")