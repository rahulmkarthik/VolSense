# volsense_core/forecasters/forecaster_api.py

from volsense_core.models.garch_methods import ARCHForecaster
from volsense_core.models.lstm_forecaster import (
    BaseLSTM,
    MultiVolDataset,
    train_lstm,
    evaluate_lstm,
)

# --- New imports for Global LSTM integration ---
from volsense_core.models.global_vol_forecaster import (
    GlobalVolForecaster,
    build_global_splits,
    train_global_model,
    TrainConfig,
    make_last_windows,
    predict_next_day,
    save_checkpoint,
    load_checkpoint,
)

from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


__all__ = [
    "ARCHForecaster",
    "BaseLSTM",
    "MultiVolDataset",
    "train_lstm",
    "evaluate_lstm",
    "VolSenseForecaster",
]


# ============================================================
# Unified VolSense Forecaster Wrapper
# ============================================================

class VolSenseForecaster:
    def __init__(self, method="lstm", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self.window = kwargs.get("window", 15)
        self.horizon = kwargs.get("horizon", 1)
        self.model = None
        self._val_loader = None
        self.device = kwargs.get("device", "cpu")

        # --- Global model attributes ---
        self.global_ckpt_path = kwargs.get("global_ckpt_path", None)
        self.global_ticker_to_id = None
        self.global_scalers = None
        self.global_window = None

        # --- GARCH initialization ---
        if self.method in ["garch", "egarch", "gjr"]:
            self.model = ARCHForecaster(
                model_type=self.method,
                p=kwargs.get("p", 1),
                q=kwargs.get("q", 1),
                dist=kwargs.get("dist", "normal"),
                scale=kwargs.get("scale", 100),
            )

    # ============================================================
    # LSTM / GARCH Training
    # ============================================================
    def fit(self, data_or_returns, epochs=5, batch_size=64):
        """
        Fit the selected forecaster.
        - For 'garch': data_or_returns is a 1D returns series.
        - For 'lstm': data_or_returns is a DataFrame with ['date','ticker','realized_vol'].
        - For 'global_lstm': same DataFrame, but trains cross-ticker model.
        """
        if self.method == "lstm":
            # Build dataset (with scaling)
            self._dataset = MultiVolDataset(data_or_returns, window=self.window, horizon=self.horizon)

            # Split train/val
            train_size = int(0.8 * len(self._dataset))
            val_size = len(self._dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(self._dataset, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self._val_loader = DataLoader(val_ds, batch_size=batch_size)

            # Determine n_horizons automatically
            n_horizons = len(self.horizon) if isinstance(self.horizon, (list, tuple)) else 1

            self.model = LSTMForecaster(
                input_dim=1,
                hidden_dim=self.kwargs.get("hidden_dim", 64),
                num_layers=self.kwargs.get("num_layers", 2),
                dropout=self.kwargs.get("dropout", 0.2),
                n_horizons=n_horizons,
            )

            # Train the LSTM model on scaled data
            self.model = train_lstm(
                self.model,
                train_loader,
                self._val_loader,
                epochs=epochs,
                lr=self.kwargs.get("lr", 1e-3),
                device=self.device,
            )

        elif self.method in ["garch", "egarch", "gjr"]:
            self.model.fit(data_or_returns)

        elif self.method == "global_lstm":
            return self.fit_global(
                data_or_returns,
                val_start=self.kwargs.get("val_start", "2025-01-01"),
                window=self.kwargs.get("window", 30),
                horizons=self.kwargs.get("horizons", 1),
                stride=self.kwargs.get("stride", 2),
                epochs=self.kwargs.get("epochs", 10),
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    # ============================================================
    # Global LSTM Fit / Load / Predict (Updated for new architecture)
    # ============================================================
    def fit_global(self, df, val_start="2025-01-01", window=30, horizons=1, stride=2, epochs=10):
        """
        Train or fine-tune a global LSTM volatility model across multiple tickers.
        Expects df with ['date', 'ticker', 'realized_vol', 'return'].
        """
        print(f"\n🌐 Training GlobalVolForecaster v2 (window={window}, horizon={horizons})")

        # --- Prepare configuration for the new API ---
        cfg = TrainConfig(
            window=window,
            horizons=horizons if isinstance(horizons, int) else len(horizons),
            stride=stride,
            val_start=val_start,
            epochs=epochs,
            device=self.device,
            oversample_high_vol=True,
        )

        # --- Train via new train_global_model() ---
        model, history, val_loader, t2i, scalers, features = train_global_model(df, cfg)

        # --- Save artifacts in the wrapper ---
        self.model = model
        self.global_ticker_to_id = t2i
        self.global_scalers = scalers
        self.global_window = window
        self.features = features

        if self.global_ckpt_path:
            save_checkpoint(self.global_ckpt_path, model, t2i, scalers)
            print(f"✅ Global model checkpoint saved to {self.global_ckpt_path}")

        return history


    def load_global(self, ckpt_path):
        """Load pretrained global model checkpoint (new version)."""
        print(f"\n📦 Loading pretrained GlobalVolForecaster v2 from {ckpt_path}")
        model, t2i, scalers = load_checkpoint(ckpt_path, device=self.device)
        self.model = model
        self.global_ticker_to_id = t2i
        self.global_scalers = scalers
        self.global_window = model.window
        print(f"✅ Loaded GlobalVolForecaster ({len(t2i)} tickers, window={model.window})")


    def predict_global(self, df):
        """Generate next-day volatility forecasts for all tickers using the v2 model."""
        if self.model is None:
            raise RuntimeError("Global model not trained or loaded.")

        # Prepare last input window per ticker
        from volsense_core.models.global_vol_forecaster import make_last_windows, predict_next_day
        df_last_windows = make_last_windows(df, window=self.global_window)

        preds = predict_next_day(
            self.model,
            df_last_windows,
            self.global_ticker_to_id,
            self.global_scalers,
            window=self.global_window,
            device=self.device,
        )
        return preds


    # ============================================================
    # Unified Prediction Interface
    # ============================================================
    def predict(self, horizon=None, loader=None, data=None):
        """
        Predict using the trained forecaster.
        - 'garch': specify horizon (int).
        - 'lstm': specify DataLoader or uses stored val_loader.
        - 'global_lstm': provide historical DataFrame for all tickers.
        """
        if self.method == "garch":
            horizon = horizon or self.horizon
            return self.model.predict(horizon=horizon)

        elif self.method == "lstm":
            loader = loader or self._val_loader
            if loader is None:
                raise RuntimeError("No val_loader stored; pass a DataLoader or re-fit first.")
            preds, actuals = evaluate_lstm(self.model, loader, device=self.device)
            return preds, actuals

        elif self.method == "global_lstm":
            if data is None:
                raise ValueError("Must provide 'data' DataFrame with ['date','ticker','realized_vol']")
            return self.predict_global(data)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ============================================================
    # Evaluation + Plotting (unchanged)
    # ============================================================
    def evaluate_and_plot(self, loader=None, garch_forecaster=None, feature_name="realized_vol", garch_rolling=False):
        if self.method == "lstm":
            preds, actuals = evaluate_lstm(self.model, loader or self._val_loader, device=self.device)
            if preds.ndim > 1:
                y_true = actuals[:, 0]
                y_pred_lstm = preds[:, 0]
            else:
                y_true, y_pred_lstm = actuals, preds

            lstm_rmse = sqrt(mean_squared_error(y_true, y_pred_lstm))
            print(f"LSTM RMSE: {lstm_rmse:.6f}")

            if garch_forecaster is not None:
                if garch_rolling:
                    garch_pred = garch_forecaster.model.predict(
                        horizon=1, rolling=True, returns=garch_forecaster.model._fitted_returns
                    )
                    garch_pred = garch_pred[-len(y_true):]
                else:
                    garch_pred = np.repeat(garch_forecaster.model.predict(horizon=1), len(y_true))

                garch_rmse = sqrt(mean_squared_error(y_true, garch_pred))
                print(f"{garch_forecaster.model_type.upper()} RMSE: {garch_rmse:.6f}")
            else:
                garch_pred = None

            plt.figure(figsize=(12, 5))
            plt.plot(y_true, label="Actual")
            plt.plot(y_pred_lstm, label="LSTM Forecast")
            if garch_pred is not None:
                plt.plot(garch_pred, label=f"{garch_forecaster.model_type.upper()} Forecast")
            plt.title("Forecast vs Actual")
            plt.legend()
            plt.show()

            return y_pred_lstm, y_true, garch_pred
        else:
            raise RuntimeError("Evaluation supported only for LSTM forecaster right now.")
