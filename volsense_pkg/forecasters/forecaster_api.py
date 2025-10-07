# volsense_pkg/forecasters/forecaster_api.py

from volsense_pkg.models.garch_methods import ARCHForecaster
from volsense_pkg.models.lstm_forecaster import (
    LSTMForecaster,
    MultiVolDataset,
    train_lstm,
    evaluate_lstm,
)
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


__all__ = [
    "GARCHForecaster",
    "LSTMForecaster",
    "MultiVolDataset",
    "train_lstm",
    "evaluate_lstm",
    "VolSenseForecaster",
]


# inside volsense_pkg/forecasters/forecaster_api.py

class VolSenseForecaster:
    def __init__(self, method="lstm", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self.window = kwargs.get("window", 15)  # ✅ Fix for LSTM compatibility
        self.horizon = kwargs.get("horizon", 1)
        self.model = None
        self._val_loader = None
        self.device = kwargs.get("device", "cpu")

        if self.method in ["garch", "egarch", "gjr"]:
            self.model = ARCHForecaster(
                model_type=self.method,
                p=kwargs.get("p", 1),
                q=kwargs.get("q", 1),
                dist=kwargs.get("dist", "normal"),
                scale=kwargs.get("scale", 100)
            )

    def fit(self, data_or_returns, epochs=5, batch_size=64):
        """
        Fit the selected forecaster.

        - For 'garch': data_or_returns is a 1D returns series.
        - For 'lstm': data_or_returns is a DataFrame with ['date','ticker','realized_vol'].
        """
        if self.method == "lstm":
            # Build dataset (with scaling)
            from volsense_pkg.models.lstm_forecaster import MultiVolDataset
            self._dataset = MultiVolDataset(data_or_returns, window=self.window, horizon=self.horizon)

            # Split train/val
            train_size = int(0.8 * len(self._dataset))
            val_size = len(self._dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(self._dataset, [train_size, val_size])

            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self._val_loader = DataLoader(val_ds, batch_size=batch_size)

            # Determine n_horizons automatically
            n_horizons = len(self.horizon) if isinstance(self.horizon, (list, tuple)) else 1
            from volsense_pkg.models.lstm_forecaster import LSTMForecaster, train_lstm

            self.model = LSTMForecaster(
                input_dim=1,
                hidden_dim=self.kwargs.get("hidden_dim", 64),
                num_layers=self.kwargs.get("num_layers", 2),
                dropout=self.kwargs.get("dropout", 0.2),
                n_horizons=n_horizons,
            )

            # Train the LSTM model on scaled data
            self.model = train_lstm(
                self.model, train_loader, self._val_loader,
                epochs=epochs,
                lr=self.kwargs.get("lr", 1e-3),
                device=self.device
            )
        elif self.method in ["garch", "egarch", "gjr"]:
            self.model.fit(data_or_returns)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self


    def predict(self, horizon=None, loader=None):
        """
        Predict using the trained forecaster.

        - For 'garch': specify horizon (int) or leave None to use self.horizon.
        - For 'lstm': specify a DataLoader or leave None to use stored val_loader.
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
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def evaluate_and_plot(self, loader=None, garch_forecaster=None, feature_name="realized_vol", garch_rolling=False):
        if self.method == "lstm":
            # --- your LSTM logic (unchanged) ---
            preds, actuals = evaluate_lstm(self.model, loader or self._val_loader, device=self.device)
            if preds.ndim > 1:
                y_true = actuals[:, 0]
                y_pred_lstm = preds[:, 0]
            else:
                y_true, y_pred_lstm = actuals, preds

            lstm_rmse = sqrt(mean_squared_error(y_true, y_pred_lstm))
            print(f"LSTM RMSE: {lstm_rmse:.6f}")

            # GARCH-family comparison
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

            # plot
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



