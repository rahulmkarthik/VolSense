# volsense_pkg/utils/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

__all__ = ["evaluate_forecasts"]


def evaluate_forecasts(preds: np.ndarray, actuals: np.ndarray, title: str = "Forecast Evaluation"):
    """
    Compute metrics + plot forecast vs actual.

    Parameters
    ----------
    preds : np.ndarray
        Forecasted values from the model.
    actuals : np.ndarray
        True observed values.
    title : str
        Title prefix for plots.

    Returns
    -------
    metrics : dict
        Dictionary with RMSE, MAE, and Bias.
    """

    # Flatten just in case
    preds = np.asarray(preds).flatten()
    actuals = np.asarray(actuals).flatten()

    # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    bias = np.mean(preds - actuals)
    metrics = {"rmse": rmse, "mae": mae, "bias": bias}

    # --- Line Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label="Actual Vol", color='blue')
    plt.plot(preds, label="Predicted Vol", color='orange')
    plt.title(f"{title}: Forecast vs Actual")
    plt.xlabel("Sample Index")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Scatter Plot ---
    plt.figure(figsize=(5, 5))
    plt.scatter(actuals, preds, alpha=0.6, color='purple')
    max_val = max(np.max(actuals), np.max(preds))
    min_val = min(np.min(actuals), np.min(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # y=x line
    plt.title(f"{title}: Predicted vs Actual Scatter")
    plt.xlabel("Actual Vol")
    plt.ylabel("Predicted Vol")
    plt.tight_layout()
    plt.show()

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:.4f}")
    return metrics