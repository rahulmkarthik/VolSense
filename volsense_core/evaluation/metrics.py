# volsense_pkg/utils/metrics.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


# =====================================================
# --- Existing Low-Level GARCH Utilities (KEEP THESE) ---
# =====================================================
def rolling_garch_forecast(model, returns, horizon=1):
    """
    Generate rolling forecasts using a fitted arch_model.
    Each step fits on past data and forecasts 1-step ahead.
    """
    forecasts = []
    n = len(returns)
    for i in range(horizon, n):
        window_data = returns[:i]
        res = model.fit(disp="off")
        fcast = res.forecast(horizon=1, reindex=False)
        vol = np.sqrt(fcast.variance.iloc[-1, 0])
        forecasts.append(vol)
    return np.array(forecasts)


# =====================================================
# --- Universal Evaluation Metrics (REPLACE OLD ONES) ---
# =====================================================
def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def r2_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def acf_sum_k10(resid, m=10):
    """
    Sum of squared autocorrelations up to lag m.
    Stable alternative to Ljung–Box p-values.
    Smaller values → weaker serial correlation.
    """
    resid = np.asarray(resid)
    if len(resid) < m + 2 or np.std(resid) < 1e-8:
        return np.nan
    vals = acf(resid, nlags=m, fft=True)[1:]  # skip lag 0
    return float(np.sum(vals**2))



# =====================================================
# --- Unified Forecast Evaluation Interface ---
# =====================================================
def evaluate_forecasts(df):
    """
    Compute evaluation metrics for volatility forecasts.
    Expects df with columns ['realized_vol', 'forecast_vol'].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if not all(c in df.columns for c in ["realized_vol", "forecast_vol"]):
        raise ValueError("DataFrame must contain 'realized_vol' and 'forecast_vol' columns.")

    metrics = {
        "RMSE": rmse(df["realized_vol"], df["forecast_vol"]),
        "MAE": mae(df["realized_vol"], df["forecast_vol"]),
        "MAPE": mape(df["realized_vol"], df["forecast_vol"]),
        "R2": r2_score(df["realized_vol"], df["forecast_vol"]),
    }
    return metrics
 

# =====================================================
# --- Backward Compatibility: old evaluate_forecast() ---
# =====================================================
def evaluate_forecast(y_true, y_pred):
    """
    Legacy wrapper for evaluate_forecasts() for backward compatibility.
    """
    df = pd.DataFrame({"realized_vol": y_true, "forecast_vol": y_pred})
    return evaluate_forecasts(df)
