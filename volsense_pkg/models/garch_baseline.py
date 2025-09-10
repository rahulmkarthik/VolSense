# volsense_pkg/models/garch_baseline.py
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series, dist: str = "normal", p: int = 1, q: int = 1):
    """
    Fit a GARCH(p,q) model to returns.

    Args:
        returns (pd.Series): log returns (mean ~0)
        dist (str): error distribution ("normal", "t", "skewt")
        p (int): order for ARCH term
        q (int): order for GARCH term

    Returns:
        fitted model (arch.__future__.base.ARCHModelResult)
    """
    model = arch_model(returns * 100, vol="GARCH", p=p, q=q, dist=dist, mean="Zero")
    fitted = model.fit(disp="off")
    return fitted


def forecast_garch(fitted_model, horizon: int = 5):
    """
    Generate volatility forecasts.

    Args:
        fitted_model: fitted arch_model
        horizon (int): forecast horizon in days

    Returns:
        pd.Series of volatility forecasts (annualized %)
    """
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    vol_forecast = forecast.variance.iloc[-1] ** 0.5
    return vol_forecast / 100 * (252 ** 0.5)  # annualize back
