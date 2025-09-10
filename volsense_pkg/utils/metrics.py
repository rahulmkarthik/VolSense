# volsense_pkg/utils/metrics.py
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rolling_garch_forecast(
    returns: pd.Series,
    window: int = 1000,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "normal"
):
    """
    Perform a rolling window GARCH(p,q) forecast.

    Args:
        returns (pd.Series): log returns (mean ~0).
        window (int): size of rolling estimation window (in days).
        horizon (int): forecast horizon (default 1-day ahead).
        p, q (int): GARCH orders.
        dist (str): error distribution ("normal", "t", "skewt").

    Returns:
        pd.DataFrame with realized vol, forecast vol, and dates.
    """
    import numpy as np
    import pandas as pd
    from arch import arch_model

    realized_vols = []
    forecast_vols = []
    dates = []

    returns = returns.dropna()

    for i in range(window, len(returns) - horizon + 1):
        train = returns.iloc[i - window:i]

        # Fit GARCH
        model = arch_model(train * 100, vol="GARCH", p=p, q=q, mean="Zero", dist=dist)
        fitted = model.fit(disp="off")

        # Forecast volatility
        forecast = fitted.forecast(horizon=horizon, reindex=False)
        fcast_vol = forecast.variance.iloc[-1] ** 0.5
        fcast_vol = fcast_vol.iloc[horizon - 1] / 100 * np.sqrt(252)  # annualized
        forecast_vols.append(fcast_vol)

        # Realized volatility: depends on horizon
        future_returns = returns.iloc[i:i + horizon]
        if horizon == 1:
            realized = np.abs(future_returns.iloc[0]) * np.sqrt(252)
        else:
            realized = future_returns.std() * np.sqrt(252)
        realized_vols.append(realized)

        # Date for this forecast
        dates.append(returns.index[i + horizon - 1])

    return pd.DataFrame({
        "date": dates,
        "forecast_vol": forecast_vols,
        "realized_vol": realized_vols
    }).set_index("date")



def evaluate_forecasts(df: pd.DataFrame):
    """
    Compute RMSE and MAE for forecast vs realized vol.
    """
    # RMSE = sqrt(MSE)
    mse = mean_squared_error(df["realized_vol"], df["forecast_vol"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df["realized_vol"], df["forecast_vol"])
    return {"RMSE": rmse, "MAE": mae}
