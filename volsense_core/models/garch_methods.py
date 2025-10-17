# volsense_pkg/models/garch_methods.py
import pandas as pd
import numpy as np
from arch import arch_model


class ARCHForecaster:
    """
    General ARCH-family forecaster (supports GARCH, EGARCH, GJR-GARCH).
    """
    def __init__(self, model_type="garch", p=1, q=1, dist="normal", scale=100):
        """
        Args:
            model_type: "garch", "egarch", or "gjr"
            p, q: GARCH orders
            dist: error distribution ("normal","t","skewt")
            scale: multiply returns by this before fitting (fixes arch warnings)
        """
        self.model_type = model_type.lower()
        self.p = p
        self.q = q
        self.dist = dist
        self.scale = scale
        self.model = None
        self.res = None
        self._fitted_returns = None

    def fit(self, returns: np.ndarray):
        """
        Fit ARCH-family model to a returns series.
        """
        self._fitted_returns = np.asarray(returns)

        vol_map = {
            "garch": "GARCH",
            "egarch": "EGARCH",
            "gjr": "GJR-GARCH"
        }
        if self.model_type not in vol_map:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        model = arch_model(
            self._fitted_returns * self.scale,
            vol=vol_map[self.model_type],
            p=self.p, q=self.q,
            dist=self.dist,
            mean="Zero"
        )
        self.model = model
        self.res = model.fit(disp="off")
        return self


    def predict(self, horizon=1, rolling=False, returns=None):
        """
        Generate volatility forecasts.

        Args:
            horizon (int): forecast horizon
            rolling (bool): if True, produce rolling 1-step forecasts for each point
            returns (array): optional new returns for rolling forecasts

        Returns:
            np.ndarray of volatility forecasts
        """
        if not rolling:
            # one-shot forecast
            fcast = self.res.forecast(horizon=horizon, reindex=False)
            vol = np.sqrt(fcast.variance.iloc[-1]) / self.scale
            return vol.values
        else:
            # rolling one-step forecasts
            ret_series = np.asarray(returns) if returns is not None else self._fitted_returns
            forecasts = []
            n = len(ret_series)
            min_start = max(self.p, self.q) + 20
            for i in range(min_start, n):
                am = arch_model(
                    ret_series[:i] * self.scale,
                    vol=self.model_type.capitalize(),
                    p=self.p, q=self.q,
                    dist=self.dist, mean="Zero"
                )
                res = am.fit(disp="off")
                fcast = res.forecast(horizon=1, reindex=False)
                vol = np.sqrt(fcast.variance.iloc[-1, 0]) / self.scale
                forecasts.append(vol)
            return np.array(forecasts)


def fit_arch(returns: pd.Series, model_type="garch", dist="normal", p=1, q=1, scale=100):
    """
    Fit an ARCH-family model (GARCH, EGARCH, GJR-GARCH).
    """
    model = arch_model(
        returns * scale,
        vol=model_type.capitalize(),
        p=p, q=q,
        dist=dist,
        mean="Zero"
    )
    fitted = model.fit(disp="off")
    return fitted


def forecast_arch(fitted_model, horizon=5, scale=100):
    """
    Generate volatility forecasts from a fitted ARCH-family model.
    """
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    vol_forecast = forecast.variance.iloc[-1] ** 0.5
    return vol_forecast / scale * (252 ** 0.5)
