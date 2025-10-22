# volsense_core/models/garch_methods.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Tuple, Union, Dict, Any

from arch import arch_model


ArrayLike = Union[np.ndarray, pd.Series, Iterable[float]]


# ============================================================
# 🧩 Configuration
# ============================================================
@dataclass
class GARCHConfig:
    """
    Configuration for ARCH-family forecasters.

    Parameters
    ----------
    model : {"garch", "egarch", "gjr"}
        GARCH = standard GARCH(p,q)
        EGARCH = exponential GARCH(p,q)
        GJR    = threshold (a.k.a. GJR/TGARCH) via vol="GARCH" and o>0
    p : int
        ARCH order.
    q : int
        GARCH order.
    o : int
        Asymmetry order (used for GJR when > 0). Ignored for EGARCH.
    dist : {"normal", "t", "skewt", "ged"}
        Innovation distribution.
    mean : {"Zero", "Constant"}
        Mean model. For daily returns, "Zero" is common.
    scale : float
        Multiply returns by this factor before fitting to stabilize
        numerical optimization (e.g., 100 for percent returns).
    annualize : bool
        If True, convert forecasted daily sigma to annualized vol
        by multiplying by sqrt(252).
    fit_kwargs : dict
        Extra kwargs passed to .fit(disp=..., update_freq=...) of arch.
    """
    model: str = "garch"
    p: int = 1
    q: int = 1
    o: int = 0
    dist: str = "normal"
    mean: str = "Zero"
    scale: float = 100.0
    annualize: bool = True
    fit_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        self.model = str(self.model).lower()
        if self.model not in {"garch", "egarch", "gjr"}:
            raise ValueError("model must be one of {'garch','egarch','gjr'}")
        if self.fit_kwargs is None:
            # quiet + deterministic defaults
            self.fit_kwargs = dict(disp="off", update_freq=0)


# ============================================================
# 🧠 Forecaster
# ============================================================
class ARCHForecaster:
    """
    Thin, safe wrapper around `arch.arch_model` supporting GARCH, EGARCH, and GJR.

    Usage
    -----
    >>> cfg = GARCHConfig(model="gjr", p=1, o=1, q=1, dist="t")
    >>> f = ARCHForecaster(cfg).fit(ret)          # ret: pd.Series or np.array
    >>> sigma_next = f.predict(horizon=5)         # np.ndarray length 5 (annualized by default)
    >>> roll = f.rolling_forecast(ret)            # DataFrame with one-step-ahead forecasts
    """

    def __init__(self, config: Optional[GARCHConfig] = None, **kwargs):
        # Allow dict-like construction via kwargs
        self.cfg = config or GARCHConfig(**kwargs)
        self._fitted_ret_scaled: Optional[np.ndarray] = None
        self._result = None

    # --------------- internal helpers ----------------
    def _to_series(self, x: ArrayLike) -> Tuple[np.ndarray, Optional[pd.Index]]:
        """Convert input returns to a clean numpy array, keep index if Series."""
        if isinstance(x, pd.Series):
            vals = x.dropna().astype(float).values
            idx = x.dropna().index
            return vals, idx
        x = np.asarray(list(x), dtype=float)
        x = x[~np.isnan(x)]
        return x, None

    def _build_model(self, y_scaled: np.ndarray):
        """Construct the underlying arch model based on config."""
        if self.cfg.model == "egarch":
            vol = "EGARCH"
            p, q, o = self.cfg.p, self.cfg.q, 0
        elif self.cfg.model == "gjr":
            # GJR/TGARCH is specified in arch as GARCH with o > 0
            vol = "GARCH"
            p, q, o = self.cfg.p, self.cfg.q, max(1, self.cfg.o or 1)
        else:
            vol = "GARCH"
            p, q, o = self.cfg.p, self.cfg.q, 0

        return arch_model(
            y_scaled,
            vol=vol,
            p=p, o=o, q=q,
            dist=self.cfg.dist,
            mean=self.cfg.mean
        )

    def _postprocess_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Undo scaling and annualize if requested."""
        sigma = sigma / self.cfg.scale
        if self.cfg.annualize:
            sigma = sigma * np.sqrt(252.0)
        return sigma

    # --------------- public API ----------------
    def fit(self, returns: ArrayLike) -> "ARCHForecaster":
        """
        Fit the selected ARCH-family model to a return series.

        Parameters
        ----------
        returns : array-like
            Daily simple returns (e.g., price.pct_change()).

        Returns
        -------
        self
        """
        y, _ = self._to_series(returns)
        y_scaled = y * self.cfg.scale
        self._fitted_ret_scaled = y_scaled

        model = self._build_model(y_scaled)
        self._result = model.fit(**self.cfg.fit_kwargs)
        return self

    @property
    def result(self):
        """Return the underlying `ARCHModelResult` after fit()."""
        if self._result is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self._result

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        k-step-ahead volatility forecast.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Shape (horizon,), sigma forecasts on the chosen scale
            (annualized if cfg.annualize=True).
        """
        res = self.result

        # EGARCH supports only analytic 1-step; use simulation for multi-step
        if self.cfg.model == "egarch" and horizon > 1:
            f = res.forecast(horizon=horizon, method="simulation", reindex=False, simulations=500)
        else:
            f = res.forecast(horizon=horizon, reindex=False)

        sigma = np.sqrt(np.asarray(f.variance.iloc[-1].values, dtype=float))
        return self._postprocess_sigma(sigma)


    def rolling_forecast(
        self,
        returns: ArrayLike,
        start: Optional[int] = None,
        min_obs: Optional[int] = None,
        refit_every: int = 1,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Rolling one-step-ahead forecasts using an expanding window.

        Parameters
        ----------
        returns : array-like
            Daily simple returns (Series or array). If a Series is provided,
            the returned forecasts will be a Series indexed by the forecast dates.
        start : int, optional
            First index to start forecasting (default: max(p+q+o, 30)).
        min_obs : int, optional
            Minimum number of observations before the first fit (overrides `start`).
        refit_every : int
            Refit frequency in steps (1 = refit each step). Use >1 for speed.

        Returns
        -------
        pandas.Series or np.ndarray
            One-step-ahead sigma forecasts aligned with the input (annualized if configured).
        """
        y, idx = self._to_series(returns)
        n = len(y)
        if n < 50:
            raise ValueError("Not enough observations for rolling forecast (need ~50+).")

        warm = max(self.cfg.p + self.cfg.q + max(0, self.cfg.o), 30)
        if min_obs is not None:
            warm = max(warm, int(min_obs))
        if start is not None:
            warm = max(warm, int(start))

        sigmas = np.full(n, np.nan, dtype=float)
        last_res = None

        i = warm
        while i < n:
            # Refit on expanding window y[:i]
            if (i == warm) or ((i - warm) % refit_every == 0) or (last_res is None):
                m = self._build_model(y[:i] * self.cfg.scale)
                last_res = m.fit(**self.cfg.fit_kwargs)

            f = last_res.forecast(horizon=1, reindex=False)
            sigma1 = np.sqrt(float(f.variance.iloc[-1, 0]))
            sigmas[i] = sigma1
            i += 1

        sigmas = self._postprocess_sigma(sigmas)
        if idx is not None:
            # forecast for time t uses info up to t-1 → align to original index
            out = pd.Series(sigmas, index=idx, name="sigma_forecast")
            return out
        return sigmas

    # convenience
    def get_config(self) -> Dict[str, Any]:
        """Return a dict copy of the current configuration."""
        return asdict(self.cfg)


# ============================================================
# 🔧 Convenience top-level helpers (backward-friendly)
# ============================================================
def fit_arch(
    returns: ArrayLike,
    model_type: str = "garch",
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: str = "Zero",
    scale: float = 100.0,
    annualize: bool = True,
    **fit_kwargs,
):
    """
    Quick-fit a GARCH/EGARCH/GJR model and return the fitted result.
    This mirrors your earlier helper but fixes GJR handling.
    """
    cfg = GARCHConfig(
        model=model_type, p=p, q=q, o=o,
        dist=dist, mean=mean, scale=scale,
        annualize=annualize, fit_kwargs=fit_kwargs or dict(disp="off", update_freq=0)
    )
    return ARCHForecaster(cfg).fit(returns).result


def forecast_arch(fitted_model, horizon: int = 1, scale: float = 100.0, annualize: bool = True):
    """
    Generate volatility forecasts from a fitted ARCHModelResult.

    Returns a numpy array of length `horizon`. Kept for compatibility with
    older code paths that had a separate `forecast_arch(...)`.
    """
    f = fitted_model.forecast(horizon=horizon, reindex=False)
    sigma = np.sqrt(np.asarray(f.variance.iloc[-1].values, dtype=float)) / scale
    if annualize:
        sigma = sigma * np.sqrt(252.0)
    return sigma

# ============================================================
# 🔄 Unified Output Standardizer
# ============================================================
def standardize_outputs(
    dates,
    tickers,
    forecast_vols,
    realized_vols=None,
    model_name="UnknownModel",
    horizons=None,
):
    """
    Standardizes model outputs for evaluation/backtesting.

    Parameters
    ----------
    dates : list or array
        Forecast or validation dates.
    tickers : list or array
        Corresponding tickers.
    forecast_vols : np.ndarray
        Model-predicted volatilities, shape (N, H) or (N,).
    realized_vols : np.ndarray or list, optional
        True realized volatilities (same shape as forecast_vols).
    model_name : str
        Model identifier, e.g. 'BaseLSTM', 'GlobalVolForecaster', 'ARCHForecaster'.
    horizons : list[int] or None
        Forecast horizons, e.g. [1,5,10]. Defaults to range(H) if not provided.

    Returns
    -------
    pd.DataFrame
        Columns: ['date','ticker','horizon','forecast_vol','realized_vol','model']
    """
    import numpy as np
    import pandas as pd

    dates = np.asarray(dates)
    tickers = np.asarray(tickers)
    forecast_vols = np.atleast_2d(forecast_vols)
    n, h = forecast_vols.shape
    if realized_vols is None:
        realized_vols = np.full_like(forecast_vols, np.nan)
    else:
        realized_vols = np.atleast_2d(realized_vols)

    if horizons is None:
        horizons = list(range(1, h + 1))

    records = []
    for i in range(n):
        for j, horizon in enumerate(horizons):
            records.append(
                dict(
                    date=pd.to_datetime(dates[i]),
                    ticker=tickers[i],
                    horizon=horizon,
                    forecast_vol=forecast_vols[i, j],
                    realized_vol=realized_vols[i, j],
                    model=model_name,
                )
            )
    return pd.DataFrame.from_records(records)
