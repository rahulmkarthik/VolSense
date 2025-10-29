import numpy as np
import pandas as pd

EPS = 1e-6


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns (common from yfinance) into a single level.

    :param df: Input DataFrame that may contain MultiIndex columns.
    :type df: pandas.DataFrame
    :return: DataFrame with a single-level column index.
    :rtype: pandas.DataFrame
    """

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns.name = None
    return df


def compute_base_features(
    df: pd.DataFrame, eps: float = EPS, vol_lookback: int = 10
) -> pd.DataFrame:
    """
    Compute and ensure core features exist on the input DataFrame.

    Ensures and/or computes:
      - date (datetime) and ticker (defaults to 'UNK' if missing)
      - return: pct-change of Adj Close (or Close if Adj Close is absent)
      - realized_vol: rolling std of return over vol_lookback, annualized by sqrt(252)
      - realized_vol_log: log(realized_vol + eps)

    Accepts either raw OHLCV (with 'Adj Close'/'Close') or a long-form dataset that
    already contains 'return' and optionally 'realized_vol'.

    :param df: Input DataFrame with either raw OHLCV or precomputed returns/volatility.
    :type df: pandas.DataFrame
    :param eps: Numerical stability constant added before log.
    :type eps: float
    :param vol_lookback: Rolling window (in days) for realized volatility.
    :type vol_lookback: int
    :raises ValueError: If 'date' is missing and index is not a DatetimeIndex.
    :raises KeyError: If 'return' must be computed but neither 'Adj Close' nor 'Close' is present.
    :return: DataFrame with core columns computed and sorted by ['ticker','date'].
    :rtype: pandas.DataFrame
    """

    df = _flatten_cols(df.copy())

    # date column
    if "date" not in df.columns:
        # yfinance frames typically have DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        else:
            raise ValueError(
                "Input DataFrame must contain a 'date' column or a DatetimeIndex."
            )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ticker column
    if "ticker" not in df.columns:
        # single-ticker fallback
        df["ticker"] = "UNK"

    # returns
    if "return" not in df.columns:
        price_col = None
        if "Adj Close" in df.columns:
            price_col = "Adj Close"
        elif "Close" in df.columns:
            price_col = "Close"

        if price_col is None:
            raise KeyError(
                "Cannot compute 'return': expected 'Adj Close' or 'Close' when 'return' is not present."
            )

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["return"] = df.groupby("ticker")[price_col].pct_change()

    # realized volatility
    if "realized_vol" not in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["realized_vol"] = df.groupby("ticker")["return"].rolling(
            vol_lookback, min_periods=3
        ).std().reset_index(level=0, drop=True) * np.sqrt(252)

    # log target
    df["realized_vol_log"] = np.log(df["realized_vol"].astype(float) + eps)

    # tidy
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """
    Add rolling volatility features and related derivatives.

    Computes per-ticker:
      - vol_3d: 3-day mean of realized_vol
      - vol_10d: 10-day mean of realized_vol
      - vol_ratio: vol_3d / (vol_10d + eps)
      - vol_chg: vol_3d - vol_10d
      - vol_vol: 10-day std of realized_vol
      - ewma_vol_10d: EWMA of realized_vol with span=10

    :param df: Input DataFrame containing at least ['ticker','realized_vol'].
    :type df: pandas.DataFrame
    :param eps: Numerical stability constant used in vol_ratio denominator.
    :type eps: float
    :return: DataFrame with added rolling features.
    :rtype: pandas.DataFrame
    """

    g = df.groupby("ticker", group_keys=False)
    df["vol_3d"] = g["realized_vol"].apply(lambda s: s.rolling(3, min_periods=1).mean())
    df["vol_10d"] = g["realized_vol"].apply(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    df["vol_ratio"] = df["vol_3d"] / (df["vol_10d"] + eps)
    df["vol_chg"] = df["vol_3d"] - df["vol_10d"]
    df["vol_vol"] = g["realized_vol"].apply(
        lambda s: s.rolling(10, min_periods=2).std()
    )
    # long-term EWMA proxy (used by later models)
    df["ewma_vol_10d"] = g["realized_vol"].apply(
        lambda s: s.ewm(span=10, adjust=False).mean()
    )
    return df


def add_market_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """
    Add cross-sectional market stress, its lag, and short-horizon return skew.

    Computes:
      - market_stress: cross-sectional std of 'return' within each date
      - market_stress_1d_lag: per-ticker 1-day lag of market_stress
      - skew_5d: rolling 5-day skewness of returns (standardized), requires >= 3 observations

    :param df: Input DataFrame with at least ['date','ticker','return'].
    :type df: pandas.DataFrame
    :param eps: Numerical stability constant used in skewness computation.
    :type eps: float
    :return: DataFrame with added market-level features.
    :rtype: pandas.DataFrame
    """

    df["market_stress"] = df.groupby("date")["return"].transform(lambda x: x.std())
    g = df.groupby("ticker", group_keys=False)
    df["market_stress_1d_lag"] = g["market_stress"].shift(1)

    def _skew5(x):
        if len(x) < 3:
            return np.nan
        m, sd = np.mean(x), np.std(x)
        sd = sd if sd > 0 else eps
        return np.mean(((x - m) / sd) ** 3)

    df["skew_5d"] = g["return"].apply(
        lambda s: s.rolling(5, min_periods=3).apply(_skew5, raw=True)
    )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar encodings and absolute return moments.

    Computes:
      - day_of_week: normalized day index in [0,1] (Mon=0, Sun=1 via 6.0 divisor)
      - month_sin, month_cos: cyclical month encodings
      - abs_return: absolute daily return
      - ret_sq: squared daily return

    :param df: Input DataFrame containing 'date' and 'return'.
    :type df: pandas.DataFrame
    :return: DataFrame with added calendar features.
    :rtype: pandas.DataFrame
    """

    df["day_of_week"] = df["date"].dt.dayofweek / 6.0
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["abs_return"] = df["return"].abs()
    df["ret_sq"] = df["return"] ** 2
    return df


def build_features(df: pd.DataFrame, include=None, exclude=None) -> pd.DataFrame:
    """
    Build a full feature set with inclusion/exclusion controls.

    Pipeline:
      1) compute_base_features
      2) add_rolling_features
      3) add_market_features
      4) add_calendar_features
      5) Filter to core columns + requested engineered features

    :param df: Input DataFrame (raw OHLCV or long table with 'return' and 'realized_vol').
    :type df: pandas.DataFrame
    :param include: Feature names to include; if None, uses the default feature list.
    :type include: list[str] or None
    :param exclude: Feature names to exclude from the final set.
    :type exclude: list[str] or set[str] or None
    :raises ValueError: Propagated if base feature computation cannot infer dates.
    :raises KeyError: Propagated if required columns for returns are missing.
    :return: Feature-enriched DataFrame sorted by ['ticker','date'].
    :rtype: pandas.DataFrame
    """

    include = include or [
        "vol_3d",
        "vol_10d",
        "vol_ratio",
        "vol_chg",
        "vol_vol",
        "ewma_vol_10d",
        "market_stress",
        "market_stress_1d_lag",
        "skew_5d",
        "day_of_week",
        "month_sin",
        "month_cos",
        "abs_return",
        "ret_sq",
    ]
    exclude = set(exclude or [])

    df = compute_base_features(df)  # <- safe on both raw or processed
    df = add_rolling_features(df)
    df = add_market_features(df)
    df = add_calendar_features(df)

    # Filter to requested engineered features + core columns
    keep = set(include) - exclude
    core = {"date", "ticker", "return", "realized_vol", "realized_vol_log"}
    cols = [c for c in list(core) + list(keep) if c in df.columns]
    return df[cols].sort_values(["ticker", "date"]).reset_index(drop=True)
