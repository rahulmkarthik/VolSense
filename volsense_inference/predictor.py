import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 🔧 Helper functions
# ---------------------------------------------------------------------------

def _get_feature_list(meta: Dict[str, Any], user_feats: Optional[List[str]] = None) -> List[str]:
    if user_feats:
        return user_feats
    if "features" in meta and meta["features"]:
        return meta["features"]
    if "extra_features" in meta and meta["extra_features"]:
        return ["return"] + meta["extra_features"]
    return ["return"]


def _scale_features(df_t: pd.DataFrame, feats: List[str], ticker: str,
                    scalers: Optional[Dict[str, Any]]) -> pd.DataFrame:
    df_scaled = df_t.copy()
    if scalers is None:
        return df_scaled
    sc = scalers.get(ticker)
    if sc is None:
        # fallback: mean-std normalize with recent window
        df_scaled[feats] = (
            (df_scaled[feats] - df_scaled[feats].mean()) /
            (df_scaled[feats].std() + 1e-8)
        )
        return df_scaled
    try:
        df_scaled[feats] = sc.transform(df_scaled[feats].astype(float).fillna(0.0))
    except Exception:
        # if scaler incompatible, fallback to raw
        pass
    return df_scaled


def _forward(model, X: torch.Tensor, tid_tensor: Optional[torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        try:
            out = model(tid_tensor, X) if tid_tensor is not None else model(X)
        except TypeError:
            # For models that don't take ticker IDs
            out = model(X)
        return out.cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# 🎯 Core prediction logic
# ---------------------------------------------------------------------------

#TODO fix flat outputs in volatility forecsting. Likely a problem with feature parsing

def predict_single(
    model: Any,
    meta: Dict[str, Any],
    df: pd.DataFrame,
    ticker: str,
    scalers: Optional[Dict[str, Any]] = None,
    ticker_to_id: Optional[Dict[str, int]] = None,
    features: Optional[List[str]] = None,
) -> Dict[str, float]:
    feats = _get_feature_list(meta, features)
    # 🔹 Align DataFrame columns with the model’s expected features
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"⚠️ {ticker}: missing features {missing}, filling with 0.")
        for f in missing:
            df[f] = 0.0

    extra = [f for f in df.columns if f not in feats + ["date","ticker","realized_vol"]]
    if extra:
        df = df.drop(columns=extra)

    window = int(meta.get("window", meta.get("lookback", 30)))
    df_t = df[df["ticker"] == ticker].sort_values("date")

    if len(df_t) < window:
        raise ValueError(f"{ticker}: insufficient data ({len(df_t)} rows, need {window})")

    df_t = _scale_features(df_t, feats, ticker, scalers)
    X_df = df_t.iloc[-window:][feats]
    X = torch.tensor(X_df.values, dtype=torch.float32).unsqueeze(0)

    # default to 0 if ticker not in dict
    tid_tensor = None
    if ticker_to_id:
        tid = ticker_to_id.get(ticker, 0)
        tid_tensor = torch.tensor([tid], dtype=torch.long)

    yhat = _forward(model, X, tid_tensor)

    if meta.get("config", {}).get("target_col", "").endswith("_log"):
        yhat = np.exp(yhat)

    horizons = meta.get("horizons", [1])
    preds = {f"pred_vol_{h}": float(yhat[i]) if i < len(yhat) else np.nan
             for i, h in enumerate(horizons)}

    return {"ticker": ticker, **preds}


def predict_batch(
    model: Any,
    meta: Dict[str, Any],
    df: pd.DataFrame,
    tickers: List[str],
    scalers: Optional[Dict[str, Any]] = None,
    ticker_to_id: Optional[Dict[str, int]] = None,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    for t in tqdm(tickers, desc="Forecasting"):
        try:
            rows.append(predict_single(model, meta, df, t, scalers, ticker_to_id, features))
        except Exception as e:
            print(f"⚠️ {t}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 📊 Post-processing utilities
# ---------------------------------------------------------------------------

def attach_realized(df_preds: pd.DataFrame, df_recent: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the latest realized volatility to each forecast row,
    compute directional change, and reorder columns for readability.
    """
    if "realized_vol" not in df_recent.columns:
        return df_preds

    # Get most recent realized vol per ticker
    last = (
        df_recent.sort_values("date")
        .groupby("ticker")["realized_vol"]
        .last()
        .rename("realized_vol")
        .reset_index()
    )

    out = df_preds.merge(last, on="ticker", how="left")

    # Compute diffs
    if "pred_vol_1" in out.columns:
        out["vol_diff"] = out["pred_vol_1"] - out["realized_vol"]
        out["vol_direction"] = np.sign(out["vol_diff"])

    # Reorder columns so realized_vol is first
    base_cols = ["ticker", "realized_vol"]
    pred_cols = [c for c in out.columns if c.startswith("pred_vol_")]
    extra_cols = [c for c in ["vol_diff", "vol_direction"] if c in out.columns]
    out = out[base_cols + pred_cols + extra_cols]

    return out