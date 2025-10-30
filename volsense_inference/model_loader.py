# ============================================================
# volsense_inference/model_loader_refactored.py
# Unified VolSense Loader
# Supports:
#   - *.full.pkl        → legacy pickled model
#   - *_bundle.pkl      → combined dict
#   - .meta.json + .pth → modern portable
# ============================================================

import os
import json
import pickle
import importlib
import torch


# ------------------------------------------------------------
# 🔹 Resolve checkpoint path
# ------------------------------------------------------------
def _resolve_ckpt_path(model_version: str, checkpoints_dir: str) -> str:
    """Return absolute path to model checkpoint base (no suffix)."""
    cwd = os.getcwd()
    local_ckpt = os.path.abspath(os.path.join(cwd, checkpoints_dir))
    parent_ckpt = os.path.abspath(os.path.join(cwd, "..", checkpoints_dir))

    # Prefer the parent-level models dir if both exist
    if os.path.exists(parent_ckpt):
        ckpt_dir = parent_ckpt
    elif os.path.exists(local_ckpt):
        ckpt_dir = local_ckpt
    else:
        raise FileNotFoundError(f"Could not locate checkpoint directory: {checkpoints_dir}")

    return os.path.join(ckpt_dir, model_version)


def _import_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ------------------------------------------------------------
# 🔹 1️⃣ Legacy full.pkl loader (v109)
# ------------------------------------------------------------
def _load_full_pickle(path: str, device: str):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # Case A: dict bundle
    if isinstance(bundle, dict):
        model = bundle["model"]
        meta = bundle.get("meta", {}) or {}
        ticker_to_id = bundle.get("ticker_to_id", {}) or {}
        features = bundle.get("features", []) or []
    else:
        # Case B: direct model object
        model = bundle
        meta = {}
        if hasattr(model, "window"):
            meta["window"] = int(getattr(model, "window"))
        if hasattr(model, "horizons"):
            meta["horizons"] = list(getattr(model, "horizons"))
        if hasattr(model, "config") and isinstance(model.config, dict):
            meta["config"] = {
                k: v for k, v in model.config.items()
                if isinstance(v, (int, float, bool, str, list))
            }
        ticker_to_id = getattr(model, "ticker_to_id", {}) or {}
        features = (
            getattr(model, "features", None)
            or getattr(model, "feature_names_", None)
            or []
        )

    # Fallbacks
    if "window" not in meta:
        meta["window"] = 40
    if "horizons" not in meta:
        meta["horizons"] = [1]

    # Enrich from sidecar meta.json (single source of truth)
    sidecar = path.replace(".full.pkl", ".meta.json")
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r") as f:
                side_meta = json.load(f)
            meta.update(side_meta)
            features = side_meta.get("features", features)
            ticker_to_id = side_meta.get("ticker_to_id", ticker_to_id)
            if hasattr(model, "input_size"):
                model.input_size = len(features)
        except Exception:
            pass

    model.to(device).eval()
    return model, meta, None, ticker_to_id, features


# ------------------------------------------------------------
# 🔹 2️⃣ Bundle pickle loader (combined dict)
# ------------------------------------------------------------
def _load_bundle_pickle(path: str, device: str):
    """
    Load model bundle saved via checkpoint_utils.save_checkpoint().

    Fully reconstructs the architecture using stored meta['arch_params'],
    and applies safe defaults for any missing parameters. Ensures that
    model.input_size aligns with the stored feature list.
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # --- Extract bundle metadata ---
    meta = bundle.get("meta", {})
    module_path = bundle.get("module_path", meta.get("module_path", "volsense_core.models.global_vol_forecaster"))
    arch = bundle.get("arch", meta.get("arch", "GlobalVolForecaster"))
    arch_params = dict(bundle.get("arch_params", meta.get("arch_params", {})))

    # --- Fallback defaults for safety ---
    ticker_to_id = bundle.get("ticker_to_id", meta.get("ticker_to_id", {}))
    n_tickers = len(ticker_to_id) if ticker_to_id else 1
    horizons = meta.get("horizons", [1])
    n_horizons = len(horizons) if isinstance(horizons, list) else int(horizons)
    features = bundle.get("features", meta.get("features", []))
    if not features:
        features = ["return"] + meta.get("extra_features", [])

    defaults = {
        "n_tickers": n_tickers,
        "window": meta.get("window", 40),
        "n_horizons": n_horizons,
        "emb_dim": 16,
        "hidden_dim": 160,
        "num_layers": 3,
        "dropout": 0.3,
        "input_size": len(features),
        "attention": True,
        "residual_head": True,
        "use_layernorm": True,
        "separate_heads": True,
        "feat_dropout_p": 0.1,
        "variational_dropout_p": 0.1,
    }

    # Fill missing keys in arch_params from defaults
    for k, v in defaults.items():
        arch_params.setdefault(k, v)

    # --- Build model ---
    ModelClass = _import_class(module_path, arch)
    model = ModelClass(**arch_params)
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.to(device).eval()

    # --- Final consistency fix ---
    model.input_size = len(features)

    return model, meta, None, ticker_to_id, features



# ------------------------------------------------------------
# 🔹 3️⃣ Meta + PTH loader (modern portable)
# ------------------------------------------------------------
def _load_meta_pth(base: str, device: str):
    """
    Load model from portable (.meta.json + .pth) format.

    Uses meta['arch_params'] when available and fills any missing
    constructor arguments with defaults. Ensures input_size matches
    the feature count before returning.
    """
    meta_path = base + ".meta.json"
    pth_path = base + ".pth"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    module_path = meta.get("module_path", "volsense_core.models.global_vol_forecaster")
    arch = meta.get("arch", "GlobalVolForecaster")
    arch_params = dict(meta.get("arch_params", {}))

    # --- Fallback defaults for safety ---
    ticker_to_id = meta.get("ticker_to_id", {})
    n_tickers = len(ticker_to_id) if ticker_to_id else 1
    horizons = meta.get("horizons", [1])
    n_horizons = len(horizons) if isinstance(horizons, list) else int(horizons)
    features = meta.get("features", [])
    if not features:
        features = ["return"] + meta.get("extra_features", [])

    defaults = {
        "n_tickers": n_tickers,
        "window": meta.get("window", 40),
        "n_horizons": n_horizons,
        "emb_dim": 16,
        "hidden_dim": 160,
        "num_layers": 3,
        "dropout": 0.3,
        "input_size": len(features),
        "attention": True,
        "residual_head": True,
        "use_layernorm": True,
        "separate_heads": True,
        "feat_dropout_p": 0.1,
        "variational_dropout_p": 0.1,
    }

    # Fill missing keys from defaults
    for k, v in defaults.items():
        arch_params.setdefault(k, v)

    # --- Build and load model ---
    ModelClass = _import_class(module_path, arch)
    model = ModelClass(**arch_params)
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # --- Final consistency fix ---
    model.input_size = len(features)

    return model, meta, None, ticker_to_id, features



# ------------------------------------------------------------
# 🔹 Entrypoint
# ------------------------------------------------------------
def load_model(model_version: str, checkpoints_dir: str = "models", device: str = "cpu"):
    """
    Universal VolSense model loader.
    Automatically detects and loads:
        - .full.pkl
        - _bundle.pkl
        - .meta.json + .pth
    """
    base = _resolve_ckpt_path(model_version, checkpoints_dir)

    # full.pkl (legacy)
    full_path = base + ".full.pkl"
    if os.path.exists(full_path):
        return _load_full_pickle(full_path, device)

    # bundle.pkl (combined)
    bundle_path = base + "_bundle.pkl"
    if os.path.exists(bundle_path):
        return _load_bundle_pickle(bundle_path, device)

    # meta.json + pth (portable)
    meta_path = base + ".meta.json"
    pth_path = base + ".pth"
    if os.path.exists(meta_path) and os.path.exists(pth_path):
        return _load_meta_pth(base, device)

    raise FileNotFoundError(f"No valid checkpoint found for {model_version} in {checkpoints_dir}")