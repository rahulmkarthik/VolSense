# volsense_core/utils/checkpoint_utils.py
# ============================================================
# 🧩 VolSense Checkpoint Utility (Final)
# Auto-detects architecture, extracts constructor args,
# and builds reconstructible meta and bundle files.
# ============================================================

import os
import json
import pickle
import torch
import inspect
from typing import Any, Dict


# ============================================================
# 🔹 Meta Builder (auto-detects architecture)
# ============================================================

def build_meta_from_model(model: Any, cfg: Any = None, ticker_to_id=None, features=None) -> Dict:
    """
    Build a standardized, reconstructible metadata dictionary
    for any VolSense model (BaseLSTM, GlobalVolForecaster, etc.).
    """

    name = model.__class__.__name__
    module_path = model.__module__

    meta = {
    "arch": name,
    "module_path": module_path,
    "features": features or getattr(cfg, "extra_features", getattr(cfg, "features", [])),
    "extra_features": getattr(cfg, "extra_features", []),
    "horizons": getattr(cfg, "horizons", []),
    "ticker_to_id": ticker_to_id or {},
    }



    # Inspect constructor and extract matching attributes
    sig = inspect.signature(model.__init__)
    arch_params = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if hasattr(model, param_name):
            val = getattr(model, param_name)
            # Only include simple reconstructible types
            if isinstance(val, (int, float, bool, str, list, tuple, dict, type(None))):
                arch_params[param_name] = val

        # --- Model-specific augmentations ---
        if name.lower().startswith("glob"):
            extra_keys = ["emb_dim", "hidden_dim", "num_layers", "dropout",
                        "separate_heads", "use_layernorm"]
            for k in extra_keys:
                if hasattr(model, k):
                    arch_params[k] = getattr(model, k)

        elif name.lower().startswith("base"):
            extra_keys = ["input_dim", "hidden_dim", "num_layers", "dropout", "n_horizons",
                        "use_layernorm", "use_attention", "feat_dropout_p", "residual_head",
                        "output_activation"]
            for k in extra_keys:
                if hasattr(model, k):
                    arch_params[k] = getattr(model, k)

        # Remove cfg-only fields not accepted by constructors
        for bad_key in ["horizons", "window", "stride", "val_start", "target_col"]:
            arch_params.pop(bad_key, None)

    meta["arch_params"] = arch_params
    
    return meta



# ============================================================
# 🔹 Bundle Builder (model-aware)
# ============================================================

def build_bundle(model: Any, meta: Dict, cfg: Any = None) -> Dict:
    """
    Construct a portable bundle for pickle-based saving.
    Includes state_dict and meta for reconstructibility.
    """
    safe_state = model.state_dict()  # ✅ only tensors
    bundle = {
        "state_dict": safe_state,
        "meta": meta,
        "arch": meta.get("arch"),
        "module_path": meta.get("module_path"),
        "arch_params": meta.get("arch_params", {}),
        "ticker_to_id": meta.get("ticker_to_id", {}),
        "features": meta.get("features", []),
        "extra_features": meta.get("extra_features", []),
        "horizons": meta.get("horizons", []),
    }
    if cfg is not None:
        bundle["config"] = getattr(cfg, "__dict__", {})
    return bundle


# ============================================================
# 💾 Unified Saver
# ============================================================

def save_checkpoint(model: Any, cfg: Any = None, version: str = "model", save_dir: str = "models",
                    ticker_to_id=None, features=None):
    """
    Save model in all supported VolSense formats:
        - .full.pkl
        - _bundle.pkl
        - .meta.json + .pth
    Works for BaseLSTM, GlobalVolForecaster, and GARCH models.
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, version)

    # --- Build meta and bundle ---
    meta = build_meta_from_model(model, cfg, ticker_to_id=ticker_to_id, features=features)
    bundle = build_bundle(model, meta, cfg)

    # 1️⃣ full.pkl
    with open(base + ".full.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Saved {base}.full.pkl")

    # 2️⃣ bundle.pkl
    with open(base + "_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"💾 Saved {base}_bundle.pkl")

    # 3️⃣ meta.json + pth
    torch.save(model.state_dict(), base + ".pth")
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Saved {base}.pth and {base}.meta.json")

    return meta


# ============================================================
# 🔍 Optional: Meta Inspector
# ============================================================

def load_meta(model_version: str, checkpoints_dir: str = "models") -> Dict:
    """Load metadata only for quick inspection."""
    meta_path = os.path.join(checkpoints_dir, model_version + ".meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found for {model_version}")
    with open(meta_path, "r") as f:
        return json.load(f)