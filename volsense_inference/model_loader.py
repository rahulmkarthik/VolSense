"""
Universal model loader for VolSense.

Supports:
  • *_full.pkl        → pickled model object (preferred)
  • *_bundle.pkl      → portable dict with state_dict + config + assets
  • .pth + .meta.json → raw fallback
Automatically imports and rebuilds any class using its recorded module path.
"""

import os
import json
import pickle
import importlib
import torch
from typing import Any, Dict, Tuple, Optional
import sys

sys.modules["volsense_pkg"] = __import__("volsense_core")


def _abs_repo_path(relative_dir: str) -> str:
    """
    Resolve a path relative to the repository root into an absolute path.

    If relative_dir is already absolute, it is returned unchanged. Otherwise,
    the path is resolved relative to the volsense_inference package root.

    :param relative_dir: Relative directory or file path to resolve.
    :type relative_dir: str
    :return: Absolute path corresponding to the provided relative path.
    :rtype: str
    """
    if os.path.isabs(relative_dir):
        return relative_dir
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, relative_dir)


def _import_class(module_path: str, class_name: str):
    """
    Dynamically import a class given its module path and class name.

    :param module_path: Dotted module path (e.g., 'volsense_core.models.global_vol_forecaster').
    :type module_path: str
    :param class_name: Name of the class to retrieve from the module.
    :type class_name: str
    :raises ModuleNotFoundError: If the module cannot be imported.
    :raises AttributeError: If the class name does not exist in the module.
    :return: Imported class object.
    :rtype: type
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def load_model(
    model_version: str = "v6a",
    checkpoints_dir: str = "models",
) -> Tuple[
    Any,
    Dict[str, Any],
    Optional[Dict[str, Any]],
    Optional[Dict[str, int]],
    Optional[list],
]:
    """
    Load a trained VolSense model and assets from disk.

    Search order:
      1) '<base>_full.pkl'   → pickled model object, optional '<base>.meta.json' for meta.
      2) '<base>_bundle.pkl' → dict with {'module_path','arch','state_dict','config','scalers','ticker_to_id','features'}.
      3) '<base>.pth' + '<base>.meta.json' → raw state_dict with metadata for dynamic reconstruction.

    Returns a tuple of (model, meta, scalers, ticker_to_id, features). On bundle/raw paths,
    attempts dynamic import and reconstruction using recorded module/class info; falls back
    to torch.load(path_pth) if dynamic rebuild fails.

    :param model_version: Version tag identifying which checkpoint set to load.
    :type model_version: str
    :param checkpoints_dir: Directory containing saved model artifacts.
    :type checkpoints_dir: str
    :raises FileNotFoundError: If no compatible artifacts are found for the version.
    :return: Loaded model, metadata/config, optional scalers, ticker-to-id map, and feature list.
    :rtype: tuple[typing.Any, dict, dict | None, dict[str, int] | None, list | None]
    """
    ckpt_dir = _abs_repo_path(checkpoints_dir)
    base = os.path.join(ckpt_dir, f"global_vol_forecaster_multi_{model_version}")

    path_full = base + "_full.pkl"
    path_bundle = base + "_bundle.pkl"
    path_pth = base + ".pth"
    path_meta = base + ".meta.json"

    # --- 1️⃣ FULL PICKLE ---
    if os.path.exists(path_full):
        with open(path_full, "rb") as f:
            model = pickle.load(f)
        meta = {}
        if os.path.exists(path_meta):
            with open(path_meta, "r") as f:
                meta = json.load(f)
        return (
            model,
            meta,
            meta.get("scalers"),
            meta.get("ticker_to_id"),
            meta.get("features"),
        )

    # --- 2️⃣ BUNDLE PICKLE ---
    if os.path.exists(path_bundle):
        with open(path_bundle, "rb") as f:
            bundle = pickle.load(f)

        arch = bundle.get("arch", "UnknownModel")
        module_path = bundle.get("module_path", "")
        state_dict = bundle.get("state_dict")
        meta = bundle.get("config", {})
        scalers = bundle.get("scalers")
        ticker_to_id = bundle.get("ticker_to_id")
        features = bundle.get("features")
        horizons = bundle.get("horizons", [1])

        try:
            ModelClass = _import_class(module_path, arch)
            model = ModelClass(
                **{
                    k: v
                    for k, v in meta.items()
                    if isinstance(v, (int, float, bool, list, str))
                }
            )
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"⚠️ Dynamic rebuild failed ({e}); attempting fallback torch.load")
            model = torch.load(path_pth, map_location="cpu")

        return (
            model,
            meta,
            scalers,
            ticker_to_id,
            features or meta.get("extra_features"),
        )

    # --- 3️⃣ RAW STATE_DICT + META.JSON ---
    if os.path.exists(path_pth) and os.path.exists(path_meta):
        with open(path_meta, "r") as f:
            meta = json.load(f)
        state_dict = torch.load(path_pth, map_location="cpu")

        # dynamic import if meta recorded module/class
        arch = meta.get("arch") or "GlobalVolForecaster"
        module_path = (
            meta.get("module_path") or "volsense_core.models.global_vol_forecaster"
        )
        ModelClass = _import_class(module_path, arch)

        model = ModelClass(
            **{
                k: v
                for k, v in meta.get("config", {}).items()
                if isinstance(v, (int, float, bool, list, str))
            }
        )
        model.load_state_dict(state_dict)

        scalers = meta.get("scalers")
        ticker_to_id = meta.get("ticker_to_id")
        features = meta.get("features")
        return model, meta, scalers, ticker_to_id, features

    raise FileNotFoundError(
        f"No artifacts found for version '{model_version}' in {ckpt_dir}."
    )
