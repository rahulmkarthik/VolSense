import os, json, torch

def load_model(model_version="v3", checkpoints_dir="models"):
    """
    Load a lightweight or production model checkpoint.
    Automatically resolves relative paths to repo root.
    """
    # If a relative path is given, resolve it relative to the package's root
    if not os.path.isabs(checkpoints_dir):
        # Find the repo root relative to this file (volsense_inference/)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        checkpoints_dir = os.path.join(repo_root, checkpoints_dir)

    base = os.path.join(checkpoints_dir, f"global_vol_forecaster_multi_{model_version}")
    ckpt_path = base + ".pth"
    meta_path = base + ".meta.json"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = torch.load(ckpt_path, map_location="cpu")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta