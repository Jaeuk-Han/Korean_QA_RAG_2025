import json, os

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def ensure_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] File not found: {path}")
    return path