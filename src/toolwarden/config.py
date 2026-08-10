"""Central config for where model weights live. Per the project's storage
split, weight files never live in this repo — they live under
TOOLWARDEN_MODEL_DIR (see that folder's own README.md for layout).

Any code that touches model weights imports paths from here, never a
hardcoded relative path into the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_ROOT = Path(
    os.environ.get("TOOLWARDEN_MODEL_DIR", r"D:\CyberSecurity\Projects\Apps and Models\ToolWarden")
)

HF_CACHE_DIR = MODEL_ROOT / "huggingface_cache"
DEBERTA_CHECKPOINT_DIR = MODEL_ROOT / "classifiers" / "deberta-v3-base-toolwarden"
LIGHTGBM_MODEL_DIR = MODEL_ROOT / "classifiers" / "lightgbm"
LIGHTGBM_MODEL_PATH = LIGHTGBM_MODEL_DIR / "lightgbm_model.txt"
LLM_DIR = MODEL_ROOT / "llm"


def configure_hf_cache_env() -> None:
    """Point HuggingFace's own cache env vars at HF_CACHE_DIR. Call this
    before importing transformers/torch anywhere a model gets downloaded,
    so the download never lands in the default user-profile cache.
    """
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
