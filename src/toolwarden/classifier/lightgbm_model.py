"""Trains LightGBM on the Phase 3 engineered features (imperative phrasing,
zero-width unicode, base64 blobs, topic/script shift, jailbreak
signatures). Model file goes to config.LIGHTGBM_MODEL_PATH (Apps and
Models, never this repo).

Same train/val carve-out as deberta.py — val is only for LightGBM's
early-stopping, never test/held_out_novel.

Run with: python -m toolwarden.classifier.lightgbm_model
"""

from __future__ import annotations

from pathlib import Path

from toolwarden import config
from toolwarden.classifier.data import LABEL2ID, load_records, train_fit_val_split
from toolwarden.features.extractors import extract_all

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = REPO_ROOT / "datasets" / "processed" / "records.jsonl"

FEATURE_NAMES = [
    "imperative_phrasing_score",
    "zero_width_char_count",
    "base64_blob_count",
    "base64_blob_max_length",
    "script_mix_count",
    "topic_shift_score",
    "jailbreak_signature_count",
    "text_length",
]


def records_to_xy(records: list[dict]):
    import numpy as np

    x = []
    y = []
    for r in records:
        features = extract_all(r["text"])
        x.append([features[name] for name in FEATURE_NAMES])
        y.append(LABEL2ID[r["label"]])
    return np.array(x), y


def train() -> None:
    import lightgbm as lgb

    records = load_records(PROCESSED_PATH)
    fit_records, val_records = train_fit_val_split(records)
    print(f"Training LightGBM on {len(fit_records)} records, validating on {len(val_records)}")

    x_fit, y_fit = records_to_xy(fit_records)
    x_val, y_val = records_to_xy(val_records)

    fit_set = lgb.Dataset(x_fit, label=y_fit, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(x_val, label=y_val, feature_name=FEATURE_NAMES, reference=fit_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        fit_set,
        num_boost_round=200,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    config.LIGHTGBM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.LIGHTGBM_MODEL_PATH))
    print(f"Saved LightGBM model to {config.LIGHTGBM_MODEL_PATH}")
    print(f"Best iteration: {booster.best_iteration}")


if __name__ == "__main__":
    train()
