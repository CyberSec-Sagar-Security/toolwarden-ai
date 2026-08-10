"""Loads the Phase 3 processed dataset (datasets/processed/records.jsonl,
gitignored — run `python -m toolwarden.datasets.build` first if missing)
and provides a deterministic train/val carve-out for model-selection use
only. Test and held_out_novel are never touched here — those splits are
reserved for final reporting in evaluate.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

LABEL2ID = {"benign": 0, "injection": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_VAL_FRACTION_PERCENT = 10


class Record(TypedDict):
    id: str
    text: str
    label: str
    source: str
    attack_type: str | None
    tool_context: str | None
    split: str
    metadata: dict


def load_records(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def by_split(records: list[Record], split: str) -> list[Record]:
    return [r for r in records if r["split"] == split]


def _is_validation(record_id: str) -> bool:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return int(digest[:4], 16) % 100 < _VAL_FRACTION_PERCENT


def train_fit_val_split(records: list[Record]) -> tuple[list[Record], list[Record]]:
    """Carves a deterministic ~10% validation slice out of the train split,
    for model-selection (best-checkpoint / early-stopping) only. Never
    reported as a final metric — that's test and held_out_novel.
    """
    train_records = by_split(records, "train")
    val = [r for r in train_records if _is_validation(r["id"])]
    fit = [r for r in train_records if not _is_validation(r["id"])]
    return fit, val
