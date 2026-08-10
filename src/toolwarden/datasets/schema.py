"""Normalized shape every dataset source gets converted into before anything
downstream (feature engineering, classifier training) touches it. Keeps the
rest of the pipeline source-agnostic — see docs/datasets.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    INJECTION = "injection"
    BENIGN = "benign"


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"  # in-distribution held-out (same source as train, disjoint rows)
    HELD_OUT_NOVEL = "held_out_novel"  # structurally disjoint source, never trained on


@dataclass
class DatasetRecord:
    id: str
    text: str
    label: Label
    source: str  # e.g. "injecagent", "agentdojo"
    attack_type: str | None = None  # e.g. "direct_harm", "data_stealing"
    tool_context: str | None = None  # tool name the text is attached to, if any
    split: Split = Split.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label.value,
            "source": self.source,
            "attack_type": self.attack_type,
            "tool_context": self.tool_context,
            "split": self.split.value,
            "metadata": self.metadata,
        }
