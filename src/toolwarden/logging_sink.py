"""Where intercepted traffic records go. Phase 2 only needs durable capture —
no classification/enforcement reads this yet (that's Phase 4/6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class LogSink(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


class JsonlFileSink:
    """Appends one JSON object per line to a log file. Creates parent dirs as needed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class InMemorySink:
    """Collects records in a list instead of touching disk — used by unit tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
