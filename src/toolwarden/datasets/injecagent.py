"""Downloads and normalizes InjecAgent (uiuc-kang-lab/InjecAgent, MIT).

Uses the four combined test_cases_{dh,ds}_{base,enhanced}.json files, not
the raw attacker/user case fragments — these already merge an attacker
instruction into a realistic tool-response string via the "Tool Response"
field, which is exactly the shape of content ToolWarden's interceptor sees
(see docs/architecture.md: inbound tool-call result). See docs/datasets.md
for the full source rationale.

Each row yields a minimal contrastive pair:
  - INJECTION: the "Tool Response" field (attacker instruction embedded)
  - BENIGN:    the same "Tool Response Template" with the placeholder
               removed (same tool response, no injected content)
This keeps both classes grounded in the benchmark's own authored text —
nothing here is model-generated or fabricated.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from toolwarden.datasets.schema import DatasetRecord, Label, Split

_RAW_BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
_TEST_CASE_FILES = {
    "dh_base": "test_cases_dh_base.json",
    "dh_enhanced": "test_cases_dh_enhanced.json",
    "ds_base": "test_cases_ds_base.json",
    "ds_enhanced": "test_cases_ds_enhanced.json",
}
_INTENT_BY_PREFIX = {"dh": "direct_harm", "ds": "data_stealing"}
_PLACEHOLDER = "<Attacker Instruction>"

# Deterministic group-holdout split: rows sharing the same attacker-tool
# combination stay entirely on one side, so near-duplicate templates for the
# same tool don't leak between train and test.
_TEST_FRACTION_PERCENT = 20


def download_raw(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _TEST_CASE_FILES.items():
        dest = raw_dir / filename
        if not dest.exists():
            urllib.request.urlretrieve(f"{_RAW_BASE}/{filename}", dest)  # noqa: S310 (fixed, known-good MIT-licensed source)
        paths[key] = dest
    return paths


def _group_split(group_key: str) -> Split:
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:4], 16) % 100
    return Split.TEST if bucket < _TEST_FRACTION_PERCENT else Split.TRAIN


def _clean_template(template: str) -> str:
    return template.replace(_PLACEHOLDER, "").strip()


def normalize(raw_dir: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []

    for key, filename in _TEST_CASE_FILES.items():
        intent = _INTENT_BY_PREFIX[key.split("_")[0]]
        rows = json.loads((raw_dir / filename).read_text(encoding="utf-8"))

        for i, row in enumerate(rows):
            tools = row.get("Attacker Tools") or []
            group_key = f"{intent}:{'|'.join(sorted(tools))}"
            split = _group_split(group_key)
            record_id_base = f"injecagent-{key}-{i}"

            records.append(
                DatasetRecord(
                    id=f"{record_id_base}-injection",
                    text=row["Tool Response"],
                    label=Label.INJECTION,
                    source="injecagent",
                    attack_type=row.get("Attack Type"),
                    tool_context=row.get("User Tool"),
                    split=split,
                    metadata={
                        "intent": intent,
                        "variant": key,
                        "attacker_tools": tools,
                    },
                )
            )
            records.append(
                DatasetRecord(
                    id=f"{record_id_base}-benign",
                    text=_clean_template(row["Tool Response Template"]),
                    label=Label.BENIGN,
                    source="injecagent",
                    attack_type=None,
                    tool_context=row.get("User Tool"),
                    split=split,
                    metadata={"intent": intent, "variant": key, "derived_from": "tool_response_template"},
                )
            )

    return records
