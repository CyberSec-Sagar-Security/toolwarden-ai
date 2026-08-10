"""Light-touch AgentDojo (ethz-spylab/agentdojo, MIT, pip package `agentdojo`)
extraction: static task/injection text only, no live agent simulation.

Every record here is Split.HELD_OUT_NOVEL — see docs/datasets.md. AgentDojo
is a structurally disjoint source (different paper, different attack
construction) from InjecAgent, so it's used as the held-out generalization
check rather than merged into the trainable pool. Full live simulation runs
(actually executing agent + tool pipeline under attack) are deferred to
Phase 8, where that belongs.
"""

from __future__ import annotations

from toolwarden.datasets.schema import DatasetRecord, Label, Split

_BENCHMARK_VERSION = "v1.2.2"


def normalize() -> list[DatasetRecord]:
    from agentdojo.task_suite.load_suites import get_suites

    records: list[DatasetRecord] = []
    suites = get_suites(_BENCHMARK_VERSION)

    for suite_name, suite in suites.items():
        for task_id, task in suite.user_tasks.items():
            records.append(
                DatasetRecord(
                    id=f"agentdojo-{suite_name}-{task_id}",
                    text=task.PROMPT,
                    label=Label.BENIGN,
                    source="agentdojo",
                    attack_type=None,
                    tool_context=suite_name,
                    split=Split.HELD_OUT_NOVEL,
                    metadata={
                        "benchmark_version": _BENCHMARK_VERSION,
                        "difficulty": str(task.DIFFICULTY),
                    },
                )
            )

        for task_id, task in suite.injection_tasks.items():
            records.append(
                DatasetRecord(
                    id=f"agentdojo-{suite_name}-{task_id}",
                    text=task.GOAL,
                    label=Label.INJECTION,
                    source="agentdojo",
                    attack_type=None,
                    tool_context=suite_name,
                    split=Split.HELD_OUT_NOVEL,
                    metadata={
                        "benchmark_version": _BENCHMARK_VERSION,
                        "difficulty": str(task.DIFFICULTY),
                    },
                )
            )

    return records
