"""Phase 2 stop gate: demonstrate captured tool-call traffic against a
trivial test agent, with logs. Makes a real OpenAI API call, so it's skipped
unless OPENAI_API_KEY is set — never runs by accident in CI or a fresh clone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — this test makes a live OpenAI API call",
)


def test_trivial_agent_tool_calls_are_captured(tmp_path):
    from openai import OpenAI

    from toolwarden.adapters.openai_adapter import OpenAIToolLoop
    from toolwarden.interceptor import Interceptor
    from toolwarden.logging_sink import JsonlFileSink

    from ._demo_tools import TOOL_FUNCTIONS, TOOLS_SCHEMA

    log_path = tmp_path / "tool_traffic.jsonl"
    interceptor = Interceptor(sink=JsonlFileSink(log_path))

    loop = OpenAIToolLoop(
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        model="gpt-4o-mini",
        tools_schema=TOOLS_SCHEMA,
        tool_functions=TOOL_FUNCTIONS,
        interceptor=interceptor,
        session_id="phase2-stop-gate",
    )

    answer = loop.run(
        "Call the search_notes tool with query 'phase2', then call get_time "
        "with timezone_label 'Europe/Dublin'. Then summarize both results."
    )

    assert answer
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  # at least one request + one result captured

    import json

    records = [json.loads(line) for line in lines]
    tool_names_seen = {r["tool_name"] for r in records}
    assert "search_notes" in tool_names_seen
    assert "get_time" in tool_names_seen
    assert any(r["event"] == "tool_call_request" for r in records)
    assert any(r["event"] == "tool_call_result" for r in records)

    # Copy the captured log next to the persistent logs/ dir as Phase 2 evidence.
    persistent_log = Path(__file__).resolve().parents[2] / "logs" / "phase2_demo_traffic.jsonl"
    persistent_log.parent.mkdir(parents=True, exist_ok=True)
    persistent_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
