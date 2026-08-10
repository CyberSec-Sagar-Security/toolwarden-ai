"""Toy local tools for the Phase 2 trivial test agent. Not library code —
just fixtures to prove the interception layer captures real tool-call
traffic end to end.
"""

from __future__ import annotations

from datetime import datetime, timezone

_NOTES = {
    "toolwarden": "ToolWarden AI intercepts tool-call traffic between an agent and its tools.",
    "phase2": "Phase 2 is passthrough-mode interception: log everything, block nothing.",
}


def get_time(timezone_label: str) -> str:
    return f"Current UTC time is {datetime.now(timezone.utc).isoformat()} (requested tz: {timezone_label})"


def search_notes(query: str) -> str:
    query_lower = query.lower()
    hits = [v for k, v in _NOTES.items() if query_lower in k or query_lower in v.lower()]
    return "; ".join(hits) if hits else "no notes found"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a given timezone label.",
            "parameters": {
                "type": "object",
                "properties": {"timezone_label": {"type": "string"}},
                "required": ["timezone_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search a small local notes store for a query string.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {"get_time": get_time, "search_notes": search_notes}
