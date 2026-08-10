"""Data model for intercepted tool-call traffic.

See docs/architecture.md — the interception layer inspects two directions of
traffic: outbound tool-call requests (agent -> tool) and inbound tool-call
results (tool -> agent). These two dataclasses are that traffic's shape.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ToolCallRequest:
    """An outbound tool-call request the agent wants to make, before execution."""

    tool_name: str
    arguments: dict[str, Any]
    source: str
    session_id: str
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: int = field(default_factory=_now_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "tool_call_request",
            "call_id": self.call_id,
            "session_id": self.session_id,
            "source": self.source,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class ToolCallResult:
    """The inbound result of an executed tool call."""

    call_id: str
    tool_name: str
    content: Any
    is_error: bool = False
    latency_ms: int | None = None
    timestamp_ms: int = field(default_factory=_now_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "tool_call_result",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "content": self.content,
            "is_error": self.is_error,
            "latency_ms": self.latency_ms,
            "timestamp_ms": self.timestamp_ms,
        }
