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
class Explanation:
    """Phase 5's explainability output (SHAP for LightGBM, attention for
    DeBERTa), attached to a score at its source so every consumer of a
    Decision (guarded_loop.py, mcp_proxy/server.py, service/app.py) gets it
    for free instead of each wiring its own call into
    toolwarden.classifier.explain. Lives here, not in classifier/ or
    enforcement/, so both can depend on the shape without a circular import.

    lightgbm_top_features is None when detector_mode="deberta_only" --
    LightGBM never runs in that mode, so there's nothing to explain.
    """

    deberta_top_tokens: list[tuple[str, float]]
    lightgbm_top_features: list[tuple[str, float]] | None

    def to_dict(self) -> dict[str, Any]:
        """JSON has no tuple type -- pairs become 2-element arrays. Shared by
        PostgresApprovalQueue (storage) and the FastAPI response model
        (service/app.py), so both serialize this the same way.
        """
        return {
            "deberta_top_tokens": [list(pair) for pair in self.deberta_top_tokens],
            "lightgbm_top_features": (
                None if self.lightgbm_top_features is None else [list(pair) for pair in self.lightgbm_top_features]
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Explanation":
        return cls(
            deberta_top_tokens=[tuple(pair) for pair in data["deberta_top_tokens"]],
            lightgbm_top_features=(
                None
                if data.get("lightgbm_top_features") is None
                else [tuple(pair) for pair in data["lightgbm_top_features"]]
            ),
        )


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
