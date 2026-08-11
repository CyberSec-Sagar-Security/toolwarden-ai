"""ToolWarden AI's public interface. Import from here rather than reaching
into internal submodules directly — everything under `toolwarden.*`
besides what's re-exported here stays free to change across phases; this
file is the surface meant to stay stable for a pip-installed consumer.

Importing this module is intentionally cheap: none of Phase 2-9's pieces
import torch/transformers/lightgbm at module level (see
toolwarden.classifier.evaluate — those libraries are only imported inside
function bodies, called when a Classifier is actually instantiated), so
`import toolwarden` does not load any model weights or heavy ML libraries
as a side effect. It does set the HF cache env vars (config.configure_hf_cache_env(),
triggered transitively through the Classifier import below) so weight
downloads land under TOOLWARDEN_MODEL_DIR regardless of import order.

The MCP adapter (GuardedMCPServer) is deliberately NOT re-exported here:
it depends on the optional `mcp` extra (`pip install toolwarden-ai[mcp]`),
and importing it unconditionally would make `mcp` a hard dependency for
every consumer, including ones only using the direct-API adapter. Import
it explicitly from `toolwarden.mcp_proxy.server` if you're using MCP.
"""

from __future__ import annotations

__version__ = "0.1.0"

from toolwarden.demo.classify import DETECTOR_MODES, Classifier
from toolwarden.demo.guarded_loop import GuardedOpenAIToolLoop, GuardedRunResult, OnHold, ToolCallEvent
from toolwarden.enforcement.approval_queue import ApprovalDecision, ApprovalQueue
from toolwarden.enforcement.engine import EnforcementEngine, EnforcementResult
from toolwarden.enforcement.policy import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_HOLD_THRESHOLD,
    Decision,
    Direction,
    PolicyEngine,
)
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import InMemorySink, JsonlFileSink, LogSink
from toolwarden.models import ToolCallRequest, ToolCallResult

__all__ = [
    "__version__",
    "Classifier",
    "DETECTOR_MODES",
    "GuardedOpenAIToolLoop",
    "GuardedRunResult",
    "OnHold",
    "ToolCallEvent",
    "ApprovalDecision",
    "ApprovalQueue",
    "EnforcementEngine",
    "EnforcementResult",
    "DEFAULT_BLOCK_THRESHOLD",
    "DEFAULT_HOLD_THRESHOLD",
    "Decision",
    "Direction",
    "PolicyEngine",
    "Interceptor",
    "InMemorySink",
    "JsonlFileSink",
    "LogSink",
    "ToolCallRequest",
    "ToolCallResult",
]
