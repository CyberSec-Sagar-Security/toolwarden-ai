"""Phase 10: the same Phase 2-6 enforcement pieces Phase 9 wired into an
OpenAI tool-calling loop, wired instead into an MCP server's single tool-call
interception point. Confirms the detection/enforcement behavior isn't an
artifact of the direct-API adapter — it holds through the second locked
integration path (build spec: direct API function-calling + MCP).

MCP's `on_call_tool` handler is the interception point: unlike the OpenAI
loop, which sees requests and results as two separate steps inside its own
`for tool_call in ...` loop, one MCP tool call is one round trip, so this
handler does the request-side check, execution, and result-side check all
in one call before returning `CallToolResult`. Same request -> classify ->
enforce -> (execute) -> classify -> enforce -> respond shape as
guarded_loop.py, just collapsed into a single async function because MCP's
protocol collapses request and result into one exchange.

Reuses, does not reimplement: Classifier (Phase 4/8's trained ensemble),
PolicyEngine/EnforcementEngine/ApprovalQueue (Phase 6), Interceptor
(Phase 2), and guarded_loop.py's own constants (QUARANTINE_MESSAGE,
BLOCK_MESSAGE_TEMPLATE, ToolCallEvent, OnHold) so the two adapters share one
definition of what "blocked" and "quarantined" mean instead of drifting.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import mcp.types as types
from mcp.server.lowlevel import Server

from toolwarden.demo.classify import Classifier
from toolwarden.demo.guarded_loop import (
    BLOCK_MESSAGE_TEMPLATE,
    QUARANTINE_MESSAGE,
    OnHold,
    ToolCallEvent,
)
from toolwarden.enforcement.approval_queue import ApprovalQueue
from toolwarden.enforcement.engine import EnforcementEngine
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import LogSink
from toolwarden.models import Explanation, ToolCallRequest, ToolCallResult


def _to_mcp_tool(schema_entry: dict[str, Any]) -> types.Tool:
    """Converts one entry of an OpenAI-shaped tools_schema (see demo/tools.py)
    into an MCP Tool. Reuses the same schema definitions for both adapters
    rather than maintaining the tool contract twice.
    """
    fn = schema_entry["function"]
    return types.Tool(name=fn["name"], description=fn.get("description", ""), input_schema=fn["parameters"])


class GuardedMCPServer:
    def __init__(
        self,
        name: str,
        tools_schema: list[dict[str, Any]],
        tool_functions: dict[str, Callable[..., Any]],
        interceptor: Interceptor,
        classifier: Classifier,
        policy: PolicyEngine,
        approval_sink: LogSink,
        on_hold: OnHold,
        session_id: str | None = None,
    ) -> None:
        self.tool_functions = tool_functions
        self.interceptor = interceptor
        self.classifier = classifier
        self.engine = EnforcementEngine(policy=policy, approval_queue=ApprovalQueue(sink=approval_sink))
        self.on_hold = on_hold
        self.session_id = session_id or str(uuid.uuid4())
        self.events: list[ToolCallEvent] = []

        self._mcp_tools = [_to_mcp_tool(entry) for entry in tools_schema]
        self.server: Server[Any] = Server(name, on_list_tools=self._on_list_tools, on_call_tool=self._on_call_tool)

    async def _on_list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self._mcp_tools)

    async def _on_call_tool(self, ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = params.arguments or {}
        call_id = str(uuid.uuid4())

        request = self.interceptor.intercept_request(
            ToolCallRequest(
                tool_name=params.name,
                arguments=arguments,
                source="mcp",
                session_id=self.session_id,
                call_id=call_id,
            )
        )

        request_score, request_explanation = self.classifier.score_and_explain(json.dumps(arguments))
        request_decision = self._enforce(
            Direction.REQUEST, request.to_dict(), request_score, request_explanation, f"request: {request.tool_name}"
        )

        if request_decision is Decision.BLOCK:
            # Mirrors guarded_loop.py: no real tool output exists to inspect
            # here, so this is our own system-generated notice, not
            # attacker-influenced content — logged, but not re-classified.
            content = BLOCK_MESSAGE_TEMPLATE.format(tool_name=request.tool_name, reason="blocked pre-execution")
            self.interceptor.intercept_result(
                ToolCallResult(call_id=call_id, tool_name=request.tool_name, content=content, is_error=True)
            )
            return types.CallToolResult(content=[types.TextContent(text=content)], is_error=True)

        content, is_error = self._execute(request.tool_name, arguments)
        result = self.interceptor.intercept_result(
            ToolCallResult(call_id=call_id, tool_name=request.tool_name, content=content, is_error=is_error)
        )

        result_score, result_explanation = self.classifier.score_and_explain(str(result.content))
        result_decision = self._enforce(
            Direction.RESULT, result.to_dict(), result_score, result_explanation, f"result: {request.tool_name}"
        )

        if result_decision is Decision.QUARANTINE:
            return types.CallToolResult(content=[types.TextContent(text=QUARANTINE_MESSAGE)])

        text = result.content if isinstance(result.content, str) else json.dumps(result.content)
        return types.CallToolResult(content=[types.TextContent(text=text)], is_error=is_error)

    def _execute(self, tool_name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        func = self.tool_functions.get(tool_name)
        if func is None:
            return f"unknown tool: {tool_name}", True
        try:
            return func(**arguments), False
        except Exception as exc:  # tool failure surfaces as a result, not a crash
            return str(exc), True

    def _enforce(
        self, direction: Direction, payload: dict[str, Any], score: float, explanation: Explanation | None, label: str
    ) -> Decision:
        outcome = self.engine.evaluate(direction, payload, score, explanation=explanation)
        self.events.append(
            ToolCallEvent(
                label=label, direction=direction.value, score=score, decision=outcome.decision.value, explanation=explanation
            )
        )

        if outcome.decision is not Decision.HOLD:
            return outcome.decision

        approval_decision = self.on_hold(outcome.pending_id, direction, payload, label, score)
        final = self.engine.apply_resolution(outcome.pending_id, approval_decision, decided_by="mcp-operator")
        self.events.append(
            ToolCallEvent(
                label=label,
                direction=direction.value,
                score=score,
                decision=outcome.decision.value,
                resolution=approval_decision.value,
                final_decision=final.decision.value,
                explanation=explanation,
            )
        )
        return final.decision
