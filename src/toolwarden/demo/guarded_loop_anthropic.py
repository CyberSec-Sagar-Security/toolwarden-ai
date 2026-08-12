"""Phase 13 follow-up: the Anthropic side of the "direct API function-calling
(OpenAI/Anthropic tool_use)" locked scope from Phase 2 -- built, tested, and
demonstrated end to end for the first time. Every phase up to this one only
ever exercised OpenAI (confirmed via the Phase 13 audit: zero references to
`anthropic`/`claude` in any toolwarden source file). Same shape of work as
Phase 10's MCP proxy: one adapter translating a different tool-calling
format into the existing intercepted structure, reusing Classifier,
PolicyEngine, EnforcementEngine, and ApprovalQueue completely unmodified.

Claude's Messages API tool-calling differs from OpenAI's chat-completions
tool-calling in three structural ways this adapter has to bridge, none of
which touch the enforcement pipeline itself:

1. Tool schema shape: Claude wants {"name", "description", "input_schema"},
   not OpenAI's {"type": "function", "function": {...}}. _to_anthropic_tool()
   converts demo/tools.py's existing OpenAI-shaped TOOLS_SCHEMA rather than
   maintaining the tool contract a third time (mcp_proxy/server.py's
   _to_mcp_tool() does the same conversion for MCP).
2. A tool call is a content block (type="tool_use") inside Message.content,
   not a separate parallel list (OpenAI's message.tool_calls) -- and
   ToolUseBlock.input is already a parsed dict, not a JSON string to
   json.loads() the way OpenAI's function.arguments is.
3. Multiple tool_use blocks in one assistant turn get answered with ONE
   user message containing multiple tool_result blocks (one per
   tool_use_id), not one separate "tool"-role message per call the way
   OpenAI's loop appends.

Reuses guarded_loop.py's own ToolCallEvent/GuardedRunResult/OnHold/
QUARANTINE_MESSAGE/BLOCK_MESSAGE_TEMPLATE directly (imported, not
redefined) so all three adapters (OpenAI, MCP, Anthropic) share one
definition of what "blocked" and "quarantined" mean.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from anthropic import Anthropic

from toolwarden.demo.classify import Classifier
from toolwarden.demo.guarded_loop import (
    BLOCK_MESSAGE_TEMPLATE,
    QUARANTINE_MESSAGE,
    GuardedRunResult,
    OnHold,
    ToolCallEvent,
)
from toolwarden.enforcement.approval_queue import ApprovalQueue
from toolwarden.enforcement.engine import EnforcementEngine
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import LogSink
from toolwarden.models import Explanation, ToolCallRequest, ToolCallResult


def _to_anthropic_tool(schema_entry: dict[str, Any]) -> dict[str, Any]:
    """Converts one entry of an OpenAI-shaped tools_schema (demo/tools.py)
    into Claude's tool shape.
    """
    fn = schema_entry["function"]
    return {"name": fn["name"], "description": fn.get("description", ""), "input_schema": fn["parameters"]}


class GuardedAnthropicToolLoop:
    def __init__(
        self,
        client: Anthropic,
        model: str,
        tools_schema: list[dict[str, Any]],
        tool_functions: dict[str, Callable[..., Any]],
        interceptor: Interceptor,
        classifier: Classifier,
        policy: PolicyEngine,
        approval_sink: LogSink,
        on_hold: OnHold,
        session_id: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_schema = [_to_anthropic_tool(entry) for entry in tools_schema]
        self.tool_functions = tool_functions
        self.interceptor = interceptor
        self.classifier = classifier
        self.engine = EnforcementEngine(policy=policy, approval_queue=ApprovalQueue(sink=approval_sink))
        self.on_hold = on_hold
        self.session_id = session_id or str(uuid.uuid4())
        self.max_tokens = max_tokens

    def run(self, user_message: str, max_rounds: int = 5) -> GuardedRunResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        events: list[ToolCallEvent] = []

        for _ in range(max_rounds):
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, messages=messages, tools=self.tools_schema
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
            if not tool_use_blocks:
                final_text = "".join(block.text for block in response.content if block.type == "text")
                return GuardedRunResult(final_text=final_text, events=events)

            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                arguments = block.input
                request = self.interceptor.intercept_request(
                    ToolCallRequest(
                        tool_name=block.name,
                        arguments=arguments,
                        source="anthropic",
                        session_id=self.session_id,
                        call_id=block.id,
                    )
                )

                request_score, request_explanation = self.classifier.score_and_explain(json.dumps(arguments))
                request_decision = self._enforce(
                    Direction.REQUEST,
                    request.to_dict(),
                    request_score,
                    request_explanation,
                    f"request: {request.tool_name}",
                    events,
                )

                if request_decision is Decision.BLOCK:
                    # No real tool output exists to inspect here — the call
                    # never ran, so this is our own system-generated notice,
                    # not attacker-influenced content. Mirrors guarded_loop.py.
                    content = BLOCK_MESSAGE_TEMPLATE.format(tool_name=request.tool_name, reason="blocked pre-execution")
                    self.interceptor.intercept_result(
                        ToolCallResult(call_id=request.call_id, tool_name=request.tool_name, content=content, is_error=True)
                    )
                    final_content, is_error_out = content, True
                else:
                    content, is_error = self._execute(request.tool_name, arguments)
                    result = self.interceptor.intercept_result(
                        ToolCallResult(call_id=request.call_id, tool_name=request.tool_name, content=content, is_error=is_error)
                    )

                    result_score, result_explanation = self.classifier.score_and_explain(str(result.content))
                    result_decision = self._enforce(
                        Direction.RESULT,
                        result.to_dict(),
                        result_score,
                        result_explanation,
                        f"result: {request.tool_name}",
                        events,
                    )

                    final_content = QUARANTINE_MESSAGE if result_decision is Decision.QUARANTINE else result.content
                    is_error_out = is_error

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": final_content if isinstance(final_content, str) else json.dumps(final_content),
                        "is_error": is_error_out,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return GuardedRunResult(final_text="[max tool-call rounds reached without a final answer]", events=events)

    def _execute(self, tool_name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        func = self.tool_functions.get(tool_name)
        if func is None:
            return f"unknown tool: {tool_name}", True
        try:
            return func(**arguments), False
        except Exception as exc:  # tool failure surfaces as a result, not a crash
            return str(exc), True

    def _enforce(
        self,
        direction: Direction,
        payload: dict[str, Any],
        score: float,
        explanation: Explanation | None,
        label: str,
        events: list[ToolCallEvent],
    ) -> Decision:
        outcome = self.engine.evaluate(direction, payload, score, explanation=explanation)
        events.append(
            ToolCallEvent(
                label=label, direction=direction.value, score=score, decision=outcome.decision.value, explanation=explanation
            )
        )

        if outcome.decision is not Decision.HOLD:
            return outcome.decision

        approval_decision = self.on_hold(outcome.pending_id, direction, payload, label, score)
        final = self.engine.apply_resolution(outcome.pending_id, approval_decision, decided_by="demo-operator")
        events.append(
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
