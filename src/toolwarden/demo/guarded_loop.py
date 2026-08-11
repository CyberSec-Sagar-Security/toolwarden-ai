"""Phase 9: chains Phase 2 (interception), Phase 4 (classifier), and
Phase 6 (policy + approval queue) into one live OpenAI tool-calling loop —
the integration none of the earlier phases actually built. Every outbound
tool-call request and inbound tool-call result gets intercepted, scored,
policy-decided, and enforced before it can affect the agent's next step.

Approval resolution is a synchronous callback, not a real async queue —
this is a demo of the flow, not Phase 12's production approval service.
The caller supplies on_hold(pending_id, direction, payload, label, score)
-> ApprovalDecision; run_demo.py controls it (scripted approve/deny for
reliability, or a live input() prompt for an interactive run).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

from toolwarden.demo.classify import Classifier
from toolwarden.enforcement.approval_queue import ApprovalDecision, ApprovalQueue
from toolwarden.enforcement.engine import EnforcementEngine
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import LogSink
from toolwarden.models import ToolCallRequest, ToolCallResult

OnHold = Callable[[str, Direction, dict, str, float | None], ApprovalDecision]

QUARANTINE_MESSAGE = (
    "[ToolWarden] Content quarantined: potential prompt injection detected in this tool result "
    "and removed before it reached the agent."
)
BLOCK_MESSAGE_TEMPLATE = "[ToolWarden] Blocked: the {tool_name} call did not execute (policy: {reason})."


@dataclass
class ToolCallEvent:
    """One decision point in the trace — a request or a result being
    scored and enforced. run_demo.py prints these for the live narrative.
    """

    label: str
    direction: str
    score: float | None
    decision: str
    resolution: str | None = None
    final_decision: str | None = None


@dataclass
class GuardedRunResult:
    final_text: str
    events: list[ToolCallEvent] = field(default_factory=list)


class GuardedOpenAIToolLoop:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools_schema: list[dict[str, Any]],
        tool_functions: dict[str, Callable[..., Any]],
        interceptor: Interceptor,
        classifier: Classifier,
        policy: PolicyEngine,
        approval_sink: LogSink,
        on_hold: OnHold,
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_schema = tools_schema
        self.tool_functions = tool_functions
        self.interceptor = interceptor
        self.classifier = classifier
        self.engine = EnforcementEngine(policy=policy, approval_queue=ApprovalQueue(sink=approval_sink))
        self.on_hold = on_hold
        self.session_id = session_id or str(uuid.uuid4())

    def run(self, user_message: str, max_rounds: int = 5) -> GuardedRunResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        events: list[ToolCallEvent] = []

        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools_schema
            )
            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))

            if not choice.message.tool_calls:
                return GuardedRunResult(final_text=choice.message.content or "", events=events)

            for tool_call in choice.message.tool_calls:
                arguments = json.loads(tool_call.function.arguments or "{}")
                request = self.interceptor.intercept_request(
                    ToolCallRequest(
                        tool_name=tool_call.function.name,
                        arguments=arguments,
                        source="openai",
                        session_id=self.session_id,
                        call_id=tool_call.id,
                    )
                )

                request_score = self.classifier.score(json.dumps(arguments))
                request_decision = self._enforce(
                    Direction.REQUEST, request.to_dict(), request_score, f"request: {request.tool_name}", events
                )

                if request_decision is Decision.BLOCK:
                    # No real tool output exists to inspect here — the call
                    # never ran, so this is our own system-generated notice,
                    # not attacker-influenced content. Log it for the audit
                    # trail, but don't spend a classifier call re-scoring
                    # text we just wrote ourselves.
                    content = BLOCK_MESSAGE_TEMPLATE.format(tool_name=request.tool_name, reason="blocked pre-execution")
                    self.interceptor.intercept_result(
                        ToolCallResult(call_id=request.call_id, tool_name=request.tool_name, content=content, is_error=True)
                    )
                    final_content = content
                else:
                    content, is_error = self._execute(request.tool_name, request.arguments)
                    result = self.interceptor.intercept_result(
                        ToolCallResult(call_id=request.call_id, tool_name=request.tool_name, content=content, is_error=is_error)
                    )

                    result_score = self.classifier.score(str(result.content))
                    result_decision = self._enforce(
                        Direction.RESULT, result.to_dict(), result_score, f"result: {request.tool_name}", events
                    )

                    final_content = QUARANTINE_MESSAGE if result_decision is Decision.QUARANTINE else result.content

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": request.call_id,
                        "content": final_content if isinstance(final_content, str) else json.dumps(final_content),
                    }
                )

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
        self, direction: Direction, payload: dict[str, Any], score: float, label: str, events: list[ToolCallEvent]
    ) -> Decision:
        outcome = self.engine.evaluate(direction, payload, score)
        events.append(ToolCallEvent(label=label, direction=direction.value, score=score, decision=outcome.decision.value))

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
            )
        )
        return final.decision
