"""Wraps an OpenAI chat-completions tool-calling loop so every tool call the
model wants to make, and every result fed back to it, passes through a
ToolWarden Interceptor first.

Phase 2 scope: passthrough only (see interceptor.py). No blocking here.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from openai import OpenAI

from toolwarden.interceptor import Interceptor
from toolwarden.models import ToolCallRequest, ToolCallResult

ToolFunction = Callable[..., Any]


class OpenAIToolLoop:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools_schema: list[dict[str, Any]],
        tool_functions: dict[str, ToolFunction],
        interceptor: Interceptor,
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_schema = tools_schema
        self.tool_functions = tool_functions
        self.interceptor = interceptor
        self.session_id = session_id or str(uuid.uuid4())

    def run(self, user_message: str, max_rounds: int = 5) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema,
            )
            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))

            if not choice.message.tool_calls:
                return choice.message.content or ""

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

                content, is_error, latency_ms = self._execute(request)

                result = self.interceptor.intercept_result(
                    ToolCallResult(
                        call_id=request.call_id,
                        tool_name=request.tool_name,
                        content=content,
                        is_error=is_error,
                        latency_ms=latency_ms,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content
                        if isinstance(result.content, str)
                        else json.dumps(result.content),
                    }
                )

        return "[max tool-call rounds reached without a final answer]"

    def _execute(self, request: ToolCallRequest) -> tuple[Any, bool, int]:
        func = self.tool_functions.get(request.tool_name)
        start = time.monotonic()
        if func is None:
            return f"unknown tool: {request.tool_name}", True, 0
        try:
            content = func(**request.arguments)
            is_error = False
        except Exception as exc:  # tool failure surfaces as a result, not a crash
            content = str(exc)
            is_error = True
        return content, is_error, int((time.monotonic() - start) * 1000)
