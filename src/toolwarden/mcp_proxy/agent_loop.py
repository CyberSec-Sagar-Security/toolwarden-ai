"""Structurally the same shape as adapters/openai_adapter.py's
OpenAIToolLoop (Phase 2, passthrough), but tool execution goes through an
MCP client's call_tool() instead of a local Python function call.

Deliberately does no enforcement of its own: enforcement already happened
inside GuardedMCPServer.on_call_tool before the result crossed back over
the wire. This loop only ever sees what a real MCP client would see —
allowed content, a quarantine notice, or a blocked-call error — which is
the point of exercising the protocol boundary instead of a shared
in-process function call, per the Phase 10 stop gate.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from mcp.client import Client
from openai import AsyncOpenAI


class MCPToolLoop:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tools_schema: list[dict[str, Any]],
        mcp_client: Client,
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_schema = tools_schema
        self.mcp_client = mcp_client
        self.session_id = session_id or str(uuid.uuid4())

    async def run(self, user_message: str, max_rounds: int = 5) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for _ in range(max_rounds):
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools_schema
            )
            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))

            if not choice.message.tool_calls:
                return choice.message.content or ""

            for tool_call in choice.message.tool_calls:
                arguments = json.loads(tool_call.function.arguments or "{}")
                result = await self.mcp_client.call_tool(tool_call.function.name, arguments)
                text = "\n".join(getattr(block, "text", str(block)) for block in result.content)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": text})

        return "[max tool-call rounds reached without a final answer]"
