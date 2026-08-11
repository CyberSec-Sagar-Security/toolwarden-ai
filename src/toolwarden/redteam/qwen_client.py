"""Thin client for the local red-teamer LLM (Qwen3.5-9B as of 2026-08-11,
previously Qwen2.5-14B-Instruct), served by llama-server's OpenAI-compatible
API (see docs/known_limitations.md and Phase 1's threat model for why the
local LLM itself is a trust boundary — its output is treated as
untrusted-but-useful generation material, never auto-trusted into the
training/eval pool without review).

Qwen3.5 is a "thinking" model: by default it emits a hidden reasoning
preamble into a separate `reasoning_content` field before the actual
answer, which can consume the whole max_tokens budget and leave `content`
empty (confirmed directly against this server — at max_tokens=220 the
model was still mid-reasoning with zero actual output). Disabled via
`chat_template_kwargs: {"enable_thinking": false}`, which llama-server
passes through to the chat template — confirmed working (clean answer,
finish_reason="stop") before this was wired in here.

Model setup is manual (Sagar's responsibility, not this code's) — see the
Phase 7 setup steps in chat history / README. This module only talks to
whatever's already running at LOCAL_LLM_BASE_URL.
"""

from __future__ import annotations

from openai import OpenAI

LOCAL_LLM_BASE_URL = "http://127.0.0.1:8080/v1"


def get_client() -> OpenAI:
    return OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key="not-needed")


def generate(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float = 0.95, max_tokens: int = 220) -> str:
    response = client.chat.completions.create(
        model="qwen3.5-9b",  # llama-server ignores this and serves whatever's loaded
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(
            "Empty completion content — the model may have burned max_tokens on hidden reasoning "
            "(check enable_thinking is actually being honored) rather than a real generation failure."
        )
    return text
