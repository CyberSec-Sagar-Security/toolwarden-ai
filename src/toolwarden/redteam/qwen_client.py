"""Thin client for the local Qwen2.5-14B-Instruct red-teamer, served by
llama-server's OpenAI-compatible API (see docs/known_limitations.md and
Phase 1's threat model for why the local LLM itself is a trust boundary —
its output is treated as untrusted-but-useful generation material, never
auto-trusted into the training/eval pool without review).

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
        model="qwen2.5-14b-instruct",  # llama-server ignores this and serves whatever's loaded
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
