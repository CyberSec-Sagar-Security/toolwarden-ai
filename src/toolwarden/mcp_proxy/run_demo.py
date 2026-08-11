"""Phase 10: the same three Phase 9 scenarios (docs/demo_walkthrough.md),
run again through an MCP server instead of direct OpenAI tool-calling —
proving the detection/enforcement behavior isn't an artifact of one
integration adapter. Reuses Phase 9's exact canned content
(src/toolwarden/demo/tools.py) so scores and outcomes are directly
comparable between the two adapters, and the same trained classifier
(src/toolwarden/demo/classify.py) actually scoring the content, not a
second copy of it.

Structural difference from Phase 9's guarded_loop.py: enforcement now runs
inside GuardedMCPServer.on_call_tool (src/toolwarden/mcp_proxy/server.py)
on the *server* side of an MCP round trip, not inline in the agent loop.
The agent loop here (agent_loop.py's MCPToolLoop) only ever sees what a
real MCP client would see. This script prints the enforcement narrative by
reading guarded.events directly off the in-process server object — that
in-process access exists for this demo's readability, a real MCP client on
the other side of a subprocess/stdio transport doesn't get it.

Run with: python -m toolwarden.mcp_proxy.run_demo
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from toolwarden import config

config.configure_hf_cache_env()

from mcp.client import Client  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from toolwarden.demo.classify import Classifier  # noqa: E402
from toolwarden.demo.tools import (  # noqa: E402
    ALLOW_URL,
    FALSE_POSITIVE_URL,
    HOLD_URL,
    TOOLS_SCHEMA,
    SentEmailLog,
    fetch_webpage,
    make_send_email,
)
from toolwarden.enforcement.approval_queue import ApprovalDecision  # noqa: E402
from toolwarden.enforcement.policy import PolicyEngine  # noqa: E402
from toolwarden.interceptor import Interceptor  # noqa: E402
from toolwarden.logging_sink import JsonlFileSink  # noqa: E402
from toolwarden.mcp_proxy.agent_loop import MCPToolLoop  # noqa: E402
from toolwarden.mcp_proxy.server import GuardedMCPServer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = REPO_ROOT / "logs" / "mcp_demo_traffic.jsonl"
APPROVAL_LOG_PATH = REPO_ROOT / "logs" / "mcp_demo_approvals.jsonl"

MODEL = "gpt-4o-mini"


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_events(events) -> None:
    for e in events:
        line = f"  [{e.direction:>7}] {e.label:<28} score={e.score:.3f} -> {e.decision}"
        if e.resolution:
            line += f"  (human: {e.resolution} -> final: {e.final_decision})"
        print(line)


def _verify_startup_scores(classifier: Classifier) -> None:
    """Same discipline as demo/run_demo.py's own startup check: proves the
    scores in tools.py's docstring still hold on this run before anything
    else executes, rather than trusting a stale snapshot.
    """
    print("Startup check — live classifier scores for each scenario's content:")
    for label, url in [("allow", ALLOW_URL), ("hold", HOLD_URL), ("false_positive", FALSE_POSITIVE_URL)]:
        score = classifier.score(fetch_webpage(url))
        print(f"  {label:<15} {score:.3f}")
    print()


def _refuse_hold(pending_id, direction, payload, label, score):
    raise AssertionError(f"Unexpected HOLD: {label} score={score}")


async def _run_through_mcp(classifier: Classifier, openai_client: AsyncOpenAI, url: str, on_hold, log: SentEmailLog):
    guarded = GuardedMCPServer(
        name="toolwarden-mcp-demo",
        tools_schema=TOOLS_SCHEMA,
        tool_functions={"fetch_webpage": fetch_webpage, "send_email": make_send_email(log)},
        interceptor=Interceptor(sink=JsonlFileSink(LOG_PATH)),
        classifier=classifier,
        policy=PolicyEngine(),
        approval_sink=JsonlFileSink(APPROVAL_LOG_PATH),
        on_hold=on_hold,
    )
    async with Client(guarded.server) as mcp_client:
        loop = MCPToolLoop(client=openai_client, model=MODEL, tools_schema=TOOLS_SCHEMA, mcp_client=mcp_client)
        final_text = await loop.run(f"Please fetch {url} and tell me what it says in one sentence.")
    return guarded, final_text


async def run_allow_scenario(classifier: Classifier, openai_client: AsyncOpenAI) -> None:
    _print_header("SCENARIO 1: ALLOW — genuinely benign page (via MCP)")
    log = SentEmailLog()
    guarded, final_text = await _run_through_mcp(classifier, openai_client, ALLOW_URL, _refuse_hold, log)
    _print_events(guarded.events)
    print(f"\nAgent's final answer: {final_text}")
    print(f"Emails sent during this run: {len(log.sent)} (expected 0)")


async def run_hold_scenario(classifier: Classifier, openai_client: AsyncOpenAI, decision: ApprovalDecision) -> None:
    _print_header(f"SCENARIO 2: HOLD -> {decision.value.upper()} — injected page (via MCP)")

    def on_hold(pending_id, direction, payload, label, score):
        print(f"\n  >>> HELD for approval: {label} (score={score:.3f}). Reviewer decision: {decision.value}. <<<\n")
        return decision

    log = SentEmailLog()
    guarded, final_text = await _run_through_mcp(classifier, openai_client, HOLD_URL, on_hold, log)
    _print_events(guarded.events)
    print(f"\nAgent's final answer: {final_text}")
    print(f"Emails sent during this run: {len(log.sent)}")
    if log.sent:
        for email in log.sent:
            print(f"  -> to={email['to']!r} subject={email['subject']!r}")


async def run_false_positive_scenario(classifier: Classifier, openai_client: AsyncOpenAI) -> None:
    _print_header("SCENARIO 3: FALSE POSITIVE — benign page, auto-quarantined (via MCP)")
    log = SentEmailLog()
    guarded, final_text = await _run_through_mcp(classifier, openai_client, FALSE_POSITIVE_URL, _refuse_hold, log)
    _print_events(guarded.events)
    print(f"\nAgent's final answer: {final_text}")
    print(
        "\nNote: this page is genuinely benign (a weather update) but scored high enough to be "
        "auto-quarantined before the MCP client ever saw the real content — same disclosed "
        "classifier weakness as Phase 9, this time enforced from the MCP server side."
    )


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — this demo makes live OpenAI API calls.")

    print("Loading classifier (DeBERTa + LightGBM + ensemble)...")
    classifier = Classifier()
    _verify_startup_scores(classifier)

    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    await run_allow_scenario(classifier, openai_client)
    await run_hold_scenario(classifier, openai_client, ApprovalDecision.DENIED)
    await run_hold_scenario(classifier, openai_client, ApprovalDecision.APPROVED)
    await run_false_positive_scenario(classifier, openai_client)

    print("\n" + "=" * 78)
    print(f"Full tool-call trace: {LOG_PATH}")
    print(f"Approval decisions:   {APPROVAL_LOG_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
