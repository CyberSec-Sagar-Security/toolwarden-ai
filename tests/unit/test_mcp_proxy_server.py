"""GuardedMCPServer exercises the same enforcement matrix as
test_guarded_loop.py's GuardedOpenAIToolLoop tests, but drives it through a
real in-memory MCP Client/Server round trip (mcp.client.Client(server) —
see the MCP SDK's own examples/stories/_harness.py pattern) instead of a
mocked OpenAI client. No subprocess/stdio transport needed for this: the
SDK supports passing a Server instance directly as the connection target.

No pytest-asyncio plugin in this project's dependencies — anyio (already an
mcp transitive dependency) drives each async scenario via anyio.run() from
an ordinary sync test function instead.
"""

from __future__ import annotations

import anyio
from mcp.client import Client

from toolwarden.enforcement.approval_queue import ApprovalDecision
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import InMemorySink
from toolwarden.mcp_proxy.server import GuardedMCPServer
from toolwarden.models import Explanation

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch something.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "subject", "body"],
            },
        },
    },
]


class FakeClassifier:
    """Returns a scripted score per call, in order — avoids loading real models in a unit test.
    score_and_explain() draws from the same sequence and pairs each score with a fake Explanation
    tagged with that score.
    """

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    def score(self, text: str) -> float:
        return next(self._scores)

    def score_and_explain(self, text: str) -> tuple[float, Explanation]:
        score = next(self._scores)
        explanation = Explanation(deberta_top_tokens=[(f"fake-token-for-{score}", score)], lightgbm_top_features=None)
        return score, explanation


def _refuse_hold(*args):
    raise AssertionError(f"should not hold, got on_hold call with args={args}")


def _make_server(tool_functions, classifier, on_hold, policy=None, approval_sink=None):
    return GuardedMCPServer(
        name="test-server",
        tools_schema=TOOLS_SCHEMA,
        tool_functions=tool_functions,
        interceptor=Interceptor(sink=InMemorySink()),
        classifier=classifier,
        policy=policy or PolicyEngine(block_threshold=0.85, hold_threshold=0.5),
        approval_sink=approval_sink or InMemorySink(),
        on_hold=on_hold,
    )


def test_low_scores_allow_tool_call_through_with_real_content():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.1])  # request score, result score
        guarded = _make_server({"fetch": lambda url: f"content for {url}"}, classifier, _refuse_hold)

        async with Client(guarded.server) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == {"fetch", "send_email"}
            result = await client.call_tool("fetch", {"url": "https://x"})

        assert result.is_error is False
        assert result.content[0].text == "content for https://x"
        assert [e.decision for e in guarded.events] == [Decision.ALLOW.value, Decision.ALLOW.value]

    anyio.run(scenario)


def test_high_request_score_blocks_before_tool_executes():
    async def scenario():
        classifier = FakeClassifier([0.95])  # request auto-block; no result-side score consumed
        executed = []
        guarded = _make_server(
            {"send_email": lambda to, subject, body: executed.append((to, subject, body)) or "sent"},
            classifier,
            _refuse_hold,
        )

        async with Client(guarded.server) as client:
            result = await client.call_tool("send_email", {"to": "x", "subject": "y", "body": "z"})

        assert executed == []  # tool never actually ran
        assert result.is_error is True
        assert "blocked" in result.content[0].text.lower()
        assert len(guarded.events) == 1  # only the request-side decision
        assert guarded.events[0].decision == Decision.BLOCK.value

    anyio.run(scenario)


def test_high_result_score_quarantines_content_before_it_reaches_the_client():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.95])  # request allow, result auto-quarantine
        guarded = _make_server({"fetch": lambda url: "malicious content"}, classifier, _refuse_hold)

        async with Client(guarded.server) as client:
            result = await client.call_tool("fetch", {"url": "https://x"})

        assert result.is_error is False
        assert "quarantined" in result.content[0].text.lower()
        assert "malicious content" not in result.content[0].text
        assert guarded.events[1].decision == Decision.QUARANTINE.value

    anyio.run(scenario)


def test_mid_result_score_holds_then_denied_quarantines():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.6])  # request allow, result hold
        calls = []

        def on_hold(pending_id, direction, payload, label, score):
            calls.append((direction, score))
            return ApprovalDecision.DENIED

        guarded = _make_server({"fetch": lambda url: "sketchy content"}, classifier, on_hold)

        async with Client(guarded.server) as client:
            result = await client.call_tool("fetch", {"url": "https://x"})

        assert len(calls) == 1
        assert calls[0][0] is Direction.RESULT
        assert "quarantined" in result.content[0].text.lower()
        assert guarded.events[-1].final_decision == Decision.QUARANTINE.value

    anyio.run(scenario)


def test_mid_result_score_holds_then_approved_passes_real_content_through():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.6])
        guarded = _make_server(
            {"fetch": lambda url: "borderline content"}, classifier, lambda *a: ApprovalDecision.APPROVED
        )

        async with Client(guarded.server) as client:
            result = await client.call_tool("fetch", {"url": "https://x"})

        assert result.content[0].text == "borderline content"
        assert guarded.events[-1].final_decision == Decision.ALLOW.value

    anyio.run(scenario)


def test_explanation_is_attached_to_events():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.95])
        guarded = _make_server({"fetch": lambda url: "malicious content"}, classifier, _refuse_hold)

        async with Client(guarded.server) as client:
            await client.call_tool("fetch", {"url": "https://x"})

        assert guarded.events[0].explanation == Explanation(deberta_top_tokens=[("fake-token-for-0.1", 0.1)], lightgbm_top_features=None)
        assert guarded.events[1].explanation == Explanation(deberta_top_tokens=[("fake-token-for-0.95", 0.95)], lightgbm_top_features=None)

    anyio.run(scenario)


def test_approval_resolution_is_attributed_and_logged():
    async def scenario():
        classifier = FakeClassifier([0.1, 0.6])
        approval_sink = InMemorySink()
        guarded = _make_server(
            {"fetch": lambda url: "content"},
            classifier,
            lambda *a: ApprovalDecision.APPROVED,
            approval_sink=approval_sink,
        )

        async with Client(guarded.server) as client:
            await client.call_tool("fetch", {"url": "https://x"})

        assert len(approval_sink.records) == 1
        assert approval_sink.records[0]["decided_by"] == "mcp-operator"
        assert approval_sink.records[0]["decision"] == "approved"

    anyio.run(scenario)
