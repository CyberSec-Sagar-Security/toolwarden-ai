"""Mirrors test_guarded_loop.py's enforcement matrix exactly, but drives
GuardedAnthropicToolLoop with a scripted fake Anthropic client instead of a
mocked OpenAI one -- proves the enforcement logic (shared with every other
adapter via EnforcementEngine, unmodified) behaves identically regardless
of which tool-calling format triggered it, without needing a live
ANTHROPIC_API_KEY for this part.
"""

import json
from types import SimpleNamespace

from toolwarden.demo.guarded_loop_anthropic import GuardedAnthropicToolLoop
from toolwarden.enforcement.approval_queue import ApprovalDecision
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import InMemorySink
from toolwarden.models import Explanation


class FakeToolUseBlock:
    def __init__(self, block_id: str, name: str, input_dict: dict):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = input_dict


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, content: list):
        self.content = content


class FakeAnthropicClient:
    """Scripted Anthropic client: returns one Message per call, in order.
    Records the `messages` list it was called with each time, so tests can
    inspect exactly what the loop sent back (e.g. tool_result blocks).
    """

    def __init__(self, responses: list[FakeMessage]):
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, max_tokens, messages, tools):
        self.calls.append(list(messages))  # snapshot -- `messages` keeps mutating after this call returns
        return next(self._responses)


class FakeClassifier:
    """Returns a scripted score per call, in order — same pattern as the OpenAI/MCP tests."""

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    def score_and_explain(self, text: str) -> tuple[float, Explanation]:
        score = next(self._scores)
        explanation = Explanation(deberta_top_tokens=[(f"fake-token-for-{score}", score)], lightgbm_top_features=None)
        return score, explanation


def _tool_call_then_final(tool_name: str, arguments: dict, final_text: str = "Done.") -> list[FakeMessage]:
    return [
        FakeMessage(content=[FakeToolUseBlock("toolu_1", tool_name, arguments)]),
        FakeMessage(content=[FakeTextBlock(final_text)]),
    ]


def _make_loop(client, classifier, tool_functions, on_hold, policy=None):
    return GuardedAnthropicToolLoop(
        client=client,
        model="claude-haiku-4-5",
        tools_schema=[],
        tool_functions=tool_functions,
        interceptor=Interceptor(sink=InMemorySink()),
        classifier=classifier,
        policy=policy or PolicyEngine(block_threshold=0.85, hold_threshold=0.5),
        approval_sink=InMemorySink(),
        on_hold=on_hold,
    )


def test_low_scores_allow_tool_to_execute_normally():
    client = FakeAnthropicClient(_tool_call_then_final("lookup", {"q": "weather"}))
    classifier = FakeClassifier([0.1, 0.1])
    tool_functions = {"lookup": lambda q: f"result for {q}"}

    def on_hold(*args):
        raise AssertionError("should not hold on low scores")

    loop = _make_loop(client, classifier, tool_functions, on_hold)
    result = loop.run("do a lookup")

    assert result.final_text == "Done."
    assert [e.decision for e in result.events] == [Decision.ALLOW.value, Decision.ALLOW.value]


def test_high_result_score_quarantines_content_before_agent_sees_it():
    client = FakeAnthropicClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.95])

    def tool_fn(url):
        return "malicious content"

    def on_hold(*args):
        raise AssertionError("0.95 should auto-quarantine, not hold")

    loop = _make_loop(client, classifier, {"fetch": tool_fn}, on_hold)
    result = loop.run("fetch it")

    assert result.events[1].decision == Decision.QUARANTINE.value


def test_mid_result_score_holds_then_denied_quarantines():
    client = FakeAnthropicClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])
    calls = []

    def on_hold(pending_id, direction, payload, label, score):
        calls.append((direction, score))
        return ApprovalDecision.DENIED

    loop = _make_loop(client, classifier, {"fetch": lambda url: "sketchy content"}, on_hold)
    result = loop.run("fetch it")

    assert len(calls) == 1
    assert calls[0][0] is Direction.RESULT
    assert result.events[-1].final_decision == Decision.QUARANTINE.value


def test_mid_result_score_holds_then_approved_passes_through():
    client = FakeAnthropicClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])

    def on_hold(pending_id, direction, payload, label, score):
        return ApprovalDecision.APPROVED

    loop = _make_loop(client, classifier, {"fetch": lambda url: "borderline content"}, on_hold)
    result = loop.run("fetch it")

    assert result.events[-1].final_decision == Decision.ALLOW.value


def test_high_request_score_blocks_before_tool_executes():
    client = FakeAnthropicClient(_tool_call_then_final("send_email", {"to": "x", "subject": "y", "body": "z"}))
    classifier = FakeClassifier([0.95])  # only one score consumed: blocked request never reaches result-side
    executed = []

    def tool_fn(to, subject, body):
        executed.append((to, subject, body))
        return "sent"

    def on_hold(*args):
        raise AssertionError("0.95 should auto-block, not hold")

    loop = _make_loop(client, classifier, {"send_email": tool_fn}, on_hold)
    result = loop.run("send it")

    assert executed == []
    assert len(result.events) == 1


def test_approval_resolution_is_attributed_and_logged():
    client = FakeAnthropicClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])
    approval_sink = InMemorySink()

    loop = GuardedAnthropicToolLoop(
        client=client,
        model="claude-haiku-4-5",
        tools_schema=[],
        tool_functions={"fetch": lambda url: "content"},
        interceptor=Interceptor(sink=InMemorySink()),
        classifier=classifier,
        policy=PolicyEngine(),
        approval_sink=approval_sink,
        on_hold=lambda *a: ApprovalDecision.APPROVED,
    )
    loop.run("fetch it")

    assert len(approval_sink.records) == 1
    assert approval_sink.records[0]["decided_by"] == "demo-operator"
    assert approval_sink.records[0]["decision"] == "approved"


def test_multiple_tool_use_blocks_in_one_turn_each_get_a_matching_tool_result():
    """Claude can request several tools in one assistant turn -- unlike
    OpenAI's one-message-per-call reply, all their results must land in a
    single follow-up user message, each tagged with its own tool_use_id.
    """
    client = FakeAnthropicClient(
        [
            FakeMessage(
                content=[
                    FakeToolUseBlock("toolu_a", "lookup", {"q": "one"}),
                    FakeToolUseBlock("toolu_b", "lookup", {"q": "two"}),
                ]
            ),
            FakeMessage(content=[FakeTextBlock("Done.")]),
        ]
    )
    classifier = FakeClassifier([0.1, 0.1, 0.1, 0.1])  # request+result for each of 2 calls

    def on_hold(*args):
        raise AssertionError("should not hold on low scores")

    loop = _make_loop(client, classifier, {"lookup": lambda q: f"result for {q}"}, on_hold)
    result = loop.run("do two lookups")

    assert result.final_text == "Done."
    assert len(result.events) == 4  # request+result per call, 2 calls

    # The second create() call is the follow-up turn -- its `messages` arg
    # should end with one user message carrying both tool_result blocks.
    second_call_messages = client.calls[1]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    tool_use_ids = {block["tool_use_id"] for block in tool_result_message["content"]}
    assert tool_use_ids == {"toolu_a", "toolu_b"}
    contents = {block["tool_use_id"]: block["content"] for block in tool_result_message["content"]}
    assert contents["toolu_a"] == "result for one"
    assert contents["toolu_b"] == "result for two"
