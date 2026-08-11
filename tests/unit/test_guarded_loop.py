import json
from types import SimpleNamespace

from toolwarden.demo.guarded_loop import GuardedOpenAIToolLoop
from toolwarden.enforcement.approval_queue import ApprovalDecision
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.interceptor import Interceptor
from toolwarden.logging_sink import InMemorySink


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content
        self.role = "assistant"

    def model_dump(self, exclude_none=True):
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [SimpleNamespace(message=message)]


class FakeClient:
    """Scripted OpenAI client: returns one response per call, in order."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = iter(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools):
        return next(self._responses)


class FakeClassifier:
    """Returns a scripted score per call, in order — avoids loading real models in a unit test."""

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    def score(self, text: str) -> float:
        return next(self._scores)


def _tool_call_then_final(tool_name: str, arguments: dict, final_text: str = "Done.") -> list[FakeResponse]:
    return [
        FakeResponse(FakeMessage(tool_calls=[FakeToolCall("call-1", tool_name, arguments)])),
        FakeResponse(FakeMessage(content=final_text)),
    ]


def _make_loop(client, classifier, tool_functions, on_hold, policy=None):
    return GuardedOpenAIToolLoop(
        client=client,
        model="gpt-4o-mini",
        tools_schema=[],
        tool_functions=tool_functions,
        interceptor=Interceptor(sink=InMemorySink()),
        classifier=classifier,
        policy=policy or PolicyEngine(block_threshold=0.85, hold_threshold=0.5),
        approval_sink=InMemorySink(),
        on_hold=on_hold,
    )


def test_low_scores_allow_tool_to_execute_normally():
    client = FakeClient(_tool_call_then_final("lookup", {"q": "weather"}))
    classifier = FakeClassifier([0.1, 0.1])  # request score, result score
    tool_functions = {"lookup": lambda q: f"result for {q}"}

    def on_hold(*args):
        raise AssertionError("should not hold on low scores")

    loop = _make_loop(client, classifier, tool_functions, on_hold)
    result = loop.run("do a lookup")

    assert result.final_text == "Done."
    assert [e.decision for e in result.events] == [Decision.ALLOW.value, Decision.ALLOW.value]


def test_high_result_score_quarantines_content_before_agent_sees_it():
    client = FakeClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.95])  # request allow, result auto-quarantine

    def tool_fn(url):
        return "malicious content"

    def on_hold(*args):
        raise AssertionError("0.95 should auto-quarantine, not hold")

    loop = _make_loop(client, classifier, {"fetch": tool_fn}, on_hold)
    result = loop.run("fetch it")

    assert result.events[1].decision == Decision.QUARANTINE.value


def test_mid_result_score_holds_then_denied_quarantines():
    client = FakeClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])  # request allow, result hold
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
    client = FakeClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])

    def on_hold(pending_id, direction, payload, label, score):
        return ApprovalDecision.APPROVED

    loop = _make_loop(client, classifier, {"fetch": lambda url: "borderline content"}, on_hold)
    result = loop.run("fetch it")

    assert result.events[-1].final_decision == Decision.ALLOW.value


def test_high_request_score_blocks_before_tool_executes():
    client = FakeClient(_tool_call_then_final("send_email", {"to": "x", "subject": "y", "body": "z"}))
    # Only one score consumed: a blocked request never reaches result-side
    # classification, since there's no real tool output to inspect.
    classifier = FakeClassifier([0.95])
    executed = []

    def tool_fn(to, subject, body):
        executed.append((to, subject, body))
        return "sent"

    def on_hold(*args):
        raise AssertionError("0.95 should auto-block, not hold")

    loop = _make_loop(client, classifier, {"send_email": tool_fn}, on_hold)
    result = loop.run("send it")

    assert executed == []  # tool never actually ran
    assert len(result.events) == 1  # only the request-side decision, no result-side classification


def test_approval_resolution_is_attributed_and_logged():
    client = FakeClient(_tool_call_then_final("fetch", {"url": "x"}))
    classifier = FakeClassifier([0.1, 0.6])
    approval_sink = InMemorySink()

    loop = GuardedOpenAIToolLoop(
        client=client,
        model="gpt-4o-mini",
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
