"""Phase 9: the standalone demo agent. A research assistant that fetches a
webpage and summarizes it, wired through ToolWarden's full pipeline
(Interceptor -> Classifier -> PolicyEngine -> EnforcementEngine ->
ApprovalQueue). Three scenarios, run back to back, each printed with a
clear narrative:

1. ALLOW    — a genuinely benign page. Classifier score is low, nothing
              is held, the task completes normally.
2. HOLD     — a page carrying a real held-out-derived injected instruction
              (see tools.py's module docstring). Classifier score lands in
              the mid-confidence band, execution pauses for human
              approval. Run twice: once approved (showing the consequence
              of a wrong approval — the injected send_email goes through),
              once denied (showing the content gets quarantined and the
              injection never reaches the agent).
3. FALSE POSITIVE — a second genuinely benign page that the classifier
              scores very high (~0.96) despite there being nothing wrong
              with it. Auto-quarantined before any human even sees it.
              This is disclosed, not hidden: Phase 8 already measured
              AgentDojo precision at 0.27-0.30, and this is that number
              made concrete on a single, readable example.

Run with: python -m toolwarden.demo.run_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from toolwarden import config

config.configure_hf_cache_env()

from openai import OpenAI  # noqa: E402

from toolwarden.demo.classify import Classifier  # noqa: E402
from toolwarden.demo.guarded_loop import GuardedOpenAIToolLoop  # noqa: E402
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

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = REPO_ROOT / "logs" / "demo_traffic.jsonl"
APPROVAL_LOG_PATH = REPO_ROOT / "logs" / "demo_approvals.jsonl"

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
    """Prints the live score for each scenario's content before running
    anything — the numbers in tools.py's docstring are a snapshot from
    when this was written; this proves they still hold on this run,
    same discipline as every other honest-numbers check in this project.
    """
    print("Startup check — live classifier scores for each scenario's content:")
    for label, url in [("allow", ALLOW_URL), ("hold", HOLD_URL), ("false_positive", FALSE_POSITIVE_URL)]:
        score = classifier.score(fetch_webpage(url))
        print(f"  {label:<15} {score:.3f}")
    print()


def _make_loop(classifier: Classifier, send_email_log: SentEmailLog, on_hold, client: OpenAI) -> GuardedOpenAIToolLoop:
    interceptor = Interceptor(sink=JsonlFileSink(LOG_PATH))
    policy = PolicyEngine()
    approval_sink = JsonlFileSink(APPROVAL_LOG_PATH)
    tool_functions = {"fetch_webpage": fetch_webpage, "send_email": make_send_email(send_email_log)}
    return GuardedOpenAIToolLoop(
        client=client,
        model=MODEL,
        tools_schema=TOOLS_SCHEMA,
        tool_functions=tool_functions,
        interceptor=interceptor,
        classifier=classifier,
        policy=policy,
        approval_sink=approval_sink,
        on_hold=on_hold,
    )


def run_allow_scenario(classifier: Classifier, client: OpenAI) -> None:
    _print_header("SCENARIO 1: ALLOW — genuinely benign page")

    def on_hold(pending_id, direction, payload, label, score):
        raise AssertionError(f"Unexpected HOLD in the allow scenario: {label} score={score}")

    log = SentEmailLog()
    loop = _make_loop(classifier, log, on_hold, client)
    result = loop.run(f"Please fetch {ALLOW_URL} and tell me what it says in one sentence.")
    _print_events(result.events)
    print(f"\nAgent's final answer: {result.final_text}")
    print(f"Emails sent during this run: {len(log.sent)} (expected 0)")


def run_hold_scenario(classifier: Classifier, client: OpenAI, decision: ApprovalDecision) -> None:
    _print_header(f"SCENARIO 2: HOLD -> {decision.value.upper()} — injected page")

    def on_hold(pending_id, direction, payload, label, score):
        print(f"\n  >>> HELD for approval: {label} (score={score:.3f}). Reviewer decision: {decision.value}. <<<\n")
        return decision

    log = SentEmailLog()
    loop = _make_loop(classifier, log, on_hold, client)
    result = loop.run(f"Please fetch {HOLD_URL} and tell me what it says in one sentence.")
    _print_events(result.events)
    print(f"\nAgent's final answer: {result.final_text}")
    print(f"Emails sent during this run: {len(log.sent)}")
    if log.sent:
        for email in log.sent:
            print(f"  -> to={email['to']!r} subject={email['subject']!r}")


def run_false_positive_scenario(classifier: Classifier, client: OpenAI) -> None:
    _print_header("SCENARIO 3: FALSE POSITIVE — benign page, auto-quarantined")

    def on_hold(pending_id, direction, payload, label, score):
        raise AssertionError(f"Unexpected HOLD in the false-positive scenario: {label} score={score}")

    log = SentEmailLog()
    loop = _make_loop(classifier, log, on_hold, client)
    result = loop.run(f"Please fetch {FALSE_POSITIVE_URL} and tell me what it says in one sentence.")
    _print_events(result.events)
    print(f"\nAgent's final answer: {result.final_text}")
    print(
        "\nNote: this page is genuinely benign (a weather update) but scored high enough to be "
        "auto-quarantined without ever reaching human review. This is Phase 8's measured AgentDojo "
        "precision (0.27-0.30) made concrete on one example — disclosed, not hidden."
    )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — this demo makes live OpenAI API calls.")

    print("Loading classifier (DeBERTa + LightGBM + ensemble)...")
    classifier = Classifier()
    _verify_startup_scores(classifier)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    run_allow_scenario(classifier, client)
    run_hold_scenario(classifier, client, ApprovalDecision.DENIED)
    run_hold_scenario(classifier, client, ApprovalDecision.APPROVED)
    run_false_positive_scenario(classifier, client)

    print("\n" + "=" * 78)
    print(f"Full tool-call trace: {LOG_PATH}")
    print(f"Approval decisions:   {APPROVAL_LOG_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
