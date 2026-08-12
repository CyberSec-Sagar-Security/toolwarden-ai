"""Phase 13 follow-up: Phase 9's exact three scenarios, run against Claude
instead of OpenAI, to actually demonstrate the Anthropic side of the
"direct API function-calling (OpenAI/Anthropic tool_use)" locked scope end
to end -- not just that GuardedAnthropicToolLoop imports cleanly. Same
canned content (src/toolwarden/demo/tools.py), same classifier, same
thresholds, same three scenarios as docs/demo_walkthrough.md:

1. ALLOW    — a genuinely benign page. Classifier score is low, nothing
              is held, the task completes normally.
2. HOLD     — a page carrying a real held-out-derived injected instruction.
              Run twice: denied (quarantined before Claude ever sees it)
              and approved (the injected send_email goes through).
3. FALSE POSITIVE — a second genuinely benign page that the classifier
              scores very high despite there being nothing wrong with it.
              Auto-quarantined before any human even sees it.

Run with: python -m toolwarden.demo.run_demo_anthropic
"""

from __future__ import annotations

import os
from pathlib import Path

from toolwarden import config

config.configure_hf_cache_env()

from anthropic import Anthropic  # noqa: E402

from toolwarden.demo.classify import Classifier  # noqa: E402
from toolwarden.demo.guarded_loop_anthropic import GuardedAnthropicToolLoop  # noqa: E402
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

# cwd-relative, not __file__-relative -- see demo/run_demo.py's identical
# comment: a __file__-relative path resolves into site-packages for a
# pip-installed run, not a sensible log location. Separate files from the
# OpenAI/MCP demos so all three adapters' runs stay independently inspectable.
LOG_PATH = Path.cwd() / "logs" / "anthropic_demo_traffic.jsonl"
APPROVAL_LOG_PATH = Path.cwd() / "logs" / "anthropic_demo_approvals.jsonl"

MODEL = "claude-haiku-4-5"


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _format_explanation(explanation) -> str:
    deberta = ", ".join(f"{tok!r}={weight:.3f}" for tok, weight in explanation.deberta_top_tokens[:3])
    if explanation.lightgbm_top_features is None:
        return f"          deberta top tokens: {deberta}  (lightgbm: n/a, deberta_only mode)"
    lightgbm = ", ".join(f"{feat}={val:+.3f}" for feat, val in explanation.lightgbm_top_features[:3])
    return f"          deberta top tokens: {deberta}\n          lightgbm top features: {lightgbm}"


def _print_events(events) -> None:
    for e in events:
        line = f"  [{e.direction:>7}] {e.label:<28} score={e.score:.3f} -> {e.decision}"
        if e.resolution:
            line += f"  (human: {e.resolution} -> final: {e.final_decision})"
        print(line)
        if e.explanation is not None and e.decision != "allow":
            print(_format_explanation(e.explanation))


def _verify_startup_scores(classifier: Classifier) -> None:
    print("Startup check — live classifier scores for each scenario's content:")
    for label, url in [("allow", ALLOW_URL), ("hold", HOLD_URL), ("false_positive", FALSE_POSITIVE_URL)]:
        score = classifier.score(fetch_webpage(url))
        print(f"  {label:<15} {score:.3f}")
    print()


def _make_loop(classifier: Classifier, send_email_log: SentEmailLog, on_hold, client: Anthropic) -> GuardedAnthropicToolLoop:
    interceptor = Interceptor(sink=JsonlFileSink(LOG_PATH))
    policy = PolicyEngine()
    approval_sink = JsonlFileSink(APPROVAL_LOG_PATH)
    tool_functions = {"fetch_webpage": fetch_webpage, "send_email": make_send_email(send_email_log)}
    return GuardedAnthropicToolLoop(
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


def run_allow_scenario(classifier: Classifier, client: Anthropic) -> None:
    _print_header("SCENARIO 1: ALLOW — genuinely benign page (via Claude)")

    def on_hold(pending_id, direction, payload, label, score):
        raise AssertionError(f"Unexpected HOLD in the allow scenario: {label} score={score}")

    log = SentEmailLog()
    loop = _make_loop(classifier, log, on_hold, client)
    result = loop.run(f"Please fetch {ALLOW_URL} and tell me what it says in one sentence.")
    _print_events(result.events)
    print(f"\nAgent's final answer: {result.final_text}")
    print(f"Emails sent during this run: {len(log.sent)} (expected 0)")


def run_hold_scenario(classifier: Classifier, client: Anthropic, decision: ApprovalDecision) -> None:
    _print_header(f"SCENARIO 2: HOLD -> {decision.value.upper()} — injected page (via Claude)")

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


def run_false_positive_scenario(classifier: Classifier, client: Anthropic) -> None:
    _print_header("SCENARIO 3: FALSE POSITIVE — benign page, auto-quarantined (via Claude)")

    def on_hold(pending_id, direction, payload, label, score):
        raise AssertionError(f"Unexpected HOLD in the false-positive scenario: {label} score={score}")

    log = SentEmailLog()
    loop = _make_loop(classifier, log, on_hold, client)
    result = loop.run(f"Please fetch {FALSE_POSITIVE_URL} and tell me what it says in one sentence.")
    _print_events(result.events)
    print(f"\nAgent's final answer: {result.final_text}")
    print(
        "\nNote: this page is genuinely benign (a weather update) but scored high enough to be "
        "auto-quarantined without ever reaching human review — same disclosed classifier weakness "
        "as Phase 9, this time reached through Claude's tool_use format instead of OpenAI's."
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — this demo makes live Anthropic API calls.")

    print("Loading classifier (DeBERTa + LightGBM + ensemble)...")
    classifier = Classifier()
    _verify_startup_scores(classifier)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
