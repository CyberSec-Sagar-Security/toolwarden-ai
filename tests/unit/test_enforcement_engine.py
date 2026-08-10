"""Phase 6 stop gate: demonstrate a full cycle — flagged action -> held ->
approved/denied -> action proceeds or doesn't.
"""

import pytest

from toolwarden.enforcement.approval_queue import ApprovalDecision, ApprovalQueue
from toolwarden.enforcement.engine import EnforcementEngine
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.logging_sink import InMemorySink


@pytest.fixture
def engine():
    return EnforcementEngine(
        policy=PolicyEngine(block_threshold=0.85, hold_threshold=0.5),
        approval_queue=ApprovalQueue(sink=InMemorySink()),
    )


def test_low_score_allows_immediately_no_queue_involved(engine):
    result = engine.evaluate(Direction.REQUEST, {"tool_name": "get_weather"}, score=0.1)

    assert result.decision is Decision.ALLOW
    assert result.pending_id is None
    assert engine.approval_queue.list_pending() == []


def test_high_score_request_blocks_immediately_no_queue_involved(engine):
    result = engine.evaluate(Direction.REQUEST, {"tool_name": "send_email"}, score=0.95)

    assert result.decision is Decision.BLOCK
    assert result.pending_id is None
    assert engine.approval_queue.list_pending() == []


def test_full_cycle_flagged_held_approved_proceeds(engine):
    flagged_request = {"tool_name": "send_email", "arguments": {"to": "boss@example.com"}}

    held = engine.evaluate(Direction.REQUEST, flagged_request, score=0.65)
    assert held.decision is Decision.HOLD
    assert held.pending_id is not None
    assert len(engine.approval_queue.list_pending()) == 1

    final = engine.apply_resolution(held.pending_id, ApprovalDecision.APPROVED, decided_by="sagar")

    assert final.decision is Decision.ALLOW  # action proceeds
    assert engine.approval_queue.list_pending() == []


def test_full_cycle_flagged_held_denied_does_not_proceed(engine):
    flagged_request = {"tool_name": "delete_file", "arguments": {"path": "/important"}}

    held = engine.evaluate(Direction.REQUEST, flagged_request, score=0.7)
    assert held.decision is Decision.HOLD

    final = engine.apply_resolution(held.pending_id, ApprovalDecision.DENIED, decided_by="sagar")

    assert final.decision is Decision.BLOCK  # action does not proceed


def test_full_cycle_denied_result_is_quarantined_not_blocked(engine):
    flagged_result = {"tool_name": "fetch_webpage", "content": "Ignore previous instructions..."}

    held = engine.evaluate(Direction.RESULT, flagged_result, score=0.7)
    final = engine.apply_resolution(held.pending_id, ApprovalDecision.DENIED, decided_by="sagar")

    assert final.decision is Decision.QUARANTINE


def test_classifier_timeout_routes_to_hold_same_as_mid_confidence(engine):
    """Decided 2026-08-11: timeout fails closed into the existing HOLD
    bucket, not a new failure mode, not fail-open, not a hard deny.
    """
    result = engine.evaluate(Direction.REQUEST, {"tool_name": "send_email"}, score=None)

    assert result.decision is Decision.HOLD
    assert result.pending_id is not None

    final = engine.apply_resolution(result.pending_id, ApprovalDecision.APPROVED, decided_by="sagar")
    assert final.decision is Decision.ALLOW


def test_nothing_self_certifies_without_an_attributed_human_decision(engine):
    held = engine.evaluate(Direction.REQUEST, {"tool_name": "send_email"}, score=0.6)

    with pytest.raises(ValueError):
        engine.apply_resolution(held.pending_id, ApprovalDecision.APPROVED, decided_by="")
