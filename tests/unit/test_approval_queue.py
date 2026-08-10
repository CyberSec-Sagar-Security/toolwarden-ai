import pytest

from toolwarden.enforcement.approval_queue import (
    AlreadyResolvedError,
    ApprovalDecision,
    ApprovalQueue,
    UnknownApprovalError,
)
from toolwarden.enforcement.policy import Decision, Direction
from toolwarden.logging_sink import InMemorySink


@pytest.fixture
def queue():
    return ApprovalQueue(sink=InMemorySink())


def test_submit_adds_to_pending(queue):
    pending = queue.submit(Direction.REQUEST, {"tool_name": "send_email"}, "mid-confidence", 0.6)

    assert pending.id in {p.id for p in queue.list_pending()}


def test_resolve_removes_from_pending_and_logs(queue):
    pending = queue.submit(Direction.REQUEST, {"tool_name": "send_email"}, "mid-confidence", 0.6)

    record = queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")

    assert pending.id not in {p.id for p in queue.list_pending()}
    assert record.decided_by == "sagar"
    assert queue.sink.records[-1]["event"] == "approval_resolved"
    assert queue.sink.records[-1]["decided_by"] == "sagar"


def test_resolve_requires_attributed_identity(queue):
    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)

    with pytest.raises(ValueError):
        queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="")


def test_resolve_unknown_id_raises(queue):
    with pytest.raises(UnknownApprovalError):
        queue.resolve("nonexistent-id", ApprovalDecision.APPROVED, decided_by="sagar")


def test_resolve_twice_raises_already_resolved(queue):
    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)
    queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")

    with pytest.raises(AlreadyResolvedError):
        queue.resolve(pending.id, ApprovalDecision.DENIED, decided_by="sagar")


def test_outcome_for_approved_request_is_allow(queue):
    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)
    record = queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")

    assert queue.outcome_for(record) is Decision.ALLOW


def test_outcome_for_denied_request_is_block(queue):
    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)
    record = queue.resolve(pending.id, ApprovalDecision.DENIED, decided_by="sagar")

    assert queue.outcome_for(record) is Decision.BLOCK


def test_outcome_for_denied_result_is_quarantine(queue):
    pending = queue.submit(Direction.RESULT, {}, "mid-confidence", 0.6)
    record = queue.resolve(pending.id, ApprovalDecision.DENIED, decided_by="sagar")

    assert queue.outcome_for(record) is Decision.QUARANTINE
