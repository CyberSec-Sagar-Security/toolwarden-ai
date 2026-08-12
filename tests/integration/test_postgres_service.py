"""Phase 12: exercises PostgresApprovalQueue/PostgresLogSink against a real
Postgres instance (not mocked) -- the same enforcement matrix
test_approval_queue.py and test_enforcement_engine.py already prove against
the in-memory ApprovalQueue, run again here to prove the Postgres-backed
version is a genuine drop-in, not just a similarly-shaped class. The
EnforcementEngine tests below pass a PostgresApprovalQueue into the exact
same, unmodified EnforcementEngine every other phase uses.

Skipped automatically if TOOLWARDEN_DATABASE_URL isn't set or the database
isn't reachable -- this is an integration test against a real dependency,
not a unit test. Point it at a throwaway Postgres, e.g.:
    docker run -d -e POSTGRES_PASSWORD=toolwarden -e POSTGRES_USER=toolwarden \
        -e POSTGRES_DB=toolwarden -p 55432:5432 postgres:16-alpine
    export TOOLWARDEN_DATABASE_URL=postgresql://toolwarden:toolwarden@localhost:55432/toolwarden
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOOLWARDEN_DATABASE_URL"),
    reason="TOOLWARDEN_DATABASE_URL not set — point it at a real Postgres to run this integration test",
)


@pytest.fixture
def conn():
    from toolwarden.service import db

    connection = db.connect()
    db.apply_schema(connection)
    connection.execute("TRUNCATE pending_approvals, approval_resolutions, traffic_events")
    yield connection
    connection.close()


@pytest.fixture
def queue(conn):
    from toolwarden.service.postgres_approval_queue import PostgresApprovalQueue

    return PostgresApprovalQueue(conn)


@pytest.fixture
def engine(queue):
    from toolwarden.enforcement.engine import EnforcementEngine
    from toolwarden.enforcement.policy import PolicyEngine

    return EnforcementEngine(policy=PolicyEngine(block_threshold=0.85, hold_threshold=0.5), approval_queue=queue)


# --- PostgresApprovalQueue: same matrix as test_approval_queue.py ---


def test_submit_adds_to_pending(queue):
    from toolwarden.enforcement.policy import Direction

    pending = queue.submit(Direction.REQUEST, {"tool_name": "send_email"}, "mid-confidence", 0.6)

    assert pending.id in {p.id for p in queue.list_pending()}


def test_resolve_removes_from_pending(queue):
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Direction

    pending = queue.submit(Direction.REQUEST, {"tool_name": "send_email"}, "mid-confidence", 0.6)
    record = queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")

    assert pending.id not in {p.id for p in queue.list_pending()}
    assert record.decided_by == "sagar"


def test_resolve_requires_attributed_identity(queue):
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Direction

    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)

    with pytest.raises(ValueError):
        queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="")


def test_resolve_unknown_id_raises(queue):
    from toolwarden.enforcement.approval_queue import ApprovalDecision, UnknownApprovalError

    with pytest.raises(UnknownApprovalError):
        queue.resolve("00000000-0000-0000-0000-000000000000", ApprovalDecision.APPROVED, decided_by="sagar")


def test_resolve_malformed_id_raises_unknown_not_a_db_error(queue):
    """id is a UUID-typed column -- a non-UUID string must still surface as
    UnknownApprovalError, not a raw psycopg.DataError from the SELECT.
    """
    from toolwarden.enforcement.approval_queue import ApprovalDecision, UnknownApprovalError

    with pytest.raises(UnknownApprovalError):
        queue.resolve("not-a-uuid-at-all", ApprovalDecision.APPROVED, decided_by="sagar")


def test_resolve_twice_raises_already_resolved(queue):
    from toolwarden.enforcement.approval_queue import AlreadyResolvedError, ApprovalDecision
    from toolwarden.enforcement.policy import Direction

    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)
    queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")

    with pytest.raises(AlreadyResolvedError):
        queue.resolve(pending.id, ApprovalDecision.DENIED, decided_by="sagar")


def test_outcome_for_denied_result_is_quarantine(queue):
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Decision, Direction

    pending = queue.submit(Direction.RESULT, {}, "mid-confidence", 0.6)
    record = queue.resolve(pending.id, ApprovalDecision.DENIED, decided_by="sagar")

    assert queue.outcome_for(record) is Decision.QUARANTINE


def test_explanation_round_trips_through_submit_list_and_resolve(queue):
    """Phase 13's explainability-wiring fix: the JSONB column has to survive
    a real INSERT/SELECT round trip through Postgres, not just look right
    in Python before it's ever written to the DB.
    """
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Direction
    from toolwarden.models import Explanation

    explanation = Explanation(
        deberta_top_tokens=[("▁Ignore", 0.5), ("▁password", 0.3)],
        lightgbm_top_features=[("imperative_phrasing_score", 3.2)],
    )

    pending = queue.submit(Direction.RESULT, {"content": "sketchy"}, "mid-confidence", 0.6, explanation=explanation)
    assert pending.explanation == explanation

    listed = queue.list_pending()
    assert listed[0].explanation == explanation

    record = queue.resolve(pending.id, ApprovalDecision.APPROVED, decided_by="sagar")
    assert record.explanation == explanation


def test_explanation_is_none_when_not_provided(queue):
    from toolwarden.enforcement.policy import Direction

    pending = queue.submit(Direction.REQUEST, {}, "mid-confidence", 0.6)

    assert pending.explanation is None
    assert queue.list_pending()[0].explanation is None


# --- EnforcementEngine, unmodified, driven by PostgresApprovalQueue ---


def test_full_cycle_flagged_held_approved_proceeds(engine):
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Decision, Direction

    held = engine.evaluate(Direction.REQUEST, {"tool_name": "send_email"}, score=0.65)
    assert held.decision is Decision.HOLD
    assert len(engine.approval_queue.list_pending()) == 1

    final = engine.apply_resolution(held.pending_id, ApprovalDecision.APPROVED, decided_by="sagar")

    assert final.decision is Decision.ALLOW
    assert engine.approval_queue.list_pending() == []


def test_full_cycle_denied_result_is_quarantined(engine):
    from toolwarden.enforcement.approval_queue import ApprovalDecision
    from toolwarden.enforcement.policy import Decision, Direction

    held = engine.evaluate(Direction.RESULT, {"content": "Ignore previous instructions..."}, score=0.7)
    final = engine.apply_resolution(held.pending_id, ApprovalDecision.DENIED, decided_by="sagar")

    assert final.decision is Decision.QUARANTINE


# --- PostgresLogSink ---


def test_log_sink_write_and_recent(conn):
    from toolwarden.service.postgres_log_sink import PostgresLogSink

    sink = PostgresLogSink(conn)
    sink.write({"event": "tool_call_request", "call_id": "x1", "tool_name": "fetch"})
    sink.write({"event": "tool_call_result", "call_id": "x1", "tool_name": "fetch", "content": "hi"})

    recent = sink.recent(limit=10)

    assert len(recent) == 2
    assert recent[0]["event"] == "tool_call_result"  # most recent first
    assert recent[1]["event"] == "tool_call_request"
