"""Phase 12: a Postgres-backed drop-in for Phase 6's ApprovalQueue.

Duck-type compatible, not a subclass: same four methods
(submit/list_pending/resolve/outcome_for), same argument and return
shapes (PendingApproval, ApprovalRecord, ApprovalDecision -- all reused
directly from enforcement.approval_queue, not redefined), same exceptions
(UnknownApprovalError, AlreadyResolvedError). EnforcementEngine calls
these methods without knowing or caring which backing store is behind
them -- exactly the reuse this class exists to make possible: the FastAPI
backend passes a PostgresApprovalQueue into the same, unmodified
EnforcementEngine every other phase already uses.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from toolwarden.enforcement.approval_queue import (
    AlreadyResolvedError,
    ApprovalDecision,
    ApprovalRecord,
    PendingApproval,
    UnknownApprovalError,
    resolution_to_decision,
)
from toolwarden.enforcement.policy import Decision, Direction
from toolwarden.models import Explanation


def _explanation_json(explanation: Explanation | None) -> Jsonb | None:
    return None if explanation is None else Jsonb(explanation.to_dict())


def _explanation_from_row(value: dict | None) -> Explanation | None:
    return None if value is None else Explanation.from_dict(value)


def _now_ms() -> int:
    return int(time.time() * 1000)


class PostgresApprovalQueue:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def submit(
        self,
        direction: Direction,
        payload: dict[str, Any],
        reason: str,
        score: float | None,
        explanation: Explanation | None = None,
    ) -> PendingApproval:
        pending = PendingApproval(direction=direction, payload=payload, reason=reason, score=score, explanation=explanation)
        self.conn.execute(
            "INSERT INTO pending_approvals (id, direction, payload, reason, score, created_at_ms, explanation) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                pending.id,
                pending.direction.value,
                Jsonb(pending.payload),
                pending.reason,
                pending.score,
                pending.created_at_ms,
                _explanation_json(pending.explanation),
            ),
        )
        return pending

    def list_pending(self) -> list[PendingApproval]:
        rows = self.conn.execute(
            "SELECT id, direction, payload, reason, score, created_at_ms, explanation "
            "FROM pending_approvals ORDER BY created_at_ms"
        ).fetchall()
        return [
            PendingApproval(
                id=str(row[0]),
                direction=Direction(row[1]),
                payload=row[2],
                reason=row[3],
                score=row[4],
                created_at_ms=row[5],
                explanation=_explanation_from_row(row[6]),
            )
            for row in rows
        ]

    def resolve(self, pending_id: str, decision: ApprovalDecision, decided_by: str, notes: str = "") -> ApprovalRecord:
        if not decided_by:
            raise ValueError("decided_by is required — approvals must be attributed, not anonymous")

        try:
            uuid.UUID(pending_id)
        except ValueError as exc:
            # Validated in Python, not left to the DB: the id column is
            # UUID-typed, so a malformed id would otherwise raise a raw
            # psycopg.DataError from the SELECT below instead of the same
            # UnknownApprovalError a truly-missing-but-well-formed id gets —
            # an HTTP caller shouldn't see a 500 for a typo'd id.
            raise UnknownApprovalError(f"no pending approval with id {pending_id}") from exc

        row = self.conn.execute(
            "SELECT direction, explanation FROM pending_approvals WHERE id = %s", (pending_id,)
        ).fetchone()
        if row is None:
            already = self.conn.execute(
                "SELECT 1 FROM approval_resolutions WHERE pending_id = %s", (pending_id,)
            ).fetchone()
            if already is not None:
                raise AlreadyResolvedError(f"{pending_id} was already resolved")
            raise UnknownApprovalError(f"no pending approval with id {pending_id}")

        record = ApprovalRecord(
            pending_id=pending_id,
            direction=Direction(row[0]),
            decision=decision,
            decided_by=decided_by,
            notes=notes,
            explanation=_explanation_from_row(row[1]),
        )
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO approval_resolutions (pending_id, direction, decision, decided_by, decided_at_ms, notes, explanation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    record.pending_id,
                    record.direction.value,
                    record.decision.value,
                    record.decided_by,
                    record.decided_at_ms,
                    record.notes,
                    _explanation_json(record.explanation),
                ),
            )
            self.conn.execute("DELETE FROM pending_approvals WHERE id = %s", (pending_id,))
        return record

    def outcome_for(self, record: ApprovalRecord) -> Decision:
        return resolution_to_decision(record)
