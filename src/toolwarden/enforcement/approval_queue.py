"""Human-approval queue: flagged actions wait for explicit sign-off.

Every resolution is logged with timestamp + decision + identity (see
docs/threat-model.md's repudiation section — an unattributed "approved"
flag isn't good enough). Nothing self-certifies: resolve() always requires
an explicit decided_by identity and always appends to the durable log
before returning, even when called in-process.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from toolwarden.enforcement.policy import Decision, Direction
from toolwarden.logging_sink import LogSink


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class PendingApproval:
    direction: Direction
    payload: dict[str, Any]
    reason: str
    score: float | None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_ms: int = field(default_factory=_now_ms)


@dataclass
class ApprovalRecord:
    pending_id: str
    direction: Direction
    decision: ApprovalDecision
    decided_by: str
    decided_at_ms: int = field(default_factory=_now_ms)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "approval_resolved",
            "pending_id": self.pending_id,
            "direction": self.direction.value,
            "decision": self.decision.value,
            "decided_by": self.decided_by,
            "decided_at_ms": self.decided_at_ms,
            "notes": self.notes,
        }


class UnknownApprovalError(KeyError):
    pass


class AlreadyResolvedError(RuntimeError):
    pass


class ApprovalQueue:
    """Pending items live in-memory (this is a library, not a service yet —
    Phase 12 moves this to Postgres). Resolutions are always appended to
    `sink` for a durable audit trail regardless of where pending state lives.
    """

    def __init__(self, sink: LogSink) -> None:
        self.sink = sink
        self._pending: dict[str, PendingApproval] = {}
        self._resolved: dict[str, ApprovalRecord] = {}

    def submit(self, direction: Direction, payload: dict[str, Any], reason: str, score: float | None) -> PendingApproval:
        pending = PendingApproval(direction=direction, payload=payload, reason=reason, score=score)
        self._pending[pending.id] = pending
        return pending

    def list_pending(self) -> list[PendingApproval]:
        return list(self._pending.values())

    def resolve(self, pending_id: str, decision: ApprovalDecision, decided_by: str, notes: str = "") -> ApprovalRecord:
        if pending_id not in self._pending:
            if pending_id in self._resolved:
                raise AlreadyResolvedError(f"{pending_id} was already resolved")
            raise UnknownApprovalError(f"no pending approval with id {pending_id}")
        if not decided_by:
            raise ValueError("decided_by is required — approvals must be attributed, not anonymous")

        pending = self._pending[pending_id]
        record = ApprovalRecord(
            pending_id=pending_id, direction=pending.direction, decision=decision, decided_by=decided_by, notes=notes
        )
        self.sink.write(record.to_dict())
        self._resolved[pending_id] = record
        del self._pending[pending_id]
        return record

    def outcome_for(self, record: ApprovalRecord) -> Decision:
        """Maps a human resolution back to the same Decision vocabulary the
        policy engine uses, so callers don't need two different result types.
        """
        if record.decision is ApprovalDecision.DENIED:
            return Decision.BLOCK if record.direction is Direction.REQUEST else Decision.QUARANTINE
        return Decision.ALLOW
