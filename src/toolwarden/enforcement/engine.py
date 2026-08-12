"""Ties PolicyEngine + ApprovalQueue into one evaluate -> (maybe hold) ->
resolve -> final-decision cycle. See docs/architecture.md's component table:
the enforcement engine maps scores to decisions and owns the approval queue,
but never self-certifies a held action as safe — that always requires
apply_resolution() with an attributed human decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolwarden.enforcement.approval_queue import ApprovalDecision, ApprovalQueue
from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine
from toolwarden.models import Explanation


@dataclass
class EnforcementResult:
    decision: Decision
    reason: str
    pending_id: str | None = None
    explanation: Explanation | None = None


class EnforcementEngine:
    def __init__(self, policy: PolicyEngine, approval_queue: ApprovalQueue) -> None:
        self.policy = policy
        self.approval_queue = approval_queue

    def evaluate(
        self,
        direction: Direction,
        payload: dict[str, Any],
        score: float | None,
        explanation: Explanation | None = None,
    ) -> EnforcementResult:
        """score is the classifier's P(injection). Pass None on a classifier
        timeout — fail-closed to HOLD, not fail-open (see policy.py).

        explanation (Phase 5's SHAP/attention output, Classifier.score_and_explain())
        is attached to the result here — the single shared point every
        consumer (guarded_loop.py, mcp_proxy/server.py, service/app.py)
        inherits it from, per the Phase 13 audit finding that explainability
        was built but never wired into any live decision path. Optional:
        callers that only need score() (e.g. Phase 8's benchmark, which
        never shows a decision to a human) aren't forced to pay for it.
        """
        outcome = self.policy.decide_on_timeout(direction) if score is None else self.policy.decide(direction, score)

        if outcome.decision is not Decision.HOLD:
            return EnforcementResult(decision=outcome.decision, reason=outcome.reason, explanation=explanation)

        pending = self.approval_queue.submit(direction, payload, outcome.reason, outcome.score, explanation=explanation)
        return EnforcementResult(
            decision=Decision.HOLD, reason=outcome.reason, pending_id=pending.id, explanation=explanation
        )

    def apply_resolution(
        self, pending_id: str, decision: ApprovalDecision, decided_by: str, notes: str = ""
    ) -> EnforcementResult:
        record = self.approval_queue.resolve(pending_id, decision, decided_by, notes)
        final_decision = self.approval_queue.outcome_for(record)
        return EnforcementResult(
            decision=final_decision,
            reason=f"human resolution: {decision.value} by {decided_by}",
            pending_id=pending_id,
            explanation=record.explanation,
        )
