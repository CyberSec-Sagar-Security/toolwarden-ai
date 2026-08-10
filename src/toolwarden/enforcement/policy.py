"""Policy layer: classifier score -> allow / block / quarantine / hold.

Two directions get different high-risk actions, per docs/architecture.md:
outbound tool-call REQUESTS can be stopped before they execute (BLOCK);
inbound tool-call RESULTS have already happened, so high risk there means
isolating/neutralizing the content (QUARANTINE) rather than a no-op
"block". Both directions share the same HOLD band for human review.

Classifier-timeout handling (decided 2026-08-11, see docs/known_limitations.md):
fail-closed into HOLD, not fail-open and not a hard deny. Fail-open would
defeat the project's enforcement thesis; a hard deny-on-timeout would break
the agent on ordinary latency, not just attacks. Timeout is treated as
"uncertain" and routed into the existing mid-confidence bucket rather than
adding a new failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Locked sane default per the build spec: auto-block high-confidence,
# hold mid-confidence. Configurable per PolicyEngine instance.
DEFAULT_BLOCK_THRESHOLD = 0.85
DEFAULT_HOLD_THRESHOLD = 0.5


class Direction(str, Enum):
    REQUEST = "request"  # outbound: agent -> tool
    RESULT = "result"  # inbound: tool -> agent


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    HOLD = "hold"


@dataclass
class PolicyOutcome:
    decision: Decision
    reason: str
    score: float | None  # None when the decision came from a timeout


class PolicyEngine:
    def __init__(
        self,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        hold_threshold: float = DEFAULT_HOLD_THRESHOLD,
    ) -> None:
        if not 0 <= hold_threshold <= block_threshold <= 1:
            raise ValueError("require 0 <= hold_threshold <= block_threshold <= 1")
        self.block_threshold = block_threshold
        self.hold_threshold = hold_threshold

    def decide(self, direction: Direction, score: float) -> PolicyOutcome:
        if not 0 <= score <= 1:
            raise ValueError(f"score must be in [0, 1], got {score}")

        if score >= self.block_threshold:
            high_risk_decision = Decision.BLOCK if direction is Direction.REQUEST else Decision.QUARANTINE
            return PolicyOutcome(high_risk_decision, f"score {score:.3f} >= block_threshold {self.block_threshold}", score)

        if score >= self.hold_threshold:
            return PolicyOutcome(Decision.HOLD, f"score {score:.3f} >= hold_threshold {self.hold_threshold}", score)

        return PolicyOutcome(Decision.ALLOW, f"score {score:.3f} < hold_threshold {self.hold_threshold}", score)

    def decide_on_timeout(self, direction: Direction) -> PolicyOutcome:
        return PolicyOutcome(Decision.HOLD, "classifier timeout: fail-closed to hold, not fail-open", None)
