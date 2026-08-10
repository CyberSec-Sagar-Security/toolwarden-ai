import pytest

from toolwarden.enforcement.policy import Decision, Direction, PolicyEngine


@pytest.fixture
def policy():
    return PolicyEngine(block_threshold=0.85, hold_threshold=0.5)


def test_low_score_request_is_allowed(policy):
    outcome = policy.decide(Direction.REQUEST, 0.1)
    assert outcome.decision is Decision.ALLOW


def test_mid_score_request_is_held(policy):
    outcome = policy.decide(Direction.REQUEST, 0.6)
    assert outcome.decision is Decision.HOLD


def test_high_score_request_is_blocked(policy):
    outcome = policy.decide(Direction.REQUEST, 0.9)
    assert outcome.decision is Decision.BLOCK


def test_high_score_result_is_quarantined_not_blocked(policy):
    """Results already happened — a block doesn't make sense, quarantine does."""
    outcome = policy.decide(Direction.RESULT, 0.9)
    assert outcome.decision is Decision.QUARANTINE


def test_mid_score_result_is_held(policy):
    outcome = policy.decide(Direction.RESULT, 0.6)
    assert outcome.decision is Decision.HOLD


def test_low_score_result_is_allowed(policy):
    outcome = policy.decide(Direction.RESULT, 0.1)
    assert outcome.decision is Decision.ALLOW


def test_score_exactly_at_thresholds_is_inclusive(policy):
    assert policy.decide(Direction.REQUEST, 0.5).decision is Decision.HOLD
    assert policy.decide(Direction.REQUEST, 0.85).decision is Decision.BLOCK


def test_score_out_of_range_raises(policy):
    with pytest.raises(ValueError):
        policy.decide(Direction.REQUEST, 1.5)
    with pytest.raises(ValueError):
        policy.decide(Direction.REQUEST, -0.1)


def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError):
        PolicyEngine(block_threshold=0.3, hold_threshold=0.5)


def test_timeout_fails_closed_to_hold_not_open_not_deny(policy):
    """The decided policy (2026-08-11): timeout is 'uncertain', routed to
    HOLD — not fail-open (defeats enforcement), not a hard deny (breaks the
    agent on ordinary latency).
    """
    request_outcome = policy.decide_on_timeout(Direction.REQUEST)
    result_outcome = policy.decide_on_timeout(Direction.RESULT)

    assert request_outcome.decision is Decision.HOLD
    assert result_outcome.decision is Decision.HOLD
    assert request_outcome.score is None
    assert "timeout" in request_outcome.reason.lower()
