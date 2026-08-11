-- Phase 12: the Postgres backing store for the human-approval queue and the
-- intercepted-traffic log. Phase 6's ApprovalQueue and Phase 2's
-- JsonlFileSink/InMemorySink stay exactly as they are for the pip library
-- (a single-process demo or script doesn't need a database) -- this schema
-- backs the NEW service-only implementations (postgres_approval_queue.py,
-- postgres_log_sink.py) that the FastAPI backend uses instead, so multiple
-- reviewers/processes see the same durable, shared state.

CREATE TABLE IF NOT EXISTS pending_approvals (
    id UUID PRIMARY KEY,
    direction TEXT NOT NULL CHECK (direction IN ('request', 'result')),
    payload JSONB NOT NULL,
    reason TEXT NOT NULL,
    score DOUBLE PRECISION,
    created_at_ms BIGINT NOT NULL
);

-- A resolved approval is deleted from pending_approvals and inserted here --
-- mirrors ApprovalQueue's in-memory behavior (Phase 6: pending items move
-- out of the pending dict on resolve(), never mutated in place) so the two
-- implementations behave identically from EnforcementEngine's point of view.
CREATE TABLE IF NOT EXISTS approval_resolutions (
    pending_id UUID PRIMARY KEY,
    direction TEXT NOT NULL CHECK (direction IN ('request', 'result')),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'denied')),
    decided_by TEXT NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

-- One row per intercepted request or result -- the durable equivalent of
-- Phase 2's JsonlFileSink, queryable by the dashboard instead of tailed as
-- a file. `record` is the same dict shape ToolCallRequest.to_dict() /
-- ToolCallResult.to_dict() already produce, stored as-is rather than
-- normalized into columns, so this table never needs a migration when
-- those shapes gain a field.
CREATE TABLE IF NOT EXISTS traffic_events (
    id BIGSERIAL PRIMARY KEY,
    event TEXT NOT NULL,
    record JSONB NOT NULL,
    created_at_ms BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS traffic_events_created_at_idx ON traffic_events (created_at_ms DESC);
