"""Connection + schema helpers for the Phase 12 Postgres backing store.

Sync psycopg (not asyncpg), deliberately: PostgresApprovalQueue and
PostgresLogSink need to be drop-in, duck-typed replacements for Phase 6's
ApprovalQueue and Phase 2's LogSink, both of which EnforcementEngine and
Interceptor call synchronously (`self.approval_queue.submit(...)`, no
`await`). An async driver would force EnforcementEngine itself to become
async -- a much bigger, riskier change to code every other phase already
depends on, for no real benefit (FastAPI runs sync route handlers in a
threadpool automatically, so this doesn't cost the API its async request
handling).
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def database_url() -> str:
    url = os.environ.get("TOOLWARDEN_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "TOOLWARDEN_DATABASE_URL is not set -- the Phase 12 service needs a Postgres connection "
            "string, e.g. postgresql://toolwarden:toolwarden@localhost:5432/toolwarden"
        )
    return url


def connect() -> psycopg.Connection:
    conn = psycopg.connect(database_url())
    conn.autocommit = True
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
