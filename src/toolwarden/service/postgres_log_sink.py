"""Phase 12: a Postgres-backed drop-in for Phase 2's LogSink protocol
(logging_sink.py) -- same single write(record) method JsonlFileSink and
InMemorySink already implement, so Interceptor (and anything else that
takes a LogSink) works unmodified against this backend instead of a file.
"""

from __future__ import annotations

import time
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def _now_ms() -> int:
    return int(time.time() * 1000)


class PostgresLogSink:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def write(self, record: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO traffic_events (event, record, created_at_ms) VALUES (%s, %s, %s)",
            (record.get("event", "unknown"), Jsonb(record), _now_ms()),
        )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record FROM traffic_events ORDER BY created_at_ms DESC LIMIT %s", (limit,)
        ).fetchall()
        return [row[0] for row in rows]
