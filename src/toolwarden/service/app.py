"""Phase 12: the FastAPI backend. Turns the pip library into a real
deployable service — an agent calls this over HTTP instead of importing
toolwarden directly, and a human reviewer resolves HOLDs through the
frontend instead of a synchronous on_hold() callback in a Python process.

Reuses, does not reimplement: Classifier (Phase 4/8's trained ensemble),
PolicyEngine/EnforcementEngine (Phase 6), Interceptor (Phase 2), the
ToolCallRequest/ToolCallResult shapes (Phase 2's models.py). The only new
code here is the HTTP surface and the Postgres-backed ApprovalQueue/LogSink
(postgres_approval_queue.py, postgres_log_sink.py) this phase adds.

One real architectural change from every other adapter (guarded_loop.py,
mcp_proxy/server.py): a HOLD here cannot block the HTTP request waiting for
a human, the way the demo's synchronous on_hold() callback does — there is
no human on the other end of an API call. A HOLD response returns
immediately with a pending_id; the caller (or a human via the frontend)
resolves it later via POST /v1/approvals/{pending_id}/resolve, and the
original tool-call caller is expected to poll or re-check before acting on
held content. This is the honest shape of a real async approval workflow,
not a simplification -- see docs/docker_walkthrough.md.

Run with: uvicorn toolwarden.service.app:app
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from toolwarden import config

config.configure_hf_cache_env()

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from toolwarden.demo.classify import DETECTOR_MODES, Classifier  # noqa: E402
from toolwarden.enforcement.approval_queue import (  # noqa: E402
    AlreadyResolvedError,
    ApprovalDecision,
    UnknownApprovalError,
)
from toolwarden.enforcement.engine import EnforcementEngine  # noqa: E402
from toolwarden.enforcement.policy import DEFAULT_BLOCK_THRESHOLD, DEFAULT_HOLD_THRESHOLD, Direction, PolicyEngine  # noqa: E402
from toolwarden.interceptor import Interceptor  # noqa: E402
from toolwarden.models import ToolCallRequest, ToolCallResult  # noqa: E402
from toolwarden.service import db  # noqa: E402
from toolwarden.service.postgres_approval_queue import PostgresApprovalQueue  # noqa: E402
from toolwarden.service.postgres_log_sink import PostgresLogSink  # noqa: E402


def _detector_mode() -> str:
    mode = os.environ.get("TOOLWARDEN_DETECTOR_MODE", "ensemble")
    if mode not in DETECTOR_MODES:
        raise RuntimeError(f"TOOLWARDEN_DETECTOR_MODE must be one of {DETECTOR_MODES}, got {mode!r}")
    return mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ConnectionPool(db.database_url(), min_size=1, max_size=10, open=True)
    app.state.pool.wait()
    with app.state.pool.connection() as conn:
        db.apply_schema(conn)

    app.state.classifier = Classifier(detector_mode=_detector_mode())
    app.state.policy = PolicyEngine(
        block_threshold=float(os.environ.get("TOOLWARDEN_BLOCK_THRESHOLD", DEFAULT_BLOCK_THRESHOLD)),
        hold_threshold=float(os.environ.get("TOOLWARDEN_HOLD_THRESHOLD", DEFAULT_HOLD_THRESHOLD)),
    )
    yield
    app.state.pool.close()


app = FastAPI(title="ToolWarden AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the frontend reaches this through nginx's reverse proxy, not the browser directly (see docker/frontend.Dockerfile) -- kept permissive for direct API use too
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn(request: Request):
    with request.app.state.pool.connection() as conn:
        yield conn


def get_engine(request: Request, conn=Depends(get_conn)) -> EnforcementEngine:
    return EnforcementEngine(policy=request.app.state.policy, approval_queue=PostgresApprovalQueue(conn))


def get_interceptor(conn=Depends(get_conn)) -> Interceptor:
    return Interceptor(sink=PostgresLogSink(conn))


class ToolCallRequestIn(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    source: str = "api"
    session_id: str
    call_id: str | None = None


class ToolCallResultIn(BaseModel):
    call_id: str
    tool_name: str
    content: Any
    is_error: bool = False


class ResolveIn(BaseModel):
    decision: Literal["approved", "denied"]
    decided_by: str
    notes: str = ""


class EnforcementOut(BaseModel):
    decision: str
    reason: str
    score: float | None
    pending_id: str | None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/tool-calls/request", response_model=EnforcementOut)
def submit_tool_call_request(
    body: ToolCallRequestIn, request: Request, interceptor: Interceptor = Depends(get_interceptor), engine: EnforcementEngine = Depends(get_engine)
):
    classifier: Classifier = request.app.state.classifier
    tc_request = interceptor.intercept_request(
        ToolCallRequest(
            tool_name=body.tool_name,
            arguments=body.arguments,
            source=body.source,
            session_id=body.session_id,
            call_id=body.call_id or uuid.uuid4().hex,
        )
    )
    score = classifier.score(json.dumps(body.arguments))
    outcome = engine.evaluate(Direction.REQUEST, tc_request.to_dict(), score)
    return EnforcementOut(decision=outcome.decision.value, reason=outcome.reason, score=score, pending_id=outcome.pending_id)


@app.post("/v1/tool-calls/result", response_model=EnforcementOut)
def submit_tool_call_result(
    body: ToolCallResultIn, request: Request, interceptor: Interceptor = Depends(get_interceptor), engine: EnforcementEngine = Depends(get_engine)
):
    classifier: Classifier = request.app.state.classifier
    tc_result = interceptor.intercept_result(
        ToolCallResult(call_id=body.call_id, tool_name=body.tool_name, content=body.content, is_error=body.is_error)
    )
    score = classifier.score(str(tc_result.content))
    outcome = engine.evaluate(Direction.RESULT, tc_result.to_dict(), score)
    return EnforcementOut(decision=outcome.decision.value, reason=outcome.reason, score=score, pending_id=outcome.pending_id)


@app.get("/v1/approvals/pending")
def list_pending_approvals(conn=Depends(get_conn)):
    queue = PostgresApprovalQueue(conn)
    return [
        {
            "id": p.id,
            "direction": p.direction.value,
            "payload": p.payload,
            "reason": p.reason,
            "score": p.score,
            "created_at_ms": p.created_at_ms,
        }
        for p in queue.list_pending()
    ]


@app.post("/v1/approvals/{pending_id}/resolve", response_model=EnforcementOut)
def resolve_approval(pending_id: str, body: ResolveIn, engine: EnforcementEngine = Depends(get_engine)):
    try:
        final = engine.apply_resolution(
            pending_id, ApprovalDecision(body.decision), decided_by=body.decided_by, notes=body.notes
        )
    except UnknownApprovalError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AlreadyResolvedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EnforcementOut(decision=final.decision.value, reason=final.reason, score=None, pending_id=pending_id)


@app.get("/v1/traffic")
def recent_traffic(limit: int = 50, conn=Depends(get_conn)):
    return PostgresLogSink(conn).recent(limit=limit)
