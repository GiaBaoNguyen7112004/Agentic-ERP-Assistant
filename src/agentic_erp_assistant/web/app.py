"""The FastAPI application: every route, and the React app it serves.

The engine underneath every route is unaware this file exists -- nothing
here is imported by anything below :mod:`agentic_erp_assistant.composition`.
Each route is ``async def`` and does no blocking work itself; anything that
touches Postgres, Qdrant, or a model runs on a worker thread via
``loop.run_in_executor``, the same discipline
:mod:`agentic_erp_assistant.web.service` is built on (ADR 0015).
"""

import asyncio
import functools
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.users import UnknownUser
from agentic_erp_assistant.persistence.connection import StoreConnectionError
from agentic_erp_assistant.persistence.postgres_queries import EvidenceQueries
from agentic_erp_assistant.rag.access import RetrievalContext, is_authorized
from agentic_erp_assistant.rag.manifest import DEFAULT_MANIFEST_PATH
from agentic_erp_assistant.web.protocol import EVENT_TYPES, encode_sse, RunReportOut
from agentic_erp_assistant.web.service import ChatService, Conflict, Forbidden, NotFound
from agentic_erp_assistant.web.stream import TurnStream

__all__ = ["create_app", "MESSAGE_MAX_CHARS", "STATIC_DIR"]

logger = logging.getLogger(__name__)

MESSAGE_MAX_CHARS = 4000
"""A chat message longer than this is refused with 422 before any turn
starts -- generous for a typed question, tight enough that a client bug
pasting a whole document cannot spend a model call finding that out."""

STATIC_DIR = Path(__file__).resolve().parent / "static"
"""Where ``npm --prefix ui run build`` writes the React app. Git-ignored;
see ``ui/vite.config.ts``."""

SOURCES_DIR = DEFAULT_MANIFEST_PATH.parent / "sources"
"""Where a PDF manifest entry's markdown twin lives -- see
``/api/documents/{document_id}``."""

_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


class ChatRequest(BaseModel):
    """The body of ``POST /api/chat``."""

    actor: str = Field(min_length=1)
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=MESSAGE_MAX_CHARS)

    @field_validator("message")
    @classmethod
    def _not_blank_after_stripping(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class SessionRequest(BaseModel):
    actor: str = Field(min_length=1)


class ApprovalDecisionRequest(BaseModel):
    """The body of ``POST /api/approvals/{trace_id}``."""

    actor: str = Field(min_length=1)
    """Who is deciding -- the approver, never the original requester."""

    approved: bool


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


async def _sse_body(stream: TurnStream) -> AsyncIterator[bytes]:
    async for event in stream.events():
        yield encode_sse(event)


async def _run_blocking(func, /, **kwargs):
    """Run a synchronous, blocking call off the event loop.

    Every call into :class:`~agentic_erp_assistant.composition.resources.
    AppResources` or :class:`~agentic_erp_assistant.web.service.ChatService`
    does I/O the engine itself would block on; this is the one place that
    offload happens, so no route has to remember to do it separately.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, **kwargs))


def create_app(
    resources: AppResources | None = None, service: ChatService | None = None
) -> FastAPI:
    """Build the application.

    Args:
        resources: Injected for a test (``TestClient`` + fakes); ``None``
            builds :meth:`AppResources.from_env` at startup, in the
            lifespan, so importing this module never touches the network.
        service: Injected the same way; ``None`` wraps ``resources``.
    """
    owns_resources = resources is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal resources, service
        if resources is None:
            resources = AppResources.from_env()
        if service is None:
            service = ChatService(resources)
        app.state.resources = resources
        app.state.service = service
        try:
            yield
        finally:
            if owns_resources:
                resources.close()

    app = FastAPI(title="Agentic ERP Assistant", lifespan=lifespan)

    def get_resources(request: Request) -> AppResources:
        return request.app.state.resources

    def get_service(request: Request) -> ChatService:
        return request.app.state.service

    # -- health, users, protocol ---------------------------------------------

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        res = get_resources(request)
        try:
            connection = await _run_blocking(res.connect)
            connection.close()
            postgres = "ok"
        except StoreConnectionError:
            postgres = "down"
        return {
            "status": "ok",
            "model": res.settings.model,
            "chunks_indexed": len(res.retrieval.lexical_index),
            "postgres": postgres,
        }

    @app.get("/api/users")
    async def list_users(request: Request) -> list[dict]:
        res = get_resources(request)
        return [
            {
                "actor": user.actor,
                "display_name": user.display_name,
                "role": user.role,
                "project_code": user.project_code,
                "scopes": sorted(user.scopes),
                "can_approve": user.can_approve,
            }
            for user in res.users.all()
        ]

    @app.get("/api/protocol")
    async def protocol() -> dict:
        return {"event_types": list(EVENT_TYPES)}

    # -- sessions --------------------------------------------------------

    @app.post("/api/sessions")
    async def create_session(body: SessionRequest, request: Request) -> dict:
        res = get_resources(request)
        try:
            res.users.get(body.actor)
        except UnknownUser:
            raise HTTPException(404, f"unknown actor {body.actor!r}") from None
        return {"session_id": f"sess-{uuid.uuid4().hex[:12]}"}

    @app.get("/api/sessions")
    async def list_sessions(actor: str, request: Request):
        res = get_resources(request)

        def query():
            connection = res.connect()
            try:
                return EvidenceQueries(connection).sessions(actor)
            finally:
                connection.close()

        return await _run_blocking(query)

    @app.get("/api/sessions/{session_id}/turns")
    async def list_session_turns(session_id: str, actor: str, request: Request):
        res = get_resources(request)

        def query():
            connection = res.connect()
            try:
                return EvidenceQueries(connection).session_turns(session_id, actor)
            finally:
                connection.close()

        return await _run_blocking(query)

    # -- chat and approvals: SSE ------------------------------------------

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        svc = get_service(request)
        loop = asyncio.get_running_loop()
        stream = TurnStream(loop)
        loop.run_in_executor(
            None,
            functools.partial(
                svc.run_turn,
                actor=body.actor,
                session_id=body.session_id,
                message=body.message,
                stream=stream,
            ),
        )
        return StreamingResponse(
            _sse_body(stream), media_type="text/event-stream", headers=_sse_headers()
        )

    @app.get("/api/approvals")
    async def list_approvals(request: Request, actor: str | None = None):
        res = get_resources(request)

        def query():
            connection = res.connect()
            try:
                return EvidenceQueries(connection).pending_approvals(actor)
            finally:
                connection.close()

        return await _run_blocking(query)

    @app.post("/api/approvals/{trace_id}")
    async def decide_approval(
        trace_id: str, body: ApprovalDecisionRequest, request: Request
    ) -> StreamingResponse:
        svc = get_service(request)
        loop = asyncio.get_running_loop()

        try:
            pending = await _run_blocking(
                svc.precheck_approval, trace_id=trace_id, approver_actor=body.actor
            )
        except Forbidden as error:
            raise HTTPException(403, str(error)) from error
        except NotFound as error:
            raise HTTPException(404, str(error)) from error
        except Conflict as error:
            raise HTTPException(409, str(error)) from error

        stream = TurnStream(loop, start_seq=pending.events_already_seen)
        loop.run_in_executor(
            None,
            functools.partial(
                svc.resume_turn, pending, approved=body.approved, stream=stream
            ),
        )
        return StreamingResponse(
            _sse_body(stream), media_type="text/event-stream", headers=_sse_headers()
        )

    # -- reading the evidence back -----------------------------------------

    @app.get("/api/runs/{trace_id}")
    async def get_run(trace_id: str, request: Request) -> RunReportOut:
        svc = get_service(request)
        result = await _run_blocking(svc.get_run_report, trace_id=trace_id)
        if result is None:
            raise HTTPException(404, f"no run {trace_id!r}")
        return result

    @app.get("/api/documents/{document_id}")
    async def get_document(document_id: str, actor: str, request: Request) -> Response:
        res = get_resources(request)
        entry = res.manifest.get(document_id)
        if entry is None:
            raise HTTPException(404, f"no document {document_id!r}")
        try:
            user = res.users.get(actor)
        except UnknownUser:
            raise HTTPException(404, f"unknown actor {actor!r}") from None

        context = RetrievalContext.for_actor(
            user.actor, project_code=user.project_code, scopes=user.scopes
        )
        if not is_authorized(entry, context):
            raise HTTPException(403, "not authorized to read this document")

        source_path = Path(entry.path)
        if source_path.suffix.lower() == ".pdf":
            source_path = SOURCES_DIR / f"{entry.document_id}.md"
        else:
            source_path = DEFAULT_MANIFEST_PATH.parent / entry.path

        media_type = _MEDIA_TYPES.get(source_path.suffix.lower())
        if media_type is None or not source_path.exists():
            raise HTTPException(
                415, f"{document_id!r} has no servable text representation"
            )

        text = await _run_blocking(source_path.read_text, encoding="utf-8")
        return PlainTextResponse(text, media_type=media_type)

    # -- the built React app, mounted last so /api/* always wins -----------

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:

        @app.get("/{_full_path:path}")
        async def _ui_not_built(_full_path: str) -> None:
            raise HTTPException(
                503,
                "UI not built: npm --prefix ui install && npm --prefix ui run build",
            )

    return app
