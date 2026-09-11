"""The FastAPI application: routes, over a fake ChatService and fake resources.

TestClient drives the real ASGI app -- request validation, routing, status
codes -- with everything below ChatService faked, the same split
test_service.py uses one layer down: what a fake build_turn proves there,
a fake ChatService proves here.
"""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.settings import Settings
from agentic_erp_assistant.composition.users import User, UserDirectory
from agentic_erp_assistant.persistence.connection import StoreConnectionError
from agentic_erp_assistant.web import app as app_module
from agentic_erp_assistant.web.app import create_app
from agentic_erp_assistant.web.protocol import (
    AnswerEvent,
    ErrorEvent,
    ModelCallTotalsOut,
    TokenEvent,
    TurnFinishedEvent,
    TurnStartedEvent,
)
from agentic_erp_assistant.web.service import Conflict, Forbidden, NotFound, PendingDecision
from agentic_erp_assistant.web.stream import TurnStream

SCOPES = frozenset({"project.status.read", "approvals.decide"})


class FakeRetrieval:
    class _Index(list):
        pass

    def __init__(self, chunk_count: int = 84) -> None:
        self.lexical_index = self._Index(range(chunk_count))


class FakeConnection:
    def close(self) -> None:
        pass


def a_directory() -> UserDirectory:
    return UserDirectory(
        (
            User(actor="priya", display_name="Priya Raman", role="Lead",
                 project_code="atlas", scopes=SCOPES),
            User(actor="wei", display_name="Wei Chen", role="Finance",
                 project_code="atlas", scopes=frozenset({"project.status.read"})),
        )
    )


def a_resources(**overrides) -> AppResources:
    fields = {
        "settings": Settings(model="gpt-4o", context_window=128_000),
        "users": a_directory(),
        "chat_client": object(),
        "embeddings": object(),
        "retrieval": FakeRetrieval(),
        "memory_index": object(),
        "erp": object(),
        "registry": object(),
        "limiter": object(),
        "manifest": {},
        "connect": lambda: FakeConnection(),
    }
    fields.update(overrides)
    return AppResources(**fields)  # type: ignore[arg-type]


class FakeService:
    """A scripted ChatService: run_turn and resume_turn just emit whatever
    this test told them to, straight onto the stream they were given."""

    def __init__(self) -> None:
        self.run_turn_script = None
        self.precheck_raises: Exception | None = None
        self.precheck_result: PendingDecision | None = None
        self.resume_turn_script = None
        self.run_turn_calls: list[dict] = []

    def run_turn(self, *, actor: str, session_id, message: str, stream: TurnStream) -> None:
        self.run_turn_calls.append(
            {"actor": actor, "session_id": session_id, "message": message}
        )
        if self.run_turn_script is not None:
            self.run_turn_script(stream)
        stream.close()

    def precheck_approval(self, *, trace_id: str, approver_actor: str) -> PendingDecision:
        if self.precheck_raises is not None:
            raise self.precheck_raises
        return self.precheck_result

    def resume_turn(self, pending: PendingDecision, *, approved: bool, stream: TurnStream) -> None:
        if self.resume_turn_script is not None:
            self.resume_turn_script(stream)
        stream.close()


def a_client(resources: AppResources, service: FakeService) -> TestClient:
    """Returns an un-entered TestClient; use as ``with a_client(...) as
    client:`` so the lifespan runs exactly once."""
    app = create_app(resources=resources, service=service)
    return TestClient(app)


def read_sse_events(response) -> list[str]:
    """The event: lines, in order, from a streamed SSE body."""
    lines = response.text.splitlines()
    return [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]


# --------------------------------------------------------------------------
# Static / not built
# --------------------------------------------------------------------------


def test_the_ui_not_built_message_when_static_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path / "nowhere")
    with a_client(a_resources(), FakeService()) as client:
        response = client.get("/")

    assert response.status_code == 503
    assert "npm --prefix ui" in response.json()["detail"]


def test_the_built_ui_is_served_when_static_exists(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>App</title>", encoding="utf-8")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)

    with a_client(a_resources(), FakeService()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "App" in response.text


# --------------------------------------------------------------------------
# /api/users, /api/protocol, /api/health
# --------------------------------------------------------------------------


def test_users_lists_the_fixture() -> None:
    with a_client(a_resources(), FakeService()) as client:
        response = client.get("/api/users")

    body = response.json()
    assert {u["actor"] for u in body} == {"priya", "wei"}
    priya = next(u for u in body if u["actor"] == "priya")
    assert priya["can_approve"] is True


def test_protocol_lists_every_event_type() -> None:
    with a_client(a_resources(), FakeService()) as client:
        response = client.get("/api/protocol")

    assert "turn_started" in response.json()["event_types"]


def test_health_reports_postgres_down_when_connect_fails() -> None:
    def failing_connect():
        raise StoreConnectionError("no database")

    resources = a_resources(connect=failing_connect)
    with a_client(resources, FakeService()) as client:
        response = client.get("/api/health")

    body = response.json()
    assert body["postgres"] == "down"
    assert body["chunks_indexed"] == 84


# --------------------------------------------------------------------------
# /api/sessions (POST)
# --------------------------------------------------------------------------


def test_create_session_for_an_unknown_actor_is_404() -> None:
    with a_client(a_resources(), FakeService()) as client:
        response = client.post("/api/sessions", json={"actor": "ghost"})

    assert response.status_code == 404


def test_create_session_returns_a_fresh_id() -> None:
    with a_client(a_resources(), FakeService()) as client:
        response = client.post("/api/sessions", json={"actor": "priya"})

    assert response.json()["session_id"].startswith("sess-")


# --------------------------------------------------------------------------
# /api/chat
# --------------------------------------------------------------------------


def test_chat_streams_turn_started_tokens_answer_turn_finished_in_order() -> None:
    service = FakeService()

    def script(stream: TurnStream) -> None:
        stream.emit(TurnStartedEvent(trace_id="run-1", session_id="s1", actor="priya", resumed=False))
        stream.delta("Hel")
        stream.delta("lo")
        stream.delta("!")
        stream.emit(
            AnswerEvent(text="Hello!", route="answer", failure="none", error_detail=None, citations=())
        )
        stream.emit(
            TurnFinishedEvent(
                outcome="terminal", route="answer", failure="none", step_count=1,
                evidence=(), memories_recalled=0, history_shown=0, observations=(),
                model_calls=ModelCallTotalsOut(
                    count=0, input_tokens=0, output_tokens=0, cost_usd=0.0, unpriced=0
                ),
            )
        )

    service.run_turn_script = script

    with a_client(a_resources(), service) as client:
        response = client.post(
            "/api/chat", json={"actor": "priya", "session_id": "s1", "message": "hi"}
        )

    assert response.status_code == 200
    assert read_sse_events(response) == [
        "turn_started", "token", "token", "token", "answer", "turn_finished",
    ]


def test_a_blank_message_is_422_before_the_thread_starts() -> None:
    service = FakeService()
    with a_client(a_resources(), service) as client:
        response = client.post("/api/chat", json={"actor": "priya", "message": "   "})

    assert response.status_code == 422
    assert service.run_turn_calls == []


def test_an_oversized_message_is_422() -> None:
    with a_client(a_resources(), FakeService()) as client:
        response = client.post(
            "/api/chat", json={"actor": "priya", "message": "x" * 4001}
        )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# /api/approvals/{trace_id}
# --------------------------------------------------------------------------


def test_an_approver_without_scope_is_403() -> None:
    service = FakeService()
    service.precheck_raises = Forbidden("no approvals.decide")

    with a_client(a_resources(), service) as client:
        response = client.post("/api/approvals/run-1", json={"actor": "wei", "approved": True})

    assert response.status_code == 403


def test_a_trace_that_never_paused_is_404() -> None:
    service = FakeService()
    service.precheck_raises = NotFound("no such pause")

    with a_client(a_resources(), service) as client:
        response = client.post("/api/approvals/run-1", json={"actor": "priya", "approved": True})

    assert response.status_code == 404


def test_a_settled_pause_is_409() -> None:
    service = FakeService()
    service.precheck_raises = Conflict("already settled")

    with a_client(a_resources(), service) as client:
        response = client.post("/api/approvals/run-1", json={"actor": "priya", "approved": True})

    assert response.status_code == 409


def test_a_valid_decision_streams_the_resumed_turn() -> None:
    service = FakeService()
    service.precheck_result = PendingDecision(
        trace_id="run-1",
        requester=a_directory().get("wei"),
        session_id="s1",
        approver=a_directory().get("priya"),
        events_already_seen=2,
    )

    def script(stream: TurnStream) -> None:
        stream.emit(
            TurnStartedEvent(trace_id="run-1", session_id="s1", actor="wei", resumed=True)
        )

    service.resume_turn_script = script

    with a_client(a_resources(), service) as client:
        response = client.post(
            "/api/approvals/run-1", json={"actor": "priya", "approved": True}
        )

    assert response.status_code == 200
    assert read_sse_events(response) == ["turn_started"]
