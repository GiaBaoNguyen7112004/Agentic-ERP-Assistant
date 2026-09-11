"""ChatService: an HTTP request becoming a call into composition.

build_turn is monkeypatched to a scripted fake orchestrator -- what is
under test here is ChatService's own logic (which event the tail emits,
how precheck_approval turns pause status into 403/404/409, that nothing
ever raises out of run_turn/resume_turn), not the engine underneath it,
which every other test file already proves.
"""

import asyncio
import threading
from dataclasses import dataclass

import pytest

from agentic_erp_assistant.composition.settings import Settings
from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.turn import TurnPorts
from agentic_erp_assistant.composition.users import User, UserDirectory
from agentic_erp_assistant.engine.orchestrator import ApprovalAlreadySettled
from agentic_erp_assistant.persistence.postgres_queries import ModelCallTotals
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.web import service as service_module
from agentic_erp_assistant.web.protocol import (
    AnswerEvent,
    ApprovalRequiredEvent,
    ErrorEvent,
    TurnFinishedEvent,
    TurnStartedEvent,
)
from agentic_erp_assistant.web.service import ChatService, Conflict, Forbidden, NotFound
from agentic_erp_assistant.web.stream import TurnStream

SCOPES = frozenset({"project.status.read", "approvals.decide"})


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeOrchestrator:
    def __init__(self, *, handle_result=None, resume_result=None, resume_raises=None):
        self.handle_result = handle_result
        self.resume_result = resume_result
        self.resume_raises = resume_raises
        self.resume_calls: list[tuple[str, bool, str | None]] = []
        self.handle_calls: list[AgentState] = []

    def handle(self, state: AgentState) -> AgentState:
        self.handle_calls.append(state)
        return self.handle_result

    def resume(self, trace_id: str, *, approved: bool, decided_by: str | None = None) -> AgentState:
        self.resume_calls.append((trace_id, approved, decided_by))
        if self.resume_raises is not None:
            raise self.resume_raises
        return self.resume_result


class FakeQueries:
    def model_calls(self, trace_id: str):
        return (), ModelCallTotals(count=1, input_tokens=10, output_tokens=5, cost_usd=0.01, unpriced=0)


class Row:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return self._value


@dataclass
class FakeConnection:
    pause_status: str | None = None
    pause_state: AgentState | None = None
    closed: bool = False

    def execute(self, sql: str, params=None):
        if "SELECT status FROM pauses" in sql:
            return Row((self.pause_status,) if self.pause_status is not None else None)
        if "SELECT state FROM pauses" in sql:
            if self.pause_state is None:
                return Row(None)
            return Row((self.pause_state.model_dump(mode="json"),))
        raise AssertionError(f"unexpected query: {sql}")

    def close(self) -> None:
        self.closed = True


def a_directory() -> UserDirectory:
    return UserDirectory(
        (
            User(actor="priya", display_name="Priya", role="Lead", project_code="atlas", scopes=SCOPES),
            User(actor="wei", display_name="Wei", role="Finance", project_code="atlas",
                 scopes=frozenset({"project.status.read"})),
        )
    )


def a_resources(**overrides) -> AppResources:
    fields = {
        "settings": Settings(model="gpt-4o", context_window=128_000),
        "users": a_directory(),
        "chat_client": object(),
        "embeddings": object(),
        "retrieval": object(),
        "memory_index": object(),
        "erp": object(),
        "registry": object(),
        "limiter": object(),
        "manifest": {},
        "connect": lambda: FakeConnection(),
    }
    fields.update(overrides)
    return AppResources(**fields)  # type: ignore[arg-type]


def answered_state(**changes) -> AgentState:
    base = AgentState(
        request="How is M2 tracking?", actor="priya", project_code="atlas",
        trace_id="run-1", scopes=SCOPES, route="answer", terminal=True,
        response="M2 is fine.",
    )
    return base.evolve(**changes) if changes else base


def paused_state(**changes) -> AgentState:
    base = AgentState(
        request="Record a risk.", actor="priya", project_code="atlas",
        trace_id="run-1", session_id="sess-1", scopes=SCOPES,
        route="request_approval", tool_name="create_risk",
        tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
        tool_mutating=True, approval="pending",
    )
    return base.evolve(**changes) if changes else base


def collect(produce) -> list:
    """Run ``produce(stream)`` on a real thread and collect every event."""
    loop = asyncio.new_event_loop()
    stream = TurnStream(loop)
    collected = []

    def worker():
        produce(stream)

    thread = threading.Thread(target=worker)

    async def consume():
        thread.start()
        async for event in stream.events():
            collected.append(event)

    try:
        loop.run_until_complete(consume())
    finally:
        thread.join(timeout=5)
        loop.close()
    return collected


def patch_build_turn(monkeypatch: pytest.MonkeyPatch, orchestrator) -> None:
    def fake_build_turn(resources, *, user, session_id, trace_id, connection, stream=None):
        return TurnPorts(orchestrator=orchestrator, connection=connection, queries=FakeQueries())

    monkeypatch.setattr(service_module, "build_turn", fake_build_turn)


# --------------------------------------------------------------------------
# run_turn
# --------------------------------------------------------------------------


def test_run_turn_emits_turn_started_then_answer_then_turn_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = FakeOrchestrator(handle_result=answered_state())
    patch_build_turn(monkeypatch, orchestrator)
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.run_turn(
            actor="priya", session_id="s1", message="How is M2?", stream=stream
        )
    )

    types = [type(e).__name__ for e in events]
    assert types == ["TurnStartedEvent", "AnswerEvent", "TurnFinishedEvent"]
    assert events[1].text == "M2 is fine."
    assert events[2].outcome == "terminal"


def test_run_turn_ending_paused_emits_approval_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = FakeOrchestrator(handle_result=paused_state())
    patch_build_turn(monkeypatch, orchestrator)
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.run_turn(
            actor="priya", session_id="s1", message="Record a risk.", stream=stream
        )
    )

    kinds = [type(e).__name__ for e in events]
    assert "ApprovalRequiredEvent" in kinds
    approval = next(e for e in events if isinstance(e, ApprovalRequiredEvent))
    assert approval.tool_name == "create_risk"
    finished = next(e for e in events if isinstance(e, TurnFinishedEvent))
    assert finished.outcome == "paused"


def test_run_turn_with_an_unknown_actor_emits_an_error_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_build_turn(monkeypatch, FakeOrchestrator())
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.run_turn(
            actor="ghost", session_id=None, message="hi", stream=stream
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "ghost" in events[0].message


def test_run_turn_generates_a_session_when_none_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = FakeOrchestrator(handle_result=answered_state())
    patch_build_turn(monkeypatch, orchestrator)
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.run_turn(
            actor="priya", session_id=None, message="hi", stream=stream
        )
    )

    started = next(e for e in events if isinstance(e, TurnStartedEvent))
    assert started.session_id is not None
    assert started.session_id.startswith("sess-")


# --------------------------------------------------------------------------
# precheck_approval
# --------------------------------------------------------------------------


def test_precheck_rejects_an_unknown_approver() -> None:
    service = ChatService(a_resources())

    with pytest.raises(Forbidden):
        service.precheck_approval(trace_id="run-1", approver_actor="ghost")


def test_precheck_rejects_an_approver_with_no_scope() -> None:
    service = ChatService(a_resources())

    with pytest.raises(Forbidden):
        service.precheck_approval(trace_id="run-1", approver_actor="wei")


def test_precheck_a_trace_that_never_paused_is_not_found() -> None:
    service = ChatService(a_resources(connect=lambda: FakeConnection(pause_status=None)))

    with pytest.raises(NotFound):
        service.precheck_approval(trace_id="run-1", approver_actor="priya")


def test_precheck_an_already_settled_pause_is_a_conflict() -> None:
    service = ChatService(a_resources(connect=lambda: FakeConnection(pause_status="resolved")))

    with pytest.raises(Conflict):
        service.precheck_approval(trace_id="run-1", approver_actor="priya")


def test_precheck_a_pending_pause_resolves_the_original_requester() -> None:
    pending = paused_state(actor="wei")
    service = ChatService(
        a_resources(connect=lambda: FakeConnection(pause_status="pending", pause_state=pending))
    )

    decision = service.precheck_approval(trace_id="run-1", approver_actor="priya")

    assert decision.requester.actor == "wei"
    assert decision.approver.actor == "priya"
    assert decision.session_id == "sess-1"


# --------------------------------------------------------------------------
# resume_turn
# --------------------------------------------------------------------------


def a_decision(**changes):
    from agentic_erp_assistant.web.service import PendingDecision

    fields = {
        "trace_id": "run-1",
        "requester": a_directory().get("priya"),
        "session_id": "sess-1",
        "approver": a_directory().get("priya"),
    }
    fields.update(changes)
    return PendingDecision(**fields)


def test_resume_turn_approved_emits_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = FakeOrchestrator(resume_result=answered_state())
    patch_build_turn(monkeypatch, orchestrator)
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.resume_turn(a_decision(), approved=True, stream=stream)
    )

    assert orchestrator.resume_calls == [("run-1", True, "priya")]
    assert any(isinstance(e, AnswerEvent) for e in events)
    assert any(isinstance(e, TurnStartedEvent) and e.resumed for e in events)


def test_resume_turn_races_an_already_settled_pause_into_an_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = FakeOrchestrator(resume_raises=ApprovalAlreadySettled("already decided"))
    patch_build_turn(monkeypatch, orchestrator)
    service = ChatService(a_resources())

    events = collect(
        lambda stream: service.resume_turn(a_decision(), approved=True, stream=stream)
    )

    assert len(events) == 2  # turn_started, then the error
    assert isinstance(events[-1], ErrorEvent)
    assert "already decided" in events[-1].message
