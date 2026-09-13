"""build_turn: the per-request wiring, over fakes for every resource.

What is asserted is the wiring itself -- which object ended up bound to
which port -- not that any of it works end to end. That is proved live,
against the real environment, by scripts/run_turn.py and
docs/manual-test.md.
"""

from dataclasses import dataclass

import pytest

from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.settings import Settings
from agentic_erp_assistant.composition.turn import (
    build_turn,
    initial_state,
    InspectedRetriever,
    offered_tools,
)
from agentic_erp_assistant.composition.users import User
from agentic_erp_assistant.llm.tools import (
    GET_PROJECT_STATUS_FLAKY_TOOL,
    GET_PROJECT_STATUS_TOOL,
    PLANNING_TOOLS,
)
from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.tools.registry import ToolRegistry


class FakeChatClient:
    model_name = "fake-model-1"

    def complete(self, messages, *, temperature, on_delta=None):
        raise AssertionError("not exercised by wiring tests")

    def call_with_tools(self, messages, *, tools, temperature, on_delta=None):
        raise AssertionError("not exercised by wiring tests")


class FakeRetriever:
    def search(self, query, *, limit):
        return ()


@dataclass
class FakeRetrievalService:
    contexts_seen: list = None

    def __post_init__(self):
        self.contexts_seen = []

    def for_context(self, context: RetrievalContext) -> FakeRetriever:
        self.contexts_seen.append(context)
        return FakeRetriever()


class RecordingStream:
    def __init__(self) -> None:
        self.contexts = []
        self.steps = []
        self.trace_events = []
        self.deltas = []
        self.resets = 0

    def context(self, state) -> None:
        self.contexts.append(state)

    def step(self, state) -> None:
        self.steps.append(state)

    def trace_event(self, event) -> None:
        self.trace_events.append(event)

    def delta(self, text: str) -> None:
        self.deltas.append(text)

    def reset(self) -> None:
        self.resets += 1


def a_user(**overrides) -> User:
    fields = {
        "actor": "priya",
        "display_name": "Priya Raman",
        "role": "Delivery lead",
        "project_code": "atlas",
        "scopes": frozenset({"project.docs.read", "project.status.read"}),
    }
    fields.update(overrides)
    return User.model_validate(fields)


def a_resources(**overrides) -> AppResources:
    fields = {
        "settings": Settings(model="gpt-4o", context_window=128_000),
        "users": None,
        "chat_client": FakeChatClient(),
        "embeddings": object(),
        "retrieval": FakeRetrievalService(),
        "memory_index": object(),
        "erp": object(),
        "registry": ToolRegistry(()),
        "limiter": object(),
        "manifest": {},
        "connect": lambda: object(),
    }
    fields.update(overrides)
    return AppResources(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# build_turn: the wiring
# --------------------------------------------------------------------------


def test_the_answering_gateway_carries_the_stream_sink() -> None:
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    assert turn.orchestrator.runtime.composer.stream is stream


def test_the_orchestrators_on_start_is_the_streams_context_method() -> None:
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    assert turn.orchestrator.on_start == stream.context


def test_without_a_stream_on_start_is_none() -> None:
    resources = a_resources()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=None,
    )

    assert turn.orchestrator.on_start is None


def test_the_memory_gateway_carries_no_sink() -> None:
    """D5: a memory proposal must never stream into the chat."""
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    proposer = turn.orchestrator.memory.service.proposer
    assert proposer is not None
    assert proposer.model.stream is None


def test_the_telemetry_sink_carries_the_trace_id() -> None:
    resources = a_resources()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-42",
        connection=object(),
    )

    assert turn.orchestrator.runtime.composer.telemetry.trace_id == "run-42"


def test_the_retriever_context_comes_from_the_same_user_the_state_does() -> None:
    resources = a_resources()
    user = a_user(actor="orion.lead", project_code="orion", scopes=frozenset({"project.status.read"}))

    build_turn(
        resources,
        user=user,
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
    )
    state = initial_state(user, session_id="sess-1", trace_id="run-1", message="hi")

    seen = resources.retrieval.contexts_seen
    assert len(seen) == 1
    assert seen[0].actor == user.actor == state.actor
    assert seen[0].project_code == user.project_code == state.project_code
    assert seen[0].scopes == user.scopes


def test_the_retriever_is_wrapped_to_report_its_diagnostics_when_a_stream_is_present() -> None:
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    retriever = turn.orchestrator.runtime.retriever
    assert isinstance(retriever, InspectedRetriever)
    assert retriever.stream is stream


def test_the_retriever_is_unwrapped_without_a_stream() -> None:
    resources = a_resources()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=None,
    )

    assert not isinstance(turn.orchestrator.runtime.retriever, InspectedRetriever)


def test_the_answering_gateways_inspector_is_the_stream() -> None:
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    assert turn.orchestrator.runtime.composer.inspector is stream


def test_the_memory_gateways_inspector_is_none_even_with_a_stream() -> None:
    """The narrower scope this phase settled for: a memory proposal's
    calls never stream, and their live I/O is not captured either."""
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    proposer = turn.orchestrator.memory.service.proposer
    assert proposer is not None
    assert proposer.model.inspector is None


def test_dev_trace_model_io_reaches_the_answering_gateway() -> None:
    resources = a_resources(settings=Settings(model="gpt-4o", context_window=128_000, dev_trace_model_io=True))
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    assert turn.orchestrator.runtime.composer.inspect_io is True


def test_the_tool_gateway_gets_the_shared_limiter_and_the_per_turn_hook() -> None:
    resources = a_resources()
    stream = RecordingStream()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=stream,
    )

    gateway = turn.orchestrator.runtime.tools
    assert gateway.limiter is resources.limiter
    assert gateway.on_event == stream.trace_event


def test_without_a_stream_nothing_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    resources = a_resources()

    turn = build_turn(
        resources,
        user=a_user(),
        session_id="sess-1",
        trace_id="run-1",
        connection=object(),
        stream=None,
    )

    assert turn.orchestrator.runtime.composer.stream is None
    assert turn.orchestrator.runtime.observer is None
    assert turn.orchestrator.runtime.tools.on_event is None


def test_the_max_steps_override_reaches_the_runtime() -> None:
    resources = a_resources(
        settings=Settings(model="gpt-4o", context_window=128_000, dev_max_steps=2)
    )

    turn = build_turn(
        resources, user=a_user(), session_id="sess-1", trace_id="run-1", connection=object()
    )

    assert turn.orchestrator.runtime.max_steps == 2


def test_the_history_turn_limit_override_reaches_the_conversation_window() -> None:
    resources = a_resources(
        settings=Settings(
            model="gpt-4o", context_window=128_000, dev_history_turn_limit=2
        )
    )

    turn = build_turn(
        resources, user=a_user(), session_id="sess-1", trace_id="run-1", connection=object()
    )

    assert turn.orchestrator.conversation.turn_limit == 2


def test_memory_proposer_off_leaves_the_service_with_no_proposer() -> None:
    resources = a_resources(
        settings=Settings(model="gpt-4o", context_window=128_000, memory_proposer=False)
    )

    turn = build_turn(
        resources, user=a_user(), session_id="sess-1", trace_id="run-1", connection=object()
    )

    assert turn.orchestrator.memory.service.proposer is None


# --------------------------------------------------------------------------
# initial_state
# --------------------------------------------------------------------------


def test_initial_state_carries_the_users_project_and_scopes() -> None:
    user = a_user(project_code="orion", scopes=frozenset({"project.status.read"}))

    state = initial_state(user, session_id="sess-1", trace_id="run-1", message="hi")

    assert state.actor == user.actor
    assert state.project_code == "orion"
    assert state.scopes == user.scopes
    assert state.session_id == "sess-1"
    assert state.trace_id == "run-1"
    assert state.request == "hi"


# --------------------------------------------------------------------------
# offered_tools: DEV_FLAKY_STATUS
# --------------------------------------------------------------------------


def test_offered_tools_defaults_to_the_standard_menu() -> None:
    resources = a_resources()

    assert offered_tools(resources) == PLANNING_TOOLS


def test_dev_flaky_status_swaps_the_status_tool_for_its_flaky_twin() -> None:
    resources = a_resources(
        settings=Settings(model="gpt-4o", context_window=128_000, dev_flaky_status=True)
    )

    tools = offered_tools(resources)

    assert GET_PROJECT_STATUS_TOOL not in tools
    assert GET_PROJECT_STATUS_FLAKY_TOOL in tools
    assert len(tools) == len(PLANNING_TOOLS)
