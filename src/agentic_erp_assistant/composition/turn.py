"""Assembling one turn's ports over the shared resources, per request.

:class:`~agentic_erp_assistant.rag.retriever.RetrievalService.for_context`,
:class:`~agentic_erp_assistant.memory.service.MemoryService.for_scope`, and
:class:`~agentic_erp_assistant.trace.telemetry.RunTelemetry` (via
``trace_id``) are all bound to one actor or one run by design -- so the only
honest place to build the rest of a turn's port graph is here, per request,
over :class:`~agentic_erp_assistant.composition.resources.AppResources`'
process-wide clients and indexes. This is cheap: every object built below is a
frozen dataclass holding references, not a connection or an index rebuild.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import psycopg

from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.users import User
from agentic_erp_assistant.context.catalogue import build_catalogue
from agentic_erp_assistant.context.history_injection import HISTORY_TURN_LIMIT
from agentic_erp_assistant.engine.orchestrator import RunOrchestrator
from agentic_erp_assistant.engine.ports import DocumentRetrieverPort
from agentic_erp_assistant.engine.workflow import MAX_STEPS, WorkflowRuntime
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.inspection import ModelRequestSnapshot, ModelResponseSnapshot
from agentic_erp_assistant.llm.prompts import Principal
from agentic_erp_assistant.llm.streaming import AnswerStreamSink
from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.llm.tools import (
    GET_PROJECT_STATUS_FLAKY_TOOL,
    GET_PROJECT_STATUS_TOOL,
    PLANNING_TOOLS,
    ToolSpec,
)
from agentic_erp_assistant.memory.conversation import ConversationMemory
from agentic_erp_assistant.memory.extractor import LLMMemoryProposer
from agentic_erp_assistant.memory.models import MemoryScope
from agentic_erp_assistant.memory.promotion import LLMSessionSummaryProposer
from agentic_erp_assistant.memory.service import MemoryService
from agentic_erp_assistant.persistence.postgres_audit import PostgresAuditLog
from agentic_erp_assistant.persistence.postgres_conversation import (
    PostgresConversationStore,
)
from agentic_erp_assistant.persistence.postgres_memory import (
    PostgresMemoryAudit,
    PostgresMemoryStore,
)
from agentic_erp_assistant.persistence.postgres_pause import PostgresPauseStore
from agentic_erp_assistant.persistence.postgres_queries import EvidenceQueries
from agentic_erp_assistant.persistence.postgres_trace import PostgresTraceStore
from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.retriever import HybridRetriever, RetrievalOutcome
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.trace import RunTelemetry

__all__ = [
    "build_turn",
    "initial_state",
    "InspectedRetriever",
    "offered_tools",
    "TurnPorts",
    "TurnStreamLike",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class TurnStreamLike(Protocol):
    """What a turn needs from whatever is watching it live.

    Declared here rather than in :mod:`agentic_erp_assistant.web`, the same
    reason every port in this project is declared next to its consumer
    rather than its implementation: composition does not import ``web/``,
    so a caller with no live stream to offer -- ``scripts/run_turn.py`` in
    console mode, a one-shot replay -- passes ``None`` and nothing here
    notices the difference.
    """

    def context(self, state: AgentState) -> None:
        """Called once, before the engine runs (or resumes), with the
        state history/memories/contract were attached to. See
        :attr:`~agentic_erp_assistant.engine.orchestrator.RunOrchestrator.on_start`."""
        ...

    def step(self, state: AgentState) -> None:
        """Called with the state after every node execution. See
        :attr:`~agentic_erp_assistant.engine.workflow.WorkflowRuntime.observer`."""
        ...

    def trace_event(self, event: TraceEvent) -> None:
        """Called with each gateway-internal event as it happens. See
        :attr:`~agentic_erp_assistant.tools.gateway.ToolGateway.on_event`."""
        ...

    def model_call(
        self,
        record: ModelCallRecord,
        request: ModelRequestSnapshot | None,
        response: ModelResponseSnapshot | None,
    ) -> None:
        """Called after every model call this turn's answering gateway
        makes. Satisfies :class:`~agentic_erp_assistant.llm.inspection.
        ModelCallInspector`; see :attr:`~agentic_erp_assistant.llm.gateway.
        LLMGateway.inspector`."""
        ...

    def retrieval(
        self, query: str, limit: int, outcome: RetrievalOutcome, minimum_similarity: float
    ) -> None:
        """Called after every search this turn's retriever makes, with the
        same diagnostics the retriever computed for itself. See
        :class:`InspectedRetriever`."""
        ...

    def delta(self, text: str) -> None:
        """More reply text has arrived. See
        :class:`~agentic_erp_assistant.llm.streaming.AnswerStreamSink`."""
        ...

    def reset(self) -> None:
        """A retried attempt is starting over. See
        :class:`~agentic_erp_assistant.llm.streaming.AnswerStreamSink`."""
        ...


@dataclass(frozen=True)
class TurnPorts:
    """What running or resuming one turn needs, and what reading its
    evidence back needs -- everything :meth:`build_turn` assembled."""

    orchestrator: RunOrchestrator
    connection: psycopg.Connection
    queries: EvidenceQueries


@dataclass(frozen=True)
class InspectedRetriever:
    """A :class:`~agentic_erp_assistant.rag.retriever.HybridRetriever`,
    reporting its own search diagnostics to whoever is watching this turn
    live, without widening :class:`~agentic_erp_assistant.engine.ports.
    DocumentRetrieverPort` or asking a single test standing in for it to
    know this class exists.

    Satisfies the port structurally (one method, ``search``, the same
    signature) by delegating to :meth:`~agentic_erp_assistant.rag.
    retriever.HybridRetriever.search_detailed` -- the same computation
    :meth:`~agentic_erp_assistant.rag.retriever.HybridRetriever.search`
    itself makes, one layer earlier, so wrapping a retriever never runs a
    search twice or changes what a node sees back.
    """

    inner: HybridRetriever
    stream: TurnStreamLike

    def search(self, query: str, *, limit: int) -> Sequence[EvidenceSnippet]:
        outcome = self.inner.search_detailed(query, limit=limit)
        try:
            self.stream.retrieval(query, limit, outcome, self.inner.minimum_similarity)
        except Exception:  # noqa: BLE001 - a screen going away must not end a turn
            logger.warning(
                "retrieval diagnostics observer raised; the turn continues",
                exc_info=True,
            )
        return tuple(hit.chunk.as_snippet() for hit in outcome.hits)


def offered_tools(resources: AppResources) -> tuple[ToolSpec, ...]:
    """What the planner offers the model this turn.

    :data:`~agentic_erp_assistant.llm.tools.PLANNING_TOOLS`, with
    ``get_project_status`` swapped for the flaky twin when
    ``DEV_FLAKY_STATUS=1`` -- the toggle that makes the retry-then-succeed
    path (manual-test.md T11) reachable from a browser without a scripted
    fake standing in for the model. Both tools are always *executable*
    (:func:`~agentic_erp_assistant.tools.registry.build_default_registry`
    binds both); this only changes which one the model is ever shown.
    """
    if not resources.settings.dev_flaky_status:
        return PLANNING_TOOLS
    return tuple(
        GET_PROJECT_STATUS_FLAKY_TOOL if tool is GET_PROJECT_STATUS_TOOL else tool
        for tool in PLANNING_TOOLS
    )


def build_turn(
    resources: AppResources,
    *,
    user: User,
    session_id: str,
    trace_id: str,
    connection: psycopg.Connection,
    stream: TurnStreamLike | None = None,
) -> TurnPorts:
    """Assemble one turn's ports over the shared resources.

    Args:
        resources: The process-wide clients and indexes.
        user: Who this turn is for -- supplies the actor, the project, and
            the scopes every port below binds to.
        session_id: The conversation this turn belongs to.
        trace_id: This run's own id.
        connection: One Postgres connection, opened for this request alone
            (``resources.connect()``) and never shared -- see
            :attr:`~agentic_erp_assistant.composition.resources.
            AppResources.connect`.
        stream: Where to observe this turn live, or ``None`` for a turn
            nobody is watching (a script, a replay). When given, it is
            wired to six places at once: the orchestrator's ``on_start``
            hook, the engine's node-by-node observer, the tool gateway's
            internal event hook, the retriever (via :class:`InspectedRetriever`),
            and both the streaming sink and the model-call inspector on the
            *answering* gateway only -- never on the memory gateway, so a
            memory proposal can never stream into the chat (D5 in the code
            plan) and its calls' live I/O is out of this phase's scope.
    """
    settings = resources.settings
    traces = PostgresTraceStore(connection)
    telemetry = RunTelemetry(trace_id=trace_id, store=traces)

    # D1 (the memory refactor): the model is told who it is talking to and
    # which project, in the system role. The gateway's project check (ADR
    # 0017) stays exactly as it is -- the block makes the model *right*, the
    # gateway still makes it *safe*.
    project = resources.erp.project(user.project_code)
    principal = Principal(
        actor=user.actor,
        display_name=user.display_name,
        role=user.role,
        project_code=user.project_code,
        project_name=project.name if project is not None else None,
    )

    # Built once and reused for both the retriever and the catalogue below --
    # one snapshot of this turn's entitlements, so what the model is told it
    # may search and what the retriever will actually let it read can never
    # drift apart (ADR 0026).
    retrieval_context = RetrievalContext.for_actor(
        user.actor, project_code=user.project_code, scopes=user.scopes
    )
    catalogue = build_catalogue(resources.manifest, retrieval_context)

    answer_sink: AnswerStreamSink | None = stream if stream is not None else None
    answering = LLMGateway(
        resources.chat_client,
        context_window=settings.context_window,
        output_reserve=settings.output_reserve,
        telemetry=telemetry,
        stream=answer_sink,
        inspector=stream if stream is not None else None,
        inspect_io=settings.dev_trace_model_io,
        principal=principal,
        catalogue=catalogue,
    )
    # A second gateway, same client and budget, no sink and no inspector:
    # memory work must never stream into the chat, and its calls' I/O is a
    # narrower scope than this phase covers -- see
    # docs/trace-inspector-plan.md's own note on the simplification. It does
    # get the principal: the proposer must know "the user" is Priya, not
    # "the assistant". It gets no catalogue -- proposing and summarizing
    # memories never routes a decision to search or refuse.
    memory_model = LLMGateway(
        resources.chat_client,
        context_window=settings.context_window,
        output_reserve=settings.output_reserve,
        telemetry=telemetry,
        principal=principal,
    )

    planner = Planner(answering, tools=offered_tools(resources))

    retriever: DocumentRetrieverPort = resources.retrieval.for_context(retrieval_context)
    if stream is not None:
        retriever = InspectedRetriever(inner=retriever, stream=stream)

    gateway = ToolGateway(
        registry=resources.registry,
        audit=PostgresAuditLog(connection),
        limiter=resources.limiter,
        on_event=stream.trace_event if stream is not None else None,
    )

    runtime = WorkflowRuntime(
        retriever=retriever,
        tools=gateway,
        planner=planner,
        composer=answering,
        max_steps=settings.dev_max_steps or MAX_STEPS,
        observer=stream.step if stream is not None else None,
    )

    memory_service = MemoryService(
        store=PostgresMemoryStore(connection),
        index=resources.memory_index,
        embeddings=resources.embeddings,
        model=settings.model,
        required_scope=settings.memory_required_scope,
        proposer=LLMMemoryProposer(model=memory_model) if settings.memory_proposer else None,
        summary_proposer=LLMSessionSummaryProposer(model=memory_model),
        audit=PostgresMemoryAudit(connection),
    )
    memory = memory_service.for_scope(
        MemoryScope.for_actor(
            user.actor,
            project_code=user.project_code,
            session_id=session_id,
            scopes=user.scopes,
        )
    )

    conversation = ConversationMemory(
        store=PostgresConversationStore(connection),
        model=settings.model,
        turn_limit=settings.dev_history_turn_limit or HISTORY_TURN_LIMIT,
    )

    orchestrator = RunOrchestrator(
        runtime,
        traces,
        PostgresPauseStore(connection),
        memory=memory,
        conversation=conversation,
        declarer=planner,
        on_start=stream.context if stream is not None else None,
    )

    return TurnPorts(
        orchestrator=orchestrator,
        connection=connection,
        queries=EvidenceQueries(connection),
    )


def initial_state(
    user: User, *, session_id: str, trace_id: str, message: str
) -> AgentState:
    """The unrouted state a fresh turn starts from.

    A free function rather than a method, the same reason
    :func:`~agentic_erp_assistant.engine.transitions.advance` is: it builds
    one thing from arguments a caller already has, with nothing of its own
    to hold.
    """
    return AgentState(
        request=message,
        actor=user.actor,
        project_code=user.project_code,
        trace_id=trace_id,
        session_id=session_id,
        scopes=user.scopes,
    )
