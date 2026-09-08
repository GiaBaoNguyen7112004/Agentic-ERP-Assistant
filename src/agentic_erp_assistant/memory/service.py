"""The composition point: recall before a turn, consolidate after it.

Everything else in this package does one thing and does it against no I/O. This
is where those pieces meet the store, the index and the provider, in the two
operations a turn actually performs:

* :meth:`SessionMemory.recall` -- gather what might bear on the request, select
  from it, and hand back what fits;
* :meth:`SessionMemory.consolidate` -- ask what the finished turn was worth, run
  every proposal through the gate, and write down all of it, refusals included.

Split the way retrieval is split
--------------------------------

:class:`MemoryService` holds the expensive shared things and
:meth:`MemoryService.for_scope` hands out a :class:`SessionMemory` bound to one
actor, one project and one conversation. That is
:class:`~agentic_erp_assistant.rag.retriever.RetrievalService`'s arrangement, for
its reason: there is deliberately no method here that reads or writes without a
scope, because an unscoped call is the one that eventually gets made from
somewhere that forgot to pass one -- and the memory it returns is somebody
else's.

Recall costs one embedding call; consolidation costs one model call and at most
one more embedding call
-----------------------------------------------------------------------------

Both are stated because both are money. Recall embeds the request once and
searches; the memories were embedded when they were written. Consolidation asks
the model once and embeds the batch it decided to store in one request, not one
per record.

What this module does not decide
--------------------------------

**Whether the task in flight has changed.** :meth:`SessionMemory.advance_intent`
and its siblings exist and are complete, and nothing here calls them. Inferring
"is this the same task?" from the words of a request is a classification this
project has no evidence it can do well, and getting it wrong produces exactly
the failure :mod:`agentic_erp_assistant.memory.intent` exists to prevent -- a
slot from the abandoned task answering a question about the new one. So the
decision belongs to the layer that has the whole conversation, and the machinery
is finished and tested for when that layer arrives.

**What a session amounts to.** :meth:`SessionMemory.consolidate` takes an
optional conversation state and summarizes it when one is given. Nothing
produces that state yet; the compaction allow-list is fed by a caller that does
not exist. The seam is here so the summary lands with everything else when it
does.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentic_erp_assistant.context.compact import compact_conversation
from agentic_erp_assistant.context.memory_injection import (
    PINNED_KINDS,
    MemorySelection,
    select_memories,
)
from agentic_erp_assistant.memory.audit import (
    InMemoryMemoryAudit,
    MemoryAuditRow,
    MemoryAuditSink,
    summarize_statement,
)
from agentic_erp_assistant.memory.extractor import MemoryProposerPort
from agentic_erp_assistant.memory.intent import IntentState
from agentic_erp_assistant.memory.models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    memory_id,
)
from agentic_erp_assistant.memory.policy import decide
from agentic_erp_assistant.memory.store import MemoryStorePort
from agentic_erp_assistant.memory.summary import SESSION_SUMMARY_KEY, summarize_session
from agentic_erp_assistant.memory.vector_store import MemoryVectorStorePort
from agentic_erp_assistant.rag.ports import EmbeddingsPort
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.memory import MemoryRecord

__all__ = [
    "MEMORY_BUDGET_TOKENS",
    "MemoryService",
    "RECALL_LIMIT",
    "SEMANTIC_CANDIDATES",
    "SessionMemory",
]

logger = logging.getLogger(__name__)


MEMORY_BUDGET_TOKENS = 400
"""How many tokens of a prompt memory may occupy.

Deliberately small, and small for a reason that is not thrift. Memory competes
with evidence for the same window, and evidence is what an answer has to cite --
a turn that spent four hundred tokens remembering and had none left for the
passage it needed would produce a confident, uncited answer, which is the
failure this project is most anxious to avoid. Four hundred tokens is roughly six
statements: the task in flight, a couple of preferences, and the session's
residue. A conversation that needs more than that from memory is one that should
be asking the documents.

A constructor argument, so a deployment can move it without editing this file.
"""

RECALL_LIMIT = 8
"""How many memories may reach the selection stage from the semantic half.

A bound before the budget rather than instead of it: the budget decides what
fits, this decides how much is worth measuring. Without it a large store would
hand the builder hundreds of candidates to tokenize on every turn, and the
tokenizing is the expensive part.
"""

SEMANTIC_CANDIDATES = 8
"""How many hits the vector index returns per recall. See :data:`RECALL_LIMIT`."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class MemoryService:
    """The shared, expensive half of memory, built once per process.

    Usage::

        service = MemoryService(
            store=PostgresMemoryStore(connection),
            index=QdrantMemoryIndex.from_env(),
            embeddings=OpenAIEmbeddingsClient(),
            audit=PostgresMemoryAudit(connection),
            proposer=LLMMemoryProposer(model=gateway),
            model=os.environ["OPENAI_MODEL"],
            required_scope="project.docs.read",
        )
        memory = service.for_scope(
            MemoryScope.for_actor("priya", project_code="atlas",
                                  session_id="sess-1", scopes=state.scopes)
        )
    """

    store: MemoryStorePort
    """Where records live, and where the task in flight lives.

    Typed as one port and required to satisfy
    :class:`~agentic_erp_assistant.memory.store.IntentStorePort` as well. One
    field rather than two, because one adapter implements both against one
    database -- and two fields would let a caller wire them to different ones,
    which is a configuration that has no correct behaviour.
    """

    index: MemoryVectorStorePort
    """The semantic index over that store."""

    embeddings: EmbeddingsPort
    """What turns a request or a statement into a vector."""

    model: str
    """Whose tokenizer sizes the memory budget.

    Required, with no default, for the reason
    :attr:`~agentic_erp_assistant.context.builder.ContextBuilder.model` is: a
    token count means nothing without the model that produced it, and a default
    would be one model's arithmetic silently applied to another.
    """

    required_scope: str
    """The entitlement a reader of a newly written memory will need.

    Configured per deployment rather than derived from the turn's evidence, and
    that gap is worth stating plainly: an
    :class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` deliberately
    carries no policy field, so by the time a passage reaches a turn's state the
    chunk's own ``required_scope`` is gone. What keeps a restricted document out
    of memory anyway is the policy's source rule -- a statement that restates a
    retrieved passage is refused ``belongs_to_rag``, whatever that passage's
    scope was. Set this to the project's baseline document scope; a deployment
    with genuinely mixed classifications should set it to the most restrictive
    one and revisit when the snippet type carries policy.
    """

    proposer: MemoryProposerPort | None = None
    """What nominates candidates. ``None`` disables consolidation's proposal
    step entirely -- recall and the intent operations still work, which is the
    configuration a deployment uses while it decides whether the model call is
    worth its cost."""

    audit: MemoryAuditSink = field(default_factory=InMemoryMemoryAudit)
    """Where every decision is written down. Defaults to the in-memory sink for
    the reason the tool gateway's does: a consolidator with no sink would
    silently decide nothing, which is worse than a sink that forgets."""

    budget_tokens: int = MEMORY_BUDGET_TOKENS
    recall_limit: int = RECALL_LIMIT
    semantic_candidates: int = SEMANTIC_CANDIDATES

    now: Callable[[], datetime] = _utc_now
    """The clock, injected so a test can make time deterministic and so nothing
    here reads ``datetime.now`` at a distance."""

    def for_scope(self, scope: MemoryScope) -> "SessionMemory":
        """Memory for one actor, on one project, in one conversation.

        The only way to get a reader or a writer. There is deliberately no
        method that operates without a scope.
        """
        return SessionMemory(service=self, scope=scope)


@dataclass(frozen=True)
class SessionMemory:
    """Memory bound to one turn's actor, project and conversation.

    Frozen and cheap to build: it holds a reference to the shared service rather
    than a copy of anything, so making one per turn costs an object -- which is
    the price of the scope being impossible to get wrong.
    """

    service: MemoryService
    scope: MemoryScope

    # -- reading -----------------------------------------------------------

    def recall(self, state: AgentState) -> tuple[MemoryRecord, ...]:
        """What this turn should be shown, already filtered and budgeted.

        Three gathering steps and one selection. The **pinned** kinds come
        straight from the store, because the task in flight and this actor's
        preferences bear on every request. The **semantic** half embeds the
        request once and asks the index, then hydrates the ids it returns
        through the store -- which re-checks liveness and scope, and is what
        makes the index an optimization rather than a second source of truth.
        The **open intent**, if there is one, is projected into a record here
        rather than stored as one, so what the model sees is the task as it
        stands now.

        Then :func:`~agentic_erp_assistant.context.memory_injection.select_memories`
        decides, and every gathered record ends up selected, skipped with a
        reason, or over budget.

        Returns:
            What to put in the prompt, in the order it will be rendered. Empty
            is the common answer and is never an error.
        """
        return self.recall_in_detail(state).selected

    def recall_in_detail(self, state: AgentState) -> MemorySelection:
        """:meth:`recall`, with what it passed over and what it cost.

        For the trace, and for anything that wants to explain why a turn was
        shown what it was. Same computation, one layer earlier.
        """
        pinned = self.service.store.live(
            self.scope, kinds=tuple(PINNED_KINDS), limit=self.service.recall_limit
        )
        gathered: dict[str, MemoryRecord] = {
            record.memory_id: record for record in pinned
        }

        semantic = self._semantic(state.request)
        for record in semantic:
            gathered.setdefault(record.memory_id, record)

        intent = self.open_intent()
        if intent is not None:
            projection = intent.as_memory(
                memory_id=f"intent-{intent.intent_id}",
                recorded_in_run=state.trace_id,
                recorded_at=self.service.now(),
            )
            gathered[projection.memory_id] = projection

        # Only the vector hits are passed as ``matched``. Handing the selector
        # every gathered id would tell it that everything is relevant, which is
        # a one-word way to turn the whole selection step into a no-op.
        return select_memories(
            state.request,
            gathered.values(),
            scope=self.scope,
            budget_tokens=self.service.budget_tokens,
            model=self.service.model,
            matched=[record.memory_id for record in semantic],
        )

    def _semantic(self, request: str) -> tuple[MemoryRecord, ...]:
        """The vector half: one embedding call, then a hydration.

        A failure here is degraded recall, not a failed turn: the pinned half
        still reaches the prompt, and the turn answers with less background
        rather than not at all. Logged at warning, because a store that has
        stopped being searchable is worth noticing even though nothing broke.
        """
        if not request.strip():
            return ()
        try:
            batch = self.service.embeddings.embed([request])
            if not batch.vectors:
                return ()
            hits = self.service.index.search(
                batch.vectors[0],
                scope=self.scope,
                limit=self.service.semantic_candidates,
            )
        except Exception as error:  # noqa: BLE001 - degraded recall, not a failure
            logger.warning(
                "semantic recall failed for session %s (%s: %s); falling back to "
                "the pinned kinds alone",
                self.scope.session_id,
                type(error).__name__,
                error,
            )
            return ()

        return self.service.store.by_id(
            [hit.memory_id for hit in hits], self.scope
        )

    # -- writing -----------------------------------------------------------

    def consolidate(
        self,
        state: AgentState,
        *,
        conversation: Mapping[str, object] | None = None,
    ) -> tuple[MemoryDecision, ...]:
        """Decide what the finished turn was worth, and write all of it down.

        Args:
            state: The turn, terminal. Consolidating a paused turn would store
                facts from a decision nobody has made yet, which is why the
                caller checks first.
            conversation: The session's durable state, if the caller keeps any.
                When given, the session summary is refreshed from it through the
                compaction allow-list. ``None`` skips that step -- see the module
                docstring on what does not exist yet.

        Returns:
            One decision per candidate, in the order they were proposed, with
            the summary's own decision appended when there was one. Rejections
            are included: they are the half of the record that shows the policy
            working.
        """
        decisions: list[MemoryDecision] = []
        stored: list[MemoryRecord] = []
        retired: list[str] = []

        existing = list(self.service.store.live(self.scope))
        for candidate in self._proposals(state):
            verdict = decide(candidate, existing=existing, scope=self.scope)
            decisions.append(verdict)
            identifier = memory_id(candidate, self.scope)

            if not verdict.stores:
                self._audit(state, verdict, candidate.kind, identifier, candidate.statement)
                continue

            record = self._record(candidate, identifier, state, verdict.supersedes)
            self.service.store.write(record)
            stored.append(record)
            # Judged against each other as well as against the store, so two
            # candidates for one key in a single turn become a write and an
            # update rather than two live rows nobody can choose between.
            existing = [
                held for held in existing if held.memory_id not in verdict.supersedes
            ] + [record]
            retired.extend(verdict.supersedes)
            self._audit(state, verdict, candidate.kind, identifier, candidate.statement)

        summary = self._summarize(state, conversation)
        if summary is not None:
            self.service.store.write(summary)
            stored.append(summary)
            retired.extend(summary.supersedes)

        self._retire(retired, state)
        self._index(stored)
        return tuple(decisions)

    def _proposals(self, state: AgentState) -> Sequence[MemoryCandidate]:
        """Ask the proposer, and treat every failure as "nothing to remember".

        The turn has already answered the user. A provider outage at this point
        is a reason to remember nothing, not a reason to fail a request that
        succeeded -- the same severity judgement
        :func:`~agentic_erp_assistant.context.compact.compact_conversation` makes
        about its summarizer.
        """
        if self.service.proposer is None:
            return ()
        try:
            return self.service.proposer.propose(
                state, required_scope=self.service.required_scope
            )
        except Exception as error:  # noqa: BLE001 - see above
            logger.warning(
                "the memory proposer failed on run %s (%s: %s); remembering "
                "nothing from this turn",
                state.trace_id,
                type(error).__name__,
                error,
            )
            return ()

    def _record(
        self,
        candidate: MemoryCandidate,
        identifier: str,
        state: AgentState,
        supersedes: Sequence[str],
    ) -> MemoryRecord:
        """The candidate as a stored record, with the scope filling in the rest."""
        return MemoryRecord(
            memory_id=identifier,
            kind=candidate.kind,
            key=candidate.key,
            statement=candidate.statement,
            project_code=self.scope.project_code,
            required_scope=candidate.required_scope,
            actor=self.scope.actor,
            session_id=self.scope.session_id,
            recorded_in_run=state.trace_id,
            recorded_at=self.service.now(),
            confidence=candidate.confidence,
            supersedes=tuple(supersedes),
        )

    def _summarize(
        self, state: AgentState, conversation: Mapping[str, object] | None
    ) -> MemoryRecord | None:
        """Refresh this session's summary, superseding the previous one."""
        if conversation is None:
            return None

        previous = [
            record.memory_id
            for record in self.service.store.live(
                self.scope, kinds=("session_summary",)
            )
            if record.key == SESSION_SUMMARY_KEY
        ]
        return summarize_session(
            compact_conversation(conversation),
            scope=self.scope,
            required_scope=self.service.required_scope,
            memory_id=f"summary-{self.scope.session_id}-{state.trace_id}",
            recorded_in_run=state.trace_id,
            recorded_at=self.service.now(),
            supersedes=tuple(sorted(previous)),
        )

    def _retire(self, memory_ids: Sequence[str], state: AgentState) -> None:
        """Supersede what the writes replaced, and audit a forget for each.

        The store is authoritative and the index follows: if the index call
        fails it says so and hydration drops the record anyway, so recall runs
        degraded rather than wrong.
        """
        if not memory_ids:
            return

        retired = self.service.store.supersede(
            list(dict.fromkeys(memory_ids)), at=self.service.now()
        )
        for record in retired:
            self.service.audit.record(
                MemoryAuditRow(
                    occurred_at=self.service.now(),
                    trace_id=state.trace_id,
                    session_id=self.scope.session_id,
                    project_code=self.scope.project_code,
                    actor=self.scope.actor,
                    memory_id=record.memory_id,
                    kind=record.kind,
                    decision="forget",
                    reason="replaced by a newer record",
                    statement_summary=summarize_statement(record.statement),
                )
            )
        self.service.index.retire([record.memory_id for record in retired])

    def _index(self, records: Sequence[MemoryRecord]) -> None:
        """Embed what was stored, in one request, and index it.

        One request for the batch rather than one per record: the cost decision
        :meth:`~agentic_erp_assistant.rag.ports.EmbeddingsPort.embed` leaves to
        its caller, made here.

        A failure leaves the records stored and unsearchable, which is the right
        way round -- the store is the record, and the index can be rebuilt from
        it. Logged, never raised: the turn has already answered.
        """
        if not records:
            return
        try:
            batch = self.service.embeddings.embed(
                [record.statement for record in records]
            )
            self.service.index.ensure_ready(batch.dimensions)
            self.service.index.upsert(records, batch.vectors)
        except Exception as error:  # noqa: BLE001 - see above
            logger.warning(
                "could not index %d newly stored memor(y|ies) (%s: %s); they are "
                "in the store and will not be found semantically until it is "
                "rebuilt",
                len(records),
                type(error).__name__,
                error,
            )

    def _audit(
        self,
        state: AgentState,
        verdict: MemoryDecision,
        kind: str,
        identifier: str,
        statement: str,
    ) -> None:
        self.service.audit.record(
            MemoryAuditRow(
                occurred_at=self.service.now(),
                trace_id=state.trace_id,
                session_id=self.scope.session_id,
                project_code=self.scope.project_code,
                actor=self.scope.actor,
                memory_id=identifier,
                kind=kind,  # type: ignore[arg-type]
                decision=verdict.decision,
                rejection=verdict.rejection,
                reason=verdict.reason,
                statement_summary=summarize_statement(statement),
            )
        )

    # -- the task in flight ------------------------------------------------

    def open_intent(self) -> IntentState | None:
        """The task this conversation is working on, or ``None``."""
        return self.service.store.open_intent(self.scope)

    def start_intent(
        self, goal: str, *, intent_id: str, unresolved_slots: Sequence[str] = ()
    ) -> IntentState:
        """Open a task, switching away from whatever was open.

        Switching rather than refusing, because a session that already has a
        task and is given a new goal has *changed* task -- and
        :meth:`~agentic_erp_assistant.memory.intent.IntentState.switch_to`
        returns the closed old one alongside the new, so both are saved and no
        slot is carried across.
        """
        at = self.service.now()
        current = self.open_intent()
        if current is not None:
            previous, fresh = current.switch_to(
                goal, intent_id=intent_id, at=at, unresolved_slots=unresolved_slots
            )
            self.service.store.save_intent(previous)
            self.service.store.save_intent(fresh)
            return fresh

        fresh = IntentState(
            intent_id=intent_id,
            goal=goal,
            project_code=self.scope.project_code,
            required_scope=self.service.required_scope,
            actor=self.scope.actor,
            session_id=self.scope.session_id,
            unresolved_slots=tuple(unresolved_slots),
            opened_at=at,
            updated_at=at,
        )
        self.service.store.save_intent(fresh)
        return fresh

    def advance_intent(self, slots: Mapping[str, str]) -> IntentState | None:
        """Fill some slots on the open task, or do nothing when there is none."""
        current = self.open_intent()
        if current is None:
            return None
        advanced = current.continue_with(slots, at=self.service.now())
        self.service.store.save_intent(advanced)
        return advanced

    def close_intent(self) -> IntentState | None:
        """Finish the open task, or do nothing when there is none."""
        current = self.open_intent()
        if current is None:
            return None
        closed = current.close(at=self.service.now())
        self.service.store.save_intent(closed)
        return closed
