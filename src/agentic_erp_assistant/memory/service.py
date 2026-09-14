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

**What a session amounts to**, is decided in
:mod:`agentic_erp_assistant.memory.promotion`, not here.
:meth:`SessionMemory.consolidate` takes the turns
:class:`~agentic_erp_assistant.memory.conversation.ConversationMemory` has
evicted from the short-term window and, when there are any, folds them into
the session's one summary -- a model may propose what they are worth, and only
:func:`~agentic_erp_assistant.memory.promotion.conversation_state` and the
allow-list decide what of that survives.
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
from agentic_erp_assistant.memory.promotion import (
    SessionSummaryProposal,
    SessionSummaryProposerPort,
    conversation_state,
)
from agentic_erp_assistant.memory.store import MemoryStorePort
from agentic_erp_assistant.memory.summary import SESSION_SUMMARY_KEY, summarize_session
from agentic_erp_assistant.memory.vector_store import MemoryVectorStorePort
from agentic_erp_assistant.rag.ports import EmbeddingsPort
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
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


def _established(state: AgentState) -> bool:
    """Only a turn that answered can have established anything.

    A refusal, a clarification and a failure all end with the user no better
    informed than they were -- and a turn whose output was "I could not answer
    that" is exactly where the absence-claim junk came from: a proposer asked
    what the turn was worth inventing something worth remembering about a turn
    that remembered nothing. Route and failure are typed fields, so the test is
    two comparisons, not prose matching.
    """
    return state.route == "answer" and state.failure == "none"


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

    summary_proposer: SessionSummaryProposerPort | None = None
    """What nominates a session summary for turns leaving the short-term
    window. ``None`` is a complete configuration, not a degraded one: a
    session still gets a summary written for it, folded structurally by
    :func:`~agentic_erp_assistant.memory.promotion.structural_state` alone --
    see that module for what code can say with no model involved."""

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
        evicted: Sequence[ConversationTurn] = (),
    ) -> tuple[MemoryDecision, ...]:
        """Decide what the finished turn was worth, and write all of it down.

        Args:
            state: The turn, terminal. Consolidating a paused turn would store
                facts from a decision nobody has made yet, which is why the
                caller checks first.
            evicted: Turns the short-term window no longer has room for, once
                this one joins it -- see
                :meth:`~agentic_erp_assistant.memory.conversation.ConversationMemory.evicted`.
                When non-empty, they are folded into the session's summary
                through :mod:`agentic_erp_assistant.memory.promotion`. Empty is
                the common case and skips that step entirely.

        Returns:
            One decision per candidate, in the order they were proposed, with
            the summary's own decision appended when ``evicted`` was
            non-empty. Rejections are included: they are the half of the
            record that shows the policy working.
        """
        decisions: list[MemoryDecision] = []
        stored: list[MemoryRecord] = []
        retired: list[str] = []

        if _established(state):
            judged, judged_stored, judged_retired = self._judge_proposals(
                state, list(self.service.store.live(self.scope))
            )
            decisions.extend(judged)
            stored.extend(judged_stored)
            retired.extend(judged_retired)
        else:
            # D4/D5 (the memory refactor): a turn that refused, asked for
            # clarification or failed established nothing, so there is nothing
            # to propose and the proposer is never asked -- the model call is
            # the cost this skip exists to avoid. The rejection is still a
            # returned decision (the trace's memory_rejected event derives from
            # it), but no audit row is written for it: that table's
            # ``memory_id``/``kind`` describe a candidate, and there is none.
            logger.info(
                "run %s ended %s; nothing to propose", state.trace_id, state.route
            )
            decisions.append(
                MemoryDecision(
                    decision="reject",
                    rejection="not_established",
                    reason=(
                        f"turn ended in {state.route} ({state.failure}); a turn "
                        f"that produced no answer established nothing"
                    ),
                )
            )

        summary, verdict = self._promote(state, evicted)
        if verdict is not None:
            decisions.append(verdict)
            self._audit(
                state,
                verdict,
                "session_summary",
                summary.memory_id if summary is not None
                else f"summary-{self.scope.session_id}-{state.trace_id}",
                summary.statement if summary is not None else "",
            )
        if summary is not None:
            self.service.store.write(summary)
            stored.append(summary)
            retired.extend(summary.supersedes)

        self._retire(retired, state)
        self._index(stored)
        return tuple(decisions)

    def _judge_proposals(
        self,
        state: AgentState,
        existing: list[MemoryRecord],
    ) -> tuple[list[MemoryDecision], list[MemoryRecord], list[str]]:
        """Ask the proposer, judge every proposal, and write the verdicts down.

        The proposal half of :meth:`consolidate`, extracted so the skip for an
        unestablished turn can leave it out entirely -- a turn that established
        nothing is never asked -- while the promotion half still runs.

        Returns:
            ``(decisions, stored, retired)``, in the order the candidates were
            proposed. Rejections are included: they are the half of the record
            that shows the policy working.
        """
        decisions: list[MemoryDecision] = []
        stored: list[MemoryRecord] = []
        retired: list[str] = []

        for candidate in self._proposals(state):
            verdict = decide(candidate, existing=existing, scope=self.scope)
            decisions.append(verdict)

            if not verdict.stores:
                identifier = memory_id(candidate, self.scope)
                self._audit(state, verdict, candidate.kind, identifier, candidate.statement)
                continue

            if verdict.key is not None and verdict.key != candidate.key:
                # A same-topic preference update names the key the store
                # already had, not the one just proposed -- adopt it before
                # deriving the id, so what is stored (and its id) reflects
                # what is actually kept, and the key stops drifting to a new
                # value on every rewrite.
                candidate = candidate.model_copy(update={"key": verdict.key})
            identifier = memory_id(candidate, self.scope)

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

        return decisions, stored, retired

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

    def _promote(
        self, state: AgentState, evicted: Sequence[ConversationTurn]
    ) -> tuple[MemoryRecord | None, MemoryDecision | None]:
        """Fold evicted turns into the session summary, or do nothing.

        Returns:
            ``(None, None)`` when ``evicted`` is empty -- there is nothing to
            ask about and nothing to decide. Otherwise a record to write (or
            ``None`` if nothing durable survived the allow-list and the safety
            check) paired with the :class:`~agentic_erp_assistant.memory.models.MemoryDecision`
            that explains it.
        """
        if not evicted:
            return None, None

        previous = next(
            (
                record
                for record in self.service.store.live(
                    self.scope, kinds=("session_summary",)
                )
                if record.key == SESSION_SUMMARY_KEY
            ),
            None,
        )
        proposal = self._summary_proposal(state, evicted, previous)
        mapping = conversation_state(evicted, proposal)
        summary = self._summarize(
            state, mapping, links=[turn.trace_id for turn in evicted]
        )

        if summary is None:
            return None, MemoryDecision(
                decision="reject",
                rejection="not_relevant",
                reason=f"no durable residue in {len(evicted)} evicted turn(s)",
            )

        decision = "update" if summary.supersedes else "write"
        return summary, MemoryDecision(
            decision=decision,
            supersedes=summary.supersedes,
            reason=f"session summary folded {len(evicted)} evicted turn(s)",
        )

    def _summary_proposal(
        self,
        state: AgentState,
        evicted: Sequence[ConversationTurn],
        previous: MemoryRecord | None,
    ) -> SessionSummaryProposal | None:
        """Ask the session-summary proposer, treating any failure as "nothing
        to add" -- the same severity judgement :meth:`_proposals` makes."""
        if self.service.summary_proposer is None:
            return None
        try:
            return self.service.summary_proposer.propose(evicted, previous=previous)
        except Exception as error:  # noqa: BLE001 - see above
            logger.warning(
                "the session summary proposer failed on run %s (%s: %s); "
                "folding %d evicted turn(s) structurally instead",
                state.trace_id,
                type(error).__name__,
                error,
                len(evicted),
            )
            return None

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
        self,
        state: AgentState,
        conversation: Mapping[str, object] | None,
        *,
        links: Sequence[str] = (),
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
            links=tuple(links),
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
