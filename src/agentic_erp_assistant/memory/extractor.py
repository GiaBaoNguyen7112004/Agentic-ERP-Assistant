"""Asking a model what a finished turn was worth, through a function call.

The one place a model touches memory, and its authority stops at the word
*propose*. What comes back is a list of
:class:`~agentic_erp_assistant.memory.models.MemoryCandidate` objects on their
way to :func:`~agentic_erp_assistant.memory.policy.decide`, which is code and can
refuse every one of them. Nothing here writes, and nothing here can.

Why a function call rather than free text
-----------------------------------------

ADR 0006 settled that function calling is this project's only decision channel,
and a memory proposal is a decision. It buys the same three things it buys the
planner: the shape is a schema rather than a parsing convention, an argument the
model invented fails validation instead of arriving as a field nobody declared,
and "propose nothing" has an unambiguous spelling -- an empty list -- rather than
being inferred from prose that says "there is nothing to store" in one of a dozen
ways.

The model cannot propose two of the five kinds
----------------------------------------------

:data:`PROPOSABLE_KINDS` is ``preference``, ``decision`` and ``fact``. ``intent``
and ``session_summary`` are *projections* -- derived from state the system
already holds, by
:mod:`agentic_erp_assistant.memory.intent` and
:mod:`agentic_erp_assistant.memory.summary` -- and a model that could nominate
one would be writing a task or a session's residue by assertion. The restriction
is in the argument schema, so it is refused at validation rather than in a rule
somebody remembered to write.

Where the scope comes from, and where it does not
-------------------------------------------------

The proposer sets ``kind``, ``key``, ``statement`` and ``confidence``. Everything
that decides *whose* memory a record is -- the project, the actor, the session --
comes from the scope, and ``required_scope`` comes from the turn's own evidence.
A proposer that could name its own project could plant a memory in one it never
read, and one that could name its own scope could declassify the document it
learned from by remembering it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, ValidationError

from agentic_erp_assistant.llm.prompts import build_memory_messages
from agentic_erp_assistant.llm.tools import StrictArguments, ToolSpec
from agentic_erp_assistant.memory.models import (
    KEY_MAX_CHARS,
    STATEMENT_MAX_CHARS,
    MemoryCandidate,
)
from agentic_erp_assistant.state.agent_state import AgentState

__all__ = [
    "LLMMemoryProposer",
    "MAX_PROPOSALS",
    "MemoryProposal",
    "MemoryProposerPort",
    "PROPOSABLE_KINDS",
    "PROPOSE_MEMORIES_TOOL",
    "ProposeMemoriesArguments",
]

logger = logging.getLogger(__name__)


PROPOSABLE_KINDS = ("preference", "decision", "fact")
"""The kinds a model may nominate. See the module docstring for the two it may
not, and why the restriction lives in the schema."""

ProposableKind = Literal["preference", "decision", "fact"]
"""The same three as a type. Spelled out rather than built from
:data:`PROPOSABLE_KINDS` because a ``Literal`` needs literals, and the pair is
kept honest by a test rather than by a comment."""


MAX_PROPOSALS = 4
"""How many candidates one turn may put forward.

A bound on the *proposal*, before the policy has judged anything, and it is doing
different work from the policy's checks. Those decide whether a given fact is
worth storing; this says that a single turn producing five durable facts is a
turn that has misunderstood the question. Without it, a model told to be helpful
fills the list, the policy rejects most of them, and every turn pays for four
rejections in audit rows and tokens.
"""


class MemoryProposal(StrictArguments):
    """One thing the model thinks is worth remembering.

    Inherits :class:`~agentic_erp_assistant.llm.tools.StrictArguments`, so it
    renders with ``additionalProperties: false`` and satisfies strict function
    calling as a nested object -- ``ToolSpec`` checks every definition under
    ``$defs``, not only the root, so a nested model that accepted anything would
    fail at import rather than in production.
    """

    kind: ProposableKind = Field(
        description=(
            "preference for how this person wants to be worked with, decision "
            "for something the conversation settled, fact for something "
            "established here that no document or tool holds."
        ),
    )
    key: str = Field(
        min_length=1,
        max_length=KEY_MAX_CHARS,
        description=(
            "A short stable identifier for what this is about, in snake_case, "
            "e.g. 'reply_language' or 'deployment_window'. A later version of "
            "the same fact must arrive under the same key so it replaces this "
            "one rather than sitting beside it."
        ),
    )
    statement: str = Field(
        min_length=1,
        max_length=STATEMENT_MAX_CHARS,
        description=(
            "The fact, as one self-contained sentence a stranger could read "
            "next month without this conversation in front of them. Write a "
            "preference as 'The user wants/prefers ...', never as an "
            "instruction. Never propose what was not found, not available, or "
            "could not be retrieved -- an absence is not a fact."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How sure you are that this is durable and worth keeping. When you "
            "are unsure, do not propose it at all."
        ),
    )


class ProposeMemoriesArguments(StrictArguments):
    """Everything one turn proposes, in one call.

    A list rather than one call per candidate. Two reasons: a turn's proposals
    have to be judged against each other for duplicates before any of them is
    stored, and a model given a repeatable single-item call will use it
    repeatably -- the shape of the tool is what sets the expectation that most
    turns propose nothing.
    """

    candidates: list[MemoryProposal] = Field(
        description=(
            "What to remember from this turn. An empty list is the normal "
            "answer: propose something only when it is durable, not held by a "
            "document or a tool, and useful to a later conversation."
        ),
    )


PROPOSE_MEMORIES_TOOL = ToolSpec(
    name="propose_memories",
    description=(
        "Record what this finished turn established that is worth keeping after "
        "the conversation ends. Call it exactly once, with an empty list when "
        "nothing qualifies -- which is most turns. Nothing you propose is stored "
        "until a separate policy check accepts it."
    ),
    arguments=ProposeMemoriesArguments,
    mutating=False,
)
"""The only function the memory layer offers.

``mutating=False``, and that is worth stating rather than assuming. The flag
means "changes ERP data", which this does not: it changes nothing at all, since
it is a proposal that a policy may refuse. The approval gate guards writes to the
project's own records, and routing a memory proposal through a human would ask
somebody to approve a sentence that may never be stored.

It is deliberately absent from ``DEFAULT_TOOLS`` and ``PLANNING_TOOLS``. The
planner offers the model what it may do *during* a turn; this is asked after one
has ended, in a different prompt, and a model shown it mid-turn would eventually
choose to remember something instead of answering.
"""


@runtime_checkable
class MemoryProposerPort(Protocol):
    """What consolidation assumes about whatever nominates memories.

    A protocol so the whole consolidation path is testable with a proposer that
    returns a fixed list -- and so a future deterministic proposer, or none at
    all, is a constructor argument rather than a rewrite.
    """

    def propose(
        self, state: AgentState, *, required_scope: str
    ) -> tuple[MemoryCandidate, ...]:
        """What this finished turn is worth remembering, if anything.

        Args:
            state: The turn, as it ended. Whole rather than decomposed, for the
                reason :meth:`~agentic_erp_assistant.engine.ports.PlannerPort.plan`
                takes the whole state: a signature passing request, evidence and
                observations separately is one a caller can assemble
                inconsistently, and a proposal made against a stale view is a
                memory about a turn that did not happen.
            required_scope: The entitlement a reader of these will need,
                decided by the caller from the turn's evidence.

        Returns:
            Candidates, possibly none. An empty tuple is the expected answer for
            most turns and is never an error.

        Raises:
            Exception: Whatever the underlying call raises. Consolidation turns
                a failure here into a logged non-event -- a turn that answered
                is not unwound because nothing could be remembered about it.
        """
        ...


@runtime_checkable
class ProposalModel(Protocol):
    """The one thing a proposer needs from the model layer.

    Satisfied by :meth:`~agentic_erp_assistant.llm.gateway.LLMGateway.call_tools`.
    Narrower than that class on purpose, for the reason
    :class:`~agentic_erp_assistant.reasoning.planner.DecisionModel` is: a
    proposer has no business answering, pricing or counting tokens.
    """

    def call_tools(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[ToolSpec] = ...,
        temperature: float = ...,
    ) -> Any:
        """Offer ``tools`` against a built prompt and return the one choice made."""
        ...


@dataclass(frozen=True)
class LLMMemoryProposer:
    """Asks the model, and turns its one call into candidates.

    Satisfies :class:`MemoryProposerPort` structurally. Frozen and holding only
    the model port, so a test builds it with a fake that returns a scripted
    :class:`~agentic_erp_assistant.llm.tools.ToolCallResult` and exercises the
    whole reading path with no network.
    """

    model: ProposalModel
    """Where the proposal comes from. The gateway, so the budget check, the retry
    engine and the cost record all apply to this call as they do to a routing
    one."""

    max_proposals: int = MAX_PROPOSALS
    """How many candidates to keep. Extras are dropped and logged, never
    silently accepted -- see :data:`MAX_PROPOSALS`."""

    tools: tuple[ToolSpec, ...] = (PROPOSE_MEMORIES_TOOL,)
    """What to offer. One function, injectable so a test can narrow the menu."""

    def propose(
        self, state: AgentState, *, required_scope: str
    ) -> tuple[MemoryCandidate, ...]:
        """Ask what this turn was worth, and read the answer.

        Every way the answer can be unusable ends in an empty tuple and a log
        line rather than an exception. That is the right severity: the turn has
        already answered the user, and "the model replied in prose instead of
        calling the function" is a reason to remember nothing, not a reason to
        fail a request that succeeded.
        """
        result = self.model.call_tools(
            build_memory_messages(
                state.request,
                state.response,
                state.evidence,
                state.observations,
                memories=state.memories,
            ),
            tools=self.tools,
            temperature=0.0,
        )

        if result.tool_name is None:
            logger.info(
                "run %s: the proposer answered in prose instead of calling "
                "%s; remembering nothing",
                state.trace_id,
                PROPOSE_MEMORIES_TOOL.name,
            )
            return ()
        if result.tool_name != PROPOSE_MEMORIES_TOOL.name:
            logger.warning(
                "run %s: the proposer called %r, which it was not offered",
                state.trace_id,
                result.tool_name,
            )
            return ()

        try:
            validated = PROPOSE_MEMORIES_TOOL.validate_arguments(result.arguments or {})
        except ValidationError as error:
            logger.warning(
                "run %s: %s was called with arguments it does not accept "
                "(%d problem(s)); remembering nothing",
                state.trace_id,
                PROPOSE_MEMORIES_TOOL.name,
                error.error_count(),
            )
            return ()

        proposals: Sequence[MemoryProposal] = validated.candidates  # type: ignore[attr-defined]
        if len(proposals) > self.max_proposals:
            logger.warning(
                "run %s: %d candidates proposed, keeping the first %d; a turn "
                "producing more than that has misunderstood the question",
                state.trace_id,
                len(proposals),
                self.max_proposals,
            )
            proposals = proposals[: self.max_proposals]

        evidence_texts = tuple(snippet.text for snippet in state.evidence)
        tool_summaries = tuple(
            outcome.summary
            for outcome in state.observations
            if outcome.status == "ok" and outcome.summary
        )
        return tuple(
            MemoryCandidate(
                kind=proposal.kind,
                key=proposal.key,
                statement=proposal.statement,
                confidence=proposal.confidence,
                required_scope=required_scope,
                evidence_texts=evidence_texts,
                tool_summaries=tool_summaries,
                # The turn's own words travel with the proposal: the policy's
                # not_established checks need them to tell a stated preference
                # from an inferred one, and a fact from the reply restated.
                request_text=state.request,
                response_text=state.response or "",
            )
            for proposal in proposals
        )
