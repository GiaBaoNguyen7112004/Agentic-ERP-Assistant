"""Whether a turn has what its own declared reply contract said it needs
(ADR 0021) -- a pure function of typed state, and nothing else.

This is the check half of the mechanism ``reasoning/planner.py::Planner.declare``
is the declaration half of. Deliberately narrow, on the same grounds
:mod:`agentic_erp_assistant.reasoning.decision` draws around ``rationale`` and
:mod:`agentic_erp_assistant.state.events` draws around ``detail``: a check that
read the reply's text, or asked a second model whether the first one's answer
"felt" complete, would be exactly the kind of judgement this project keeps out
of prose. :func:`assess` reads two things -- whether retrieval put anything in
``state.evidence`` and whether any of this turn's own tool calls in
``state.observations`` succeeded -- and nothing else. Deterministic, free, and
the same answer every time it is asked about the same state.

Why an ``ok`` observation, not a *read-only* one
-------------------------------------------------

``erp_field`` is satisfied by any observation with ``status == "ok"``, mutating
tools included. A write's own receipt -- ``create_risk -> ok: Recorded R-6
against orion.`` -- is a live ERP fact exactly the way a read is: the risk now
exists, with the id the response names, and that is precisely the kind of thing
``erp_field`` means. Narrowing this to read-only tools would make a contract
declaring ``erp_field`` for "record a risk and tell me its id" impossible to
satisfy by the write itself, which is backwards -- the declaration step already
handles this the simpler way, by declaring no needs at all for a request that is
a write (see ``eval/routing_cases.py``'s A1/A9 cases and the live check in
``docs/completeness-plan.md`` Phase P2). This function does not have to
duplicate that judgement; it only has to be right when a contract *does* name a
need and a write happens to be what satisfies it.
"""

from dataclasses import dataclass

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.reply_contract import ReplyContract, ReplyNeed

__all__ = ["Completeness", "REDIRECT_ORDER", "assess", "next_redirect"]


REDIRECT_ORDER: tuple[ReplyNeed, ...] = ("erp_field", "document_passage")
"""Which missing need gets redirected first, when more than one is missing.

The field before the passage (D5 in ``docs/completeness-plan.md``):
composition happens in ``retrieve_and_answer``, at the end of the turn, so a
compound question's ERP half should already be sitting in ``observations``
by the time retrieval runs and the composer is asked to write from both. The
reverse order would let a retrieval-first turn compose without the field it
still needs, and then have nowhere left to fetch it from -- ``engine/
transitions.py`` deliberately has no edge from an answered retrieval back to
a tool call.
"""


@dataclass(frozen=True)
class Completeness:
    """The result of checking one state against one contract.

    Frozen, like every other object that crosses this boundary: a caller that
    read ``missing`` and then acted on stale advice from a later call would be
    exactly the bug immutability elsewhere in this project exists to prevent.
    """

    satisfied: frozenset[ReplyNeed]
    """Needs the contract named that this state already rests on."""

    missing: frozenset[ReplyNeed]
    """Needs the contract named that this state does not yet rest on."""

    @property
    def complete(self) -> bool:
        """Whether every declared need is satisfied. ``True`` for a contract
        that named none, which is a real, met contract -- not an unset one."""
        return not self.missing


def assess(contract: ReplyContract | None, state: AgentState) -> Completeness:
    """Check ``state`` against what ``contract`` said a complete reply needs.

    Args:
        contract: What was declared, or ``None`` for a turn that was never
            checked (see :attr:`~agentic_erp_assistant.state.agent_state.AgentState.contract`'s
            own docstring for the two ways a turn ends up here). Treated
            identically to a contract that named no needs at all -- there is
            nothing to hold either kind of turn to.
        state: The turn as it stands. Only ``evidence`` and ``observations``
            are read.

    Returns:
        A :class:`Completeness` naming exactly which of the contract's needs
        (if any) are met by what this state has actually produced so far.
    """
    if contract is None:
        return Completeness(satisfied=frozenset(), missing=frozenset())

    satisfied: set[ReplyNeed] = set()
    if "document_passage" in contract.needs and state.evidence:
        satisfied.add("document_passage")
    if "erp_field" in contract.needs and any(
        observation.status == "ok" for observation in state.observations
    ):
        satisfied.add("erp_field")

    return Completeness(
        satisfied=frozenset(satisfied),
        missing=frozenset(contract.needs) - frozenset(satisfied),
    )


def next_redirect(
    completeness: Completeness, redirected: frozenset[ReplyNeed]
) -> ReplyNeed | None:
    """Which missing need to redirect for next, or ``None`` if there is none
    left to try.

    Args:
        completeness: The result of :func:`assess` for this decision.
        redirected: Needs this turn has already spent its one redirect on
            (:attr:`~agentic_erp_assistant.state.agent_state.AgentState.redirected_needs`).
            A need in this set is never returned again -- the bound is one
            redirect per need, not one per turn, but also never a second
            attempt at the same one.

    Returns:
        The first need in :data:`REDIRECT_ORDER` that is missing and has not
        already been redirected, or ``None``.
    """
    for need in REDIRECT_ORDER:
        if need in completeness.missing and need not in redirected:
            return need
    return None
