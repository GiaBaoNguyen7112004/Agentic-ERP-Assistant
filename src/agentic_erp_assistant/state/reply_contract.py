"""What a complete reply to this turn's question must rest on, declared by the
planner before the graph runs.

ADR 0020 measured a real routing gap: "why is milestone M2 late and by how
much" is two questions -- one the ERP answers as a field, one only a project
document answers as a passage -- and gpt-4o deterministically chose the tool
and stopped, delivering a grounded but incomplete reply. Nothing in the graph
could tell, because nothing typed said what the reply needed in the first
place. This module is that typed thing (ADR 0021).

Why a contract, not a route
----------------------------

The planner already names one route per decision
(:mod:`agentic_erp_assistant.reasoning.decision`), and that route is what
*happens* next. A ``ReplyContract`` is a different kind of fact: what the
finished reply has to be able to point to, decided once, before any action has
been taken, and checked -- deterministically, never by asking a second model to
grade the first -- against everything the turn ends up doing. It is
declaration and verification kept apart, the same separation
:mod:`agentic_erp_assistant.reasoning.decision` already draws between a typed
route and the free-text ``rationale`` beside it.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["EMPTY_CONTRACT", "ReplyContract", "ReplyNeed"]


ReplyNeed = Literal[
    "document_passage",  # something quoted from a project document
    "erp_field",          # a live value the ERP holds
]
"""The two kinds of fact a reply can need to rest on.

Closed, and only two members, for the same reason
:data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute` is closed: a
need nobody checks for has to fail at declaration, not appear later as a kind
of incompleteness nothing in the graph was ever told to look for. A write (a
risk to record), a refusal, and a clarification all declare no need at all --
``needs=frozenset()`` is a real, complete answer for those, not a missing one.
"""


class ReplyContract(BaseModel):
    """One turn's declaration of what its reply must rest on.

    Frozen, like every other object that crosses a node boundary in this
    project: a contract declared once and then checked against, over and
    over, as the turn proceeds must not be a contract that could have changed
    underneath the check. ``extra="forbid"`` for the same reason every other
    model in ``state/`` forbids it -- an unmodelled field is an input the
    completeness check never sees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs: frozenset[ReplyNeed] = frozenset()
    """What kinds of fact the reply must rest on. Empty is a real answer --
    a write, a refusal, a clarification -- not an unset field."""

    document_query: str | None = Field(default=None, min_length=1)
    """What a document search would have to find, in the user's own terms.

    Required exactly when ``"document_passage" in needs`` and rejected
    otherwise -- the same "a field that means something only on the routes
    that use it" rule
    :class:`~agentic_erp_assistant.reasoning.decision.ReasoningDecision`
    already applies to ``search_query``. A contract that needs a passage but
    names no query would leave the redirect
    (:mod:`agentic_erp_assistant.reasoning.completeness`) nothing to search
    for; a contract that names one without needing a passage is an input no
    check will ever read.
    """

    @model_validator(mode="after")
    def _query_matches_the_need(self) -> "ReplyContract":
        needs_passage = "document_passage" in self.needs
        if needs_passage and self.document_query is None:
            raise ValueError(
                "document_query: a contract that needs a document_passage "
                "must say what a search would have to find"
            )
        if not needs_passage and self.document_query is not None:
            raise ValueError(
                "document_query: present without document_passage in needs, "
                "so it is an input no redirect will ever read"
            )
        return self


EMPTY_CONTRACT = ReplyContract(needs=frozenset())
"""What an unreadable or absent declaration becomes.

Not ``None``: :attr:`~agentic_erp_assistant.state.agent_state.AgentState.contract`
being ``None`` means "this turn was never checked" (a replay, a hand-built
test state, a declaration that raised). This constant means "the turn *was*
checked, and the check found nothing to hold it to" -- the honest reading of a
declaration call that came back unreadable, per
:meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare`'s
never-fail rule.
"""
