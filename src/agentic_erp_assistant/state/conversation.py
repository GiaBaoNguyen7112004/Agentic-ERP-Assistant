"""One finished, or paused, turn -- as it will be shown to a later turn.

This is not a memory kind. :mod:`agentic_erp_assistant.state.memory` refuses on
purpose to keep a transcript (its own docstring: "the transcript is the thing it
refuses to keep"), and :mod:`agentic_erp_assistant.memory.policy` exists to keep
a model's own proposals about what to remember from becoming instructions. A
``ConversationTurn`` is neither: it is a bounded window of the same principal's
own words, kept verbatim because trimming it would make the assistant misquote
the user, and it is never judged by ``policy.decide`` -- its defence is
structural (its own prompt role, no locator, nothing shaped like a citation),
not a policy that might refuse it.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.approval import ApprovalDecision

if TYPE_CHECKING:
    from agentic_erp_assistant.state.agent_state import AgentState

__all__ = ["ConversationTurn"]


class ConversationTurn(BaseModel):
    """One turn of a session, projected for the next turn to be shown.

    Frozen and ``extra="forbid"``, for the same reason every other record
    crossing a node boundary is: a field smuggled in here is context a later
    turn is shown that nothing validated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    request: str = Field(min_length=1)
    response: str | None = None
    route: DecisionRoute | None = None
    failure: FailureMode = "none"
    tool_name: str | None = Field(default=None, min_length=1)
    approval: ApprovalDecision = "not_required"
    started_at: datetime
    finished_at: datetime

    @property
    def paused(self) -> bool:
        """Whether this turn is still waiting on a human, not yet settled."""
        return self.route == "request_approval" and self.approval == "pending"

    @property
    def settled(self) -> bool:
        """The opposite of :attr:`paused` -- finished, one way or another."""
        return not self.paused

    @model_validator(mode="after")
    def _must_describe_a_turn_that_could_exist(self) -> "ConversationTurn":
        if self.finished_at < self.started_at:
            raise ValueError(
                "finished_at: cannot precede started_at; one of the two "
                "clocks is wrong"
            )
        if self.paused and self.tool_name is None:
            raise ValueError(
                "tool_name: a turn waiting for approval must name the tool "
                "it is waiting to run"
            )
        return self

    @classmethod
    def from_state(
        cls,
        state: "AgentState",
        *,
        started_at: datetime,
        finished_at: datetime,
    ) -> "ConversationTurn":
        """Project a turn's state into what a later turn will be shown.

        Raises:
            ValueError: ``state.session_id`` is ``None``. A turn outside a
                session is never history -- there is no window for it to join.
        """
        if state.session_id is None:
            raise ValueError(
                f"session_id: run {state.trace_id!r} belongs to no session; "
                f"a turn with no session cannot become another turn's history"
            )
        return cls(
            trace_id=state.trace_id,
            session_id=state.session_id,
            actor=state.actor,
            request=state.request,
            response=state.response,
            route=state.route,
            failure=state.failure,
            tool_name=state.tool_name,
            approval=state.approval,
            started_at=started_at,
            finished_at=finished_at,
        )
