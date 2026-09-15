"""What each tool actually does, once policy has already said yes.

A handler is the smallest thing in the tool layer: it receives arguments that
have already been validated, reads or writes the ERP, and reports what it found
and what it read it from. It checks no permission, asks for no approval, counts
no attempts and builds no :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`.
All of that is the gateway's, in one ordered place -- a handler that also
enforced policy would be a second place the rule lives, and the one that gets
forgotten when the rule changes.

Which is why they return :class:`HandlerResult` and not an outcome. A handler
knows the two things only it can know -- what to say, and what it read -- and
the gateway supplies everything it alone knows: the status, and how many
attempts it took to get here.

Failure is raised here, not returned
------------------------------------

Inside this boundary an exception is the right shape:
:class:`~agentic_erp_assistant.tools.models.TransientToolError` for something
worth trying again, :class:`~agentic_erp_assistant.tools.models.ToolError` for
something that will fail the same way next time. The gateway catches both and
turns them into statuses, so nothing raised here ever reaches the graph.
"""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.erp.mock import ErpAccessError, ErpNotPersistedError, MockErp
from agentic_erp_assistant.llm.tools import (
    BudgetSummaryArguments,
    CreateRiskArguments,
    ListRisksArguments,
    ProjectStatusArguments,
    SprintProgressArguments,
)
from agentic_erp_assistant.tools.models import (
    ExecutionContext,
    ToolError,
    TransientToolError,
)

__all__ = ["build_handlers", "Handler", "HandlerResult"]


class HandlerResult(BaseModel):
    """What a handler reports: one line, and what it read to say it.

    Frozen, and only two fields. Everything else on the eventual outcome is
    decided by the gateway, so a handler cannot claim a status it is not in a
    position to judge -- "ok" is a statement about policy and retries as much
    as about data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    """What to tell the model, in a sentence it can quote."""

    source_ids: tuple[str, ...] = Field(min_length=1)
    """The records this was read from, or written to.

    At least one, enforced here rather than only at the outcome: a handler that
    forgot its sources should fail where the mistake is, not two layers later
    where the message is about a model that could not be constructed.
    """


Handler = Callable[[BaseModel, ExecutionContext], HandlerResult]
"""A validated-arguments-in, result-out function, also told whose call this is
and which project to read through. Bound to a store when the registry is
built, so the registry entry is complete and nothing has to be looked up at
call time."""


def build_handlers(erp: MockErp) -> dict[str, Handler]:
    """Bind every handler to one ERP store and return them by tool name.

    A factory rather than module-level functions reading a global. Two tests
    must not be able to see each other's writes, and a test that wants three
    milestones instead of the repo fixture should be able to have them.
    """

    def get_project_status(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        assert isinstance(arguments, ProjectStatusArguments)
        store = erp.for_project(context.project_code)
        milestone = store.milestone(arguments.milestone_id)
        if milestone is None:
            raise ToolError(f"no milestone {arguments.milestone_id!r} exists")

        late = (
            f"{milestone.days_late} day{'s' if milestone.days_late != 1 else ''} late"
            if milestone.days_late
            else "not late"
        )
        return HandlerResult(
            summary=(
                f"{milestone.milestone_id} ({milestone.title}) is "
                f"{milestone.schedule_status.replace('_', ' ')} and {late}; "
                f"due {milestone.due_on}."
            ),
            source_ids=(milestone.source_id,),
        )

    # A counter in a closure, not a random draw: "fails once, then works" has to
    # be the same on every run, or the test that proves the retry budget works
    # is a test that sometimes proves nothing.
    flaky_calls = {"count": 0}

    def get_project_status_flaky(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        flaky_calls["count"] += 1
        if flaky_calls["count"] == 1:
            raise TransientToolError(
                "the ERP status endpoint timed out; it is usually up again "
                "immediately"
            )
        return get_project_status(arguments, context)

    def get_sprint_progress(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        assert isinstance(arguments, SprintProgressArguments)
        store = erp.for_project(context.project_code)
        sprint = store.sprint(arguments.sprint_id)
        if sprint is None:
            raise ToolError(f"no sprint {arguments.sprint_id!r} exists")

        return HandlerResult(
            summary=(
                f"{sprint.sprint_id}: {sprint.points_completed} of "
                f"{sprint.points_committed} points complete with "
                f"{sprint.days_remaining} day"
                f"{'s' if sprint.days_remaining != 1 else ''} remaining."
            ),
            source_ids=(sprint.source_id,),
        )

    def get_budget_summary(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        assert isinstance(arguments, BudgetSummaryArguments)
        store = erp.for_project(context.project_code)
        budget = store.budget(arguments.project_id)
        if budget is None:
            raise ToolError(f"no budget recorded for project {arguments.project_id!r}")

        consumed = budget.spent_usd / budget.approved_usd if budget.approved_usd else 0.0
        summary = (
            f"{arguments.project_id}: ${budget.spent_usd:,.0f} of "
            f"${budget.approved_usd:,.0f} approved ({consumed:.0%}) as of "
            f"{budget.as_of}."
        )
        if arguments.include_forecast:
            summary += (
                f" Forecast at completion ${budget.forecast_at_completion_usd:,.0f}."
            )

        return HandlerResult(summary=summary, source_ids=(budget.source_id,))

    def list_risks(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        assert isinstance(arguments, ListRisksArguments)
        store = erp.for_project(context.project_code)
        project = store.project(arguments.project_id)
        if project is None:
            raise ToolError(f"no project {arguments.project_id!r} exists")

        # Open rows only -- the tool description promises "the open risks",
        # and the dataset mirrors the register, which keeps closed ones on
        # file. The store does the filtering so the rule is not a check this
        # handler has to remember.
        risks = store.open_risks_for(arguments.project_id)
        if risks:
            listed = "; ".join(f"{risk.risk_id} ({risk.severity}) {risk.title}" for risk in risks)
            summary = f"{len(risks)} open risk{'s' if len(risks) != 1 else ''}: {listed}"
        else:
            summary = f"No open risks recorded for {arguments.project_id}."

        # The project's own record is always cited, so an empty register still
        # says where the emptiness was read from. Without it, "no risks" would
        # be the one successful answer that could not be cited.
        return HandlerResult(
            summary=summary,
            source_ids=(project.source_id, *(risk.source_id for risk in risks)),
        )

    def create_risk(
        arguments: BaseModel, context: ExecutionContext
    ) -> HandlerResult:
        assert isinstance(arguments, CreateRiskArguments)
        store = erp.for_project(context.project_code)
        if store.project(arguments.project_id) is None:
            raise ToolError(f"no project {arguments.project_id!r} exists")

        try:
            created = store.create_risk(
                project_id=arguments.project_id,
                title=arguments.title,
                severity=arguments.severity,
            )
        except (ErpNotPersistedError, ErpAccessError) as error:
            # Re-raised as a ToolError because the gateway catches exactly the
            # two tool-layer exceptions and nothing else; a store failure that
            # unwound past it would violate "nothing escapes the gateway" --
            # and both are permanent by nature, so no retry budget is spent
            # proving either. ErpAccessError should never actually fire here:
            # the gateway's project check refuses a mismatched call before
            # this handler is reached. Caught anyway as defence in depth, the
            # same reason the view itself still checks.
            raise ToolError(str(error)) from error
        return HandlerResult(
            summary=(
                f"Recorded {created.risk_id} ({created.severity}) against "
                f"{created.project_id}: {created.title}"
            ),
            source_ids=(created.source_id,),
        )

    return {
        "get_project_status": get_project_status,
        "get_project_status_flaky": get_project_status_flaky,
        "get_sprint_progress": get_sprint_progress,
        "get_budget_summary": get_budget_summary,
        "list_risks": list_risks,
        "create_risk": create_risk,
    }
