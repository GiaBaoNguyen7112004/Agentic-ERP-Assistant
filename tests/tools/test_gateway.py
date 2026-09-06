"""The order is the safety, so the tests assert on what did not happen."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from agentic_erp_assistant.erp.mock import MockErp
from agentic_erp_assistant.llm.tools import (
    CREATE_RISK_TOOL,
    GET_PROJECT_STATUS_FLAKY_TOOL,
    GET_PROJECT_STATUS_TOOL,
    LIST_RISKS_TOOL,
    ToolSpec,
)
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.tool_request import ToolRequest
from agentic_erp_assistant.tools.gateway import GATEWAY_NODE, ToolGateway
from agentic_erp_assistant.tools.handlers import HandlerResult
from agentic_erp_assistant.tools.models import ToolError, TransientToolError
from agentic_erp_assistant.tools.registry import (
    build_default_registry,
    NO_RETRY,
    RateLimitPolicy,
    RetryPolicy,
    ToolDefinition,
    ToolRegistry,
)

WHEN = datetime(2026, 9, 5, 9, 30, tzinfo=timezone.utc)

READ_SCOPES = frozenset(
    {
        "project.status.read",
        "project.sprint.read",
        "project.budget.read",
        "project.risk.read",
    }
)
WRITE_SCOPE = frozenset({"project.risk.write"})


@pytest.fixture
def erp() -> MockErp:
    return MockErp.load()


@pytest.fixture
def events() -> list[TraceEvent]:
    return []


@pytest.fixture
def gateway(erp: MockErp, events: list[TraceEvent]) -> ToolGateway:
    """A gateway that never sleeps and never reads a real clock, so a retry
    schedule and an audit timestamp are both things a test can assert on."""
    return ToolGateway(
        build_default_registry(erp),
        on_event=events.append,
        now=lambda: WHEN,
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )


def call(tool_name: str, arguments: dict[str, object], **overrides: object) -> ToolRequest:
    fields: dict[str, object] = {
        "tool_name": tool_name,
        "arguments": arguments,
        "actor": "pm@example.com",
        "scopes": READ_SCOPES | WRITE_SCOPE,
    }
    fields.update(overrides)
    return ToolRequest(**fields)  # type: ignore[arg-type]


def new_risk(
    arguments: dict[str, object] | None = None, **overrides: object
) -> ToolRequest:
    return call(
        "create_risk",
        arguments
        if arguments is not None
        else {
            "project_id": "atlas",
            "title": "Vendor may miss the integration test window",
            "severity": "high",
        },
        **overrides,
    )


class SpyHandler:
    """Records whether it ran at all. The point of most tests in this file."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, arguments: BaseModel) -> HandlerResult:
        self.calls += 1
        return HandlerResult(summary="ran", source_ids=("spy",))


def registry_with(definition: ToolDefinition) -> ToolRegistry:
    return ToolRegistry((definition,))


# --------------------------------------------------------------------------
# The five cases the unit asks for
# --------------------------------------------------------------------------


def test_a_permission_denial_returns_at_once_with_one_attempt(erp: MockErp) -> None:
    """No retry: a missing scope is still missing on the second try, and a
    second attempt would only make the denial look like a flaky call."""
    spy = SpyHandler()
    gateway = ToolGateway(
        registry_with(
            ToolDefinition(
                spec=GET_PROJECT_STATUS_TOOL,
                required_scope="project.status.read",
                approval_required=False,
                timeout_seconds=5.0,
                retry=RetryPolicy(max_attempts=3),
                handler=spy,
            )
        ),
        sleep=lambda _: None,
    )

    outcome = gateway.execute(
        call("get_project_status", {"milestone_id": "M2"}, scopes=frozenset())
    )

    assert outcome.status == "denied"
    assert outcome.attempts == 1
    assert spy.calls == 0


def test_an_unapproved_create_risk_records_nothing(
    gateway: ToolGateway, erp: MockErp
) -> None:
    """The case the whole ordering exists for: a write that stopped for a human
    must not have already happened."""
    before = len(erp.risks_for("atlas"))

    outcome = gateway.execute(new_risk())

    assert outcome.status == "approval_required"
    assert len(erp.risks_for("atlas")) == before


def test_an_approved_create_risk_writes_once_and_is_audited(
    gateway: ToolGateway, erp: MockErp
) -> None:
    before = len(erp.risks_for("atlas"))

    outcome = gateway.execute(new_risk(approval="approved"))

    assert outcome.status == "ok"
    assert len(erp.risks_for("atlas")) == before + 1

    (row,) = gateway.audit.rows  # type: ignore[union-attr]
    assert (row.approval, row.status, row.actor) == ("approved", "ok", "pm@example.com")
    assert row.occurred_at == WHEN
    assert row.source_ids == outcome.source_ids


def test_the_flaky_read_succeeds_on_its_second_attempt(gateway: ToolGateway) -> None:
    outcome = gateway.execute(call("get_project_status_flaky", {"milestone_id": "M2"}))

    assert outcome.status == "ok"
    assert outcome.attempts == 2


def test_an_unknown_argument_raises_before_any_handler_runs() -> None:
    """The raise itself is in ToolSpec.validate_arguments, which the gateway
    calls at step 2 -- so an argument the model invented cannot reach a
    handler. extra='forbid' is what makes it a raise rather than a key that is
    quietly ignored."""
    with pytest.raises(ValidationError):
        CREATE_RISK_TOOL.validate_arguments(
            {
                "project_id": "atlas",
                "title": "x",
                "severity": "low",
                "skip_approval": True,
            }
        )


def test_the_gateway_reports_that_raise_instead_of_letting_it_escape(
    gateway: ToolGateway, erp: MockErp
) -> None:
    """A model inventing an argument is a runtime condition the turn has to
    record and route on. An exception through the graph would leave the state
    describing the attempt unwritten."""
    before = len(erp.risks_for("atlas"))

    outcome = gateway.execute(
        new_risk(
            {
                "project_id": "atlas",
                "title": "x",
                "severity": "low",
                "skip_approval": True,
            },
            approval="approved",
        )
    )

    assert outcome.status == "invalid_arguments"
    assert "skip_approval" in (outcome.error or "")
    assert len(erp.risks_for("atlas")) == before


# --------------------------------------------------------------------------
# The order, checked step by step
# --------------------------------------------------------------------------


def test_a_missing_required_argument_is_caught_before_the_handler() -> None:
    spy = SpyHandler()
    gateway = ToolGateway(
        registry_with(
            ToolDefinition(
                spec=GET_PROJECT_STATUS_TOOL,
                required_scope="project.status.read",
                approval_required=False,
                timeout_seconds=5.0,
                retry=NO_RETRY,
                handler=spy,
            )
        )
    )

    outcome = gateway.execute(call("get_project_status", {}))

    assert outcome.status == "invalid_arguments"
    assert spy.calls == 0


def test_permission_is_checked_before_approval(erp: MockErp) -> None:
    """Approval decides whether a permitted call should happen now; it can
    never grant an entitlement its holder never had. So an actor without the
    write scope is refused even holding an approval."""
    gateway = ToolGateway(build_default_registry(erp), now=lambda: WHEN)
    before = len(erp.risks_for("atlas"))

    outcome = gateway.execute(new_risk(approval="approved", scopes=READ_SCOPES))

    assert outcome.status == "denied"
    assert "project.risk.write" in (outcome.error or "")
    assert len(erp.risks_for("atlas")) == before


def test_a_denied_approval_is_not_the_same_as_an_unasked_one(
    gateway: ToolGateway,
) -> None:
    """One goes back to a human, the other ends the turn."""
    unasked = gateway.execute(new_risk())
    refused = gateway.execute(new_risk(approval="denied"))

    assert unasked.status == "approval_required"
    assert refused.status == "denied"


def test_a_pending_approval_is_still_not_permission(gateway: ToolGateway, erp: MockErp) -> None:
    before = len(erp.risks_for("atlas"))

    outcome = gateway.execute(new_risk(approval="pending"))

    assert outcome.status == "approval_required"
    assert len(erp.risks_for("atlas")) == before


def test_an_unknown_tool_is_reported_and_not_raised(gateway: ToolGateway) -> None:
    outcome = gateway.execute(call("close_milestone", {"milestone_id": "M2"}))

    assert outcome.status == "failed"
    assert "close_milestone" in (outcome.error or "")


def executable_strings(source: Path) -> set[str]:
    """Every string literal in a file except the docstrings.

    Docstrings are excluded because prose may name a tool -- explaining a rule
    with an example is not applying the rule to that example. What must not
    appear is a tool name the code itself can compare against.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    docstring_nodes = {
        id(node.body[0])
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    parents_of_docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and id(node.body[0]) in docstring_nodes
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in parents_of_docstrings
    }


def test_the_gateway_branches_on_no_tool_name_of_its_own(
    gateway: ToolGateway,
) -> None:
    """Every policy is read off the registry entry. A special case written into
    the execution path would be a rule that exists in one code path and in no
    document."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "agentic_erp_assistant"
        / "tools"
        / "gateway.py"
    )

    assert executable_strings(source).isdisjoint(gateway.registry.names())


# --------------------------------------------------------------------------
# Reads stay automatic once permission passes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_project_status", {"milestone_id": "M2"}),
        ("get_sprint_progress", {"sprint_id": "SPR-13"}),
        ("get_budget_summary", {"project_id": "atlas", "include_forecast": True}),
        ("list_risks", {"project_id": "atlas"}),
    ],
)
def test_a_permitted_read_runs_without_anyone_being_asked(
    gateway: ToolGateway, tool_name: str, arguments: dict[str, object]
) -> None:
    outcome = gateway.execute(call(tool_name, arguments))

    assert outcome.status == "ok"
    assert outcome.attempts == 1
    assert outcome.summary and outcome.source_ids


def test_a_read_produces_no_audit_row(gateway: ToolGateway) -> None:
    """An audit trail with one row per read is one where the writes are buried,
    and the writes are the reason it exists."""
    gateway.execute(call("list_risks", {"project_id": "atlas"}))

    assert gateway.audit.rows == []  # type: ignore[union-attr]


def test_the_budget_summary_cites_the_record_it_read(gateway: ToolGateway) -> None:
    outcome = gateway.execute(
        call("get_budget_summary", {"project_id": "atlas", "include_forecast": True})
    )

    assert outcome.source_ids == ("budget-summary-q3",)


# --------------------------------------------------------------------------
# Failure inside the handler
# --------------------------------------------------------------------------


def test_a_bad_identifier_comes_back_as_a_failure_not_a_crash(
    gateway: ToolGateway,
) -> None:
    outcome = gateway.execute(call("get_project_status", {"milestone_id": "M9"}))

    assert outcome.status == "failed"
    assert "M9" in (outcome.error or "")
    assert outcome.attempts == 1


def test_a_permanent_failure_does_not_spend_the_retry_budget() -> None:
    """The handler said this will fail the same way next time; repeating it
    only buys latency."""
    calls = {"n": 0}

    def always_broken(_: BaseModel) -> HandlerResult:
        calls["n"] += 1
        raise ToolError("the ERP rejected it")

    gateway = ToolGateway(
        registry_with(
            ToolDefinition(
                spec=GET_PROJECT_STATUS_TOOL,
                required_scope="project.status.read",
                approval_required=False,
                timeout_seconds=5.0,
                retry=RetryPolicy(max_attempts=3),
                handler=always_broken,
            )
        ),
        sleep=lambda _: None,
    )

    outcome = gateway.execute(call("get_project_status", {"milestone_id": "M2"}))

    assert outcome.status == "failed"
    assert calls["n"] == 1


def test_an_exhausted_budget_reports_a_transient_failure_with_its_cost() -> None:
    def always_timing_out(_: BaseModel) -> HandlerResult:
        raise TransientToolError("the ERP timed out")

    gateway = ToolGateway(
        registry_with(
            ToolDefinition(
                spec=GET_PROJECT_STATUS_TOOL,
                required_scope="project.status.read",
                approval_required=False,
                timeout_seconds=5.0,
                retry=RetryPolicy(max_attempts=3),
                handler=always_timing_out,
            )
        ),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )

    outcome = gateway.execute(call("get_project_status", {"milestone_id": "M2"}))

    assert outcome.status == "transient_failure"
    assert outcome.attempts == 3


def test_a_write_that_failed_is_still_audited(erp: MockErp) -> None:
    """A row is written on every path a gated call can take. "Approved, then
    unknown" is not evidence."""

    def refusing_backend(_: BaseModel) -> HandlerResult:
        raise ToolError("the ERP rejected the risk")

    gateway = ToolGateway(
        registry_with(
            ToolDefinition(
                spec=CREATE_RISK_TOOL,
                required_scope="project.risk.write",
                approval_required=True,
                timeout_seconds=5.0,
                retry=NO_RETRY,
                handler=refusing_backend,
            )
        ),
        now=lambda: WHEN,
    )

    gateway.execute(new_risk(approval="approved"))

    (row,) = gateway.audit.rows  # type: ignore[union-attr]
    assert (row.approval, row.status) == ("approved", "failed")


def test_a_refused_write_is_audited_too(gateway: ToolGateway) -> None:
    gateway.execute(new_risk(approval="denied"))

    (row,) = gateway.audit.rows  # type: ignore[union-attr]
    assert (row.approval, row.status) == ("denied", "denied")


def test_the_audit_line_says_what_the_call_would_do(gateway: ToolGateway) -> None:
    """Values, not just keys: an auditor came to find out what changed."""
    gateway.execute(new_risk(approval="approved"))

    (row,) = gateway.audit.rows  # type: ignore[union-attr]
    assert "project_id=atlas" in row.arguments_summary
    assert row.arguments_summary.startswith("create_risk(")


def test_a_long_argument_cannot_stretch_the_audit_line(gateway: ToolGateway) -> None:
    """The cap is the protection; a tool that takes a credential as an argument
    is the thing to fix, and no renderer can make that safe."""
    gateway.execute(
        new_risk(
            {"project_id": "atlas", "title": "x" * 500, "severity": "low"},
            approval="approved",
        )
    )

    (row,) = gateway.audit.rows  # type: ignore[union-attr]
    assert len(row.arguments_summary) <= 200


# --------------------------------------------------------------------------
# The trace
# --------------------------------------------------------------------------


def test_a_successful_call_is_traced(
    gateway: ToolGateway, events: list[TraceEvent]
) -> None:
    gateway.execute(call("list_risks", {"project_id": "atlas"}))

    assert [event.kind for event in events] == ["tool_called"]
    assert all(event.node == GATEWAY_NODE for event in events)


def test_every_retry_leaves_a_trace_record(
    gateway: ToolGateway, events: list[TraceEvent]
) -> None:
    """A retry that leaves no record is exactly the silent cost this project is
    meant to make visible."""
    gateway.execute(call("get_project_status_flaky", {"milestone_id": "M2"}))

    assert [event.kind for event in events] == ["retry_scheduled", "tool_called"]


def test_a_write_traces_both_the_approval_and_the_call(
    gateway: ToolGateway, events: list[TraceEvent]
) -> None:
    gateway.execute(new_risk(approval="approved"))

    assert [event.kind for event in events] == ["approval_recorded", "tool_called"]


def test_a_call_stopped_for_a_human_is_traced_as_such(
    gateway: ToolGateway, events: list[TraceEvent]
) -> None:
    gateway.execute(new_risk())

    assert "approval_requested" in [event.kind for event in events]


def test_a_gateway_with_no_listener_still_works(erp: MockErp) -> None:
    """The hook is optional; the checks are not."""
    gateway = ToolGateway(build_default_registry(erp))

    assert gateway.execute(call("list_risks", {"project_id": "atlas"})).status == "ok"


# --------------------------------------------------------------------------
# The budget is registry policy, and it fires before the handler
# --------------------------------------------------------------------------


class Clock:
    """A clock a test moves by hand.

    The gateway fixture pins ``now`` to a constant, which is what makes an
    audit timestamp assertable; a window needs the opposite, so this advances
    on demand. Still no wall clock anywhere: a limit that only fails on a slow
    machine is a limit nobody can debug.
    """

    def __init__(self, start: datetime = WHEN) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


STATUS_ARGS = {"milestone_id": "M2"}


def limited(
    spec: ToolSpec,
    handler: object,
    *,
    max_calls: int = 1,
    per_seconds: float = 60.0,
    scope: str = "project.status.read",
    approval_required: bool = False,
    retry: RetryPolicy = NO_RETRY,
) -> ToolDefinition:
    """One tool whose only unusual policy is a budget small enough to hit."""
    return ToolDefinition(
        spec=spec,
        required_scope=scope,
        approval_required=approval_required,
        timeout_seconds=5.0,
        retry=retry,
        rate_limit=RateLimitPolicy(max_calls=max_calls, per_seconds=per_seconds),
        handler=handler,  # type: ignore[arg-type]
    )


def gateway_over(
    *definitions: ToolDefinition,
    events: list[TraceEvent] | None = None,
    clock: Clock | None = None,
) -> ToolGateway:
    return ToolGateway(
        ToolRegistry(definitions),
        on_event=(events.append if events is not None else None),
        now=clock if clock is not None else (lambda: WHEN),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )


def test_the_call_at_the_limit_runs_and_the_next_never_reaches_the_handler(
    events: list[TraceEvent],
) -> None:
    """The acceptance case. The assertion that matters is ``spy.calls``: a
    limit checked after execution would return the same outcome having already
    served the load it exists to refuse."""
    spy = SpyHandler()
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, spy, max_calls=1), events=events
    )

    allowed = gateway.execute(call("get_project_status", STATUS_ARGS))
    refused = gateway.execute(call("get_project_status", STATUS_ARGS))

    assert allowed.status == "ok"
    assert refused.status == "rate_limited"
    assert refused.attempts == 1
    assert spy.calls == 1


def test_a_refusal_says_when_to_come_back(events: list[TraceEvent]) -> None:
    """A rate limiter always knows the answer -- it is holding the window --
    so a refusal that made the caller guess would be withholding the one fact
    it was in a position to give."""
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, SpyHandler(), max_calls=1, per_seconds=45.0),
        events=events,
    )
    gateway.execute(call("get_project_status", STATUS_ARGS))

    refused = gateway.execute(call("get_project_status", STATUS_ARGS))

    assert refused.retry_after_seconds == 45.0


def test_the_wait_it_reports_is_the_wait_that_actually_works() -> None:
    """The honesty check on the sliding log: waiting exactly as long as the
    refusal asked for is enough, and no longer."""
    clock = Clock()
    spy = SpyHandler()
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, spy, max_calls=1, per_seconds=10.0),
        clock=clock,
    )
    gateway.execute(call("get_project_status", STATUS_ARGS))
    refused = gateway.execute(call("get_project_status", STATUS_ARGS))

    clock.advance(refused.retry_after_seconds or 0.0)
    allowed = gateway.execute(call("get_project_status", STATUS_ARGS))

    assert allowed.status == "ok"
    assert spy.calls == 2


def test_a_rate_limited_read_is_audited_even_though_reads_are_not(
    events: list[TraceEvent],
) -> None:
    """Reads produce no rows, on purpose -- but a limit that fires silently on
    reads hides exactly the runaway pattern it was added to catch."""
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, SpyHandler(), max_calls=1), events=events
    )
    gateway.execute(call("get_project_status", STATUS_ARGS))
    gateway.execute(call("get_project_status", STATUS_ARGS))

    rows = gateway.audit.rows  # type: ignore[union-attr]
    assert [row.status for row in rows] == ["rate_limited"]
    assert rows[0].actor == "pm@example.com"
    assert rows[0].tool_name == "get_project_status"
    assert rows[0].occurred_at == WHEN


def test_a_refusal_for_a_budget_is_not_traced_as_a_failure(
    events: list[TraceEvent],
) -> None:
    """A reviewer counting outages must not be counting the safety layer
    working."""
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, SpyHandler(), max_calls=1), events=events
    )
    gateway.execute(call("get_project_status", STATUS_ARGS))
    gateway.execute(call("get_project_status", STATUS_ARGS))

    kinds = [event.kind for event in events]
    assert "rate_limited" in kinds
    assert "failed" not in kinds


def test_one_actor_spending_its_budget_does_not_spend_another_s() -> None:
    """Per-actor, never global: a shared counter makes one busy user the reason
    another user's read is refused."""
    spy = SpyHandler()
    gateway = gateway_over(limited(GET_PROJECT_STATUS_TOOL, spy, max_calls=1))
    gateway.execute(call("get_project_status", STATUS_ARGS))

    other = gateway.execute(
        call("get_project_status", STATUS_ARGS, actor="lead@example.com")
    )

    assert other.status == "ok"
    assert spy.calls == 2


def test_a_budget_is_spent_per_tool_and_not_across_them() -> None:
    """The other half of the key. A counter shared across tools would let a
    cheap read exhaust the budget for an expensive one."""
    status = SpyHandler()
    risks = SpyHandler()
    gateway = gateway_over(
        limited(GET_PROJECT_STATUS_TOOL, status, max_calls=1),
        limited(LIST_RISKS_TOOL, risks, max_calls=1, scope="project.risk.read"),
    )
    gateway.execute(call("get_project_status", STATUS_ARGS))

    other_tool = gateway.execute(call("list_risks", {"project_id": "atlas"}))

    assert other_tool.status == "ok"
    assert status.calls == 1
    assert risks.calls == 1


def test_a_call_refused_for_a_missing_scope_costs_no_budget() -> None:
    """Quota is consumption of the backend, and a denial consumed nothing. If
    it counted, a model firing calls it is not entitled to could lock the
    actor out of the tools it is."""
    spy = SpyHandler()
    gateway = gateway_over(limited(GET_PROJECT_STATUS_TOOL, spy, max_calls=1))
    gateway.execute(call("get_project_status", STATUS_ARGS, scopes=frozenset()))

    allowed = gateway.execute(call("get_project_status", STATUS_ARGS))

    assert allowed.status == "ok"
    assert spy.calls == 1


def test_a_call_with_invented_arguments_costs_no_budget() -> None:
    spy = SpyHandler()
    gateway = gateway_over(limited(GET_PROJECT_STATUS_TOOL, spy, max_calls=1))
    gateway.execute(call("get_project_status", {"milestone_id": "M2", "force": True}))

    allowed = gateway.execute(call("get_project_status", STATUS_ARGS))

    assert allowed.status == "ok"
    assert spy.calls == 1


def test_stopping_for_a_human_costs_no_budget() -> None:
    """The reason the check and the count are split. A pending approval that
    charged the budget would make the approve-then-resubmit round trip cost
    two units, so a limit of one could never be used at all."""
    spy = SpyHandler()
    gateway = gateway_over(
        limited(
            CREATE_RISK_TOOL,
            spy,
            max_calls=1,
            scope="project.risk.write",
            approval_required=True,
        )
    )
    pending = gateway.execute(new_risk())

    approved = gateway.execute(new_risk(approval="approved"))

    assert pending.status == "approval_required"
    assert approved.status == "ok"
    assert spy.calls == 1


def test_a_human_is_never_asked_to_approve_a_call_the_budget_will_refuse() -> None:
    """Why the check sits above the approval gate: spending an approver's
    attention on a foregone refusal is the expensive mistake."""
    spy = SpyHandler()
    gateway = gateway_over(
        limited(
            CREATE_RISK_TOOL,
            spy,
            max_calls=1,
            scope="project.risk.write",
            approval_required=True,
        )
    )
    gateway.execute(new_risk(approval="approved"))

    second = gateway.execute(new_risk())

    assert second.status == "rate_limited"
    assert spy.calls == 1


def test_a_retried_call_costs_one_unit_and_not_one_per_attempt(
    erp: MockErp,
) -> None:
    """Otherwise an actor's real budget would be a function of how flaky the
    backend was that minute -- which is the one thing a limit described as
    deterministic must not be."""
    handlers = build_default_registry(erp).get("get_project_status_flaky").handler
    gateway = gateway_over(
        limited(
            GET_PROJECT_STATUS_FLAKY_TOOL,
            handlers,
            max_calls=2,
            retry=RetryPolicy(max_attempts=2),
        )
    )

    first = gateway.execute(call("get_project_status_flaky", STATUS_ARGS))
    second = gateway.execute(call("get_project_status_flaky", STATUS_ARGS))

    assert first.attempts == 2
    assert second.status == "ok"


def test_a_permitted_call_is_still_summarized_rather_than_returned_whole(
    gateway: ToolGateway,
) -> None:
    """The limit changed the refusal path and nothing else: a call inside its
    budget still comes back as a sentence plus source identifiers, not as the
    ERP records it read."""
    outcome = gateway.execute(call("list_risks", {"project_id": "atlas"}))

    assert outcome.status == "ok"
    assert outcome.retry_after_seconds is None
    assert outcome.source_ids
    assert "{" not in outcome.summary and "source_id" not in outcome.summary


# --------------------------------------------------------------------------
# The declared budgets
# --------------------------------------------------------------------------


def test_every_registered_tool_carries_a_real_budget(erp: MockErp) -> None:
    """There is no spelling for "unlimited", so a tool added later inherits a
    limit whether or not its author thought about one."""
    registry = build_default_registry(erp)

    for name in registry.names():
        policy = registry.get(name).rate_limit
        assert policy.max_calls >= 1
        assert policy.per_seconds > 0


def test_the_write_is_budgeted_more_tightly_than_the_reads(erp: MockErp) -> None:
    """Not because the write is the primary risk -- approval already covers
    that -- but because the limit is the backstop for what approval cannot
    see: a resubmission loop replaying one approved call."""
    registry = build_default_registry(erp)

    assert (
        registry.get("create_risk").rate_limit.max_calls
        < registry.get("list_risks").rate_limit.max_calls
    )

