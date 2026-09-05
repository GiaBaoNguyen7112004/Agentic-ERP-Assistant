"""The order is the safety, so the tests assert on what did not happen."""

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from agentic_erp_assistant.erp.mock import MockErp
from agentic_erp_assistant.llm.tools import CREATE_RISK_TOOL, GET_PROJECT_STATUS_TOOL
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.tool_request import ToolRequest
from agentic_erp_assistant.tools.gateway import GATEWAY_NODE, ToolGateway
from agentic_erp_assistant.tools.handlers import HandlerResult
from agentic_erp_assistant.tools.models import ToolError, TransientToolError
from agentic_erp_assistant.tools.registry import (
    build_default_registry,
    NO_RETRY,
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
