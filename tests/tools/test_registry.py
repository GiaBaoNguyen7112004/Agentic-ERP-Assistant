"""If the policy is not readable off the registry, it is not a control plane."""

from pathlib import Path

import pytest

from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, MockErp
from agentic_erp_assistant.llm.tools import (
    CREATE_RISK_TOOL,
    DEFAULT_TOOLS,
    GET_PROJECT_STATUS_TOOL,
    LIST_RISKS_TOOL,
)
from agentic_erp_assistant.tools.handlers import HandlerResult
from agentic_erp_assistant.tools.registry import (
    build_default_registry,
    NO_RETRY,
    RetryPolicy,
    ToolDefinition,
    ToolRegistry,
    UnknownTool,
)

EXPECTED = (
    "get_project_status",
    "get_project_status_flaky",
    "get_sprint_progress",
    "get_budget_summary",
    "list_risks",
    "create_risk",
)


@pytest.fixture
def registry(erp_file) -> ToolRegistry:
    """Over a writable copy of the fixture, because the handler-call tests
    below execute real writes now."""
    return build_default_registry(MockErp.load(erp_file))


def anything(_: object) -> HandlerResult:
    return HandlerResult(summary="done", source_ids=("x",))


# --------------------------------------------------------------------------
# What is registered
# --------------------------------------------------------------------------


def test_the_default_registry_holds_all_six_tools(registry: ToolRegistry) -> None:
    assert registry.names() == EXPECTED


def test_an_unknown_tool_is_a_named_failure(registry: ToolRegistry) -> None:
    """The gateway has to tell 'no such tool' apart from any other lookup
    failure to report it as an outcome instead of crashing a turn."""
    with pytest.raises(UnknownTool, match="close_milestone"):
        registry.get("close_milestone")


def test_a_tool_declared_twice_is_rejected() -> None:
    """One of the two policies would silently win, and nobody would know which."""
    definition = ToolDefinition(
        spec=GET_PROJECT_STATUS_TOOL,
        required_scope="project.status.read",
        approval_required=False,
        timeout_seconds=5.0,
        retry=NO_RETRY,
        handler=anything,
    )

    with pytest.raises(ValueError, match="declared twice"):
        ToolRegistry((definition, definition))


def test_the_registry_and_the_model_s_menu_are_not_the_same_list(
    registry: ToolRegistry,
) -> None:
    """A tool can be executable without being advertised. The flaky twin is:
    offering the model two tools that do the same thing means it sometimes
    picks the broken one and the trace records a retry nobody asked for."""
    offered = {spec.name for spec in DEFAULT_TOOLS}

    assert "get_project_status_flaky" in registry
    assert "get_project_status_flaky" not in offered
    assert offered <= set(registry.names())


# --------------------------------------------------------------------------
# The acceptance criteria, read straight off the definitions
# --------------------------------------------------------------------------


def test_create_risk_is_a_write_that_needs_approval(registry: ToolRegistry) -> None:
    tool = registry.get("create_risk")

    assert tool.side_effect == "write" and tool.approval_required is True


def test_create_risk_is_the_only_tool_requiring_approval(
    registry: ToolRegistry,
) -> None:
    needing = [
        name for name in registry.names() if registry.get(name).approval_required
    ]

    assert needing == ["create_risk"]


def test_every_read_tool_runs_automatically_once_permission_passes(
    registry: ToolRegistry,
) -> None:
    reads = [
        registry.get(name)
        for name in registry.names()
        if registry.get(name).side_effect == "read"
    ]

    assert len(reads) == 5
    assert all(tool.approval_required is False for tool in reads)


def test_create_risk_requires_the_write_scope(registry: ToolRegistry) -> None:
    assert registry.get("create_risk").required_scope == "project.risk.write"


def test_every_tool_names_an_entitlement(registry: ToolRegistry) -> None:
    assert all(registry.get(name).required_scope for name in registry.names())


def test_a_blank_scope_is_rejected() -> None:
    """It would be a permission check that always passes."""
    with pytest.raises(ValueError, match="required_scope"):
        ToolDefinition(
            spec=GET_PROJECT_STATUS_TOOL,
            required_scope="  ",
            approval_required=False,
            timeout_seconds=5.0,
            retry=NO_RETRY,
            handler=anything,
        )


# --------------------------------------------------------------------------
# side_effect is a view, not a second copy
# --------------------------------------------------------------------------


def test_the_side_effect_comes_from_the_spec_and_not_from_a_second_field(
    registry: ToolRegistry,
) -> None:
    """The same fact the runtime's transition guard reads before letting a
    write execute. A stored copy would be the one that is wrong the day
    somebody edits only one of them."""
    tool = registry.get("create_risk")

    assert tool.spec.mutating is True
    assert "side_effect" not in vars(tool)


def test_the_name_is_not_stored_twice_either(registry: ToolRegistry) -> None:
    tool = registry.get("get_project_status")

    assert tool.name == tool.spec.name
    assert "name" not in vars(tool)


def test_a_write_that_runs_unattended_cannot_be_declared() -> None:
    """The invariant that makes approval_required safe to store: it may be
    True for a read (an export worth stopping for a human), but never False for
    a write."""
    with pytest.raises(ValueError, match="approval_required"):
        ToolDefinition(
            spec=CREATE_RISK_TOOL,
            required_scope="project.risk.write",
            approval_required=False,
            timeout_seconds=5.0,
            retry=NO_RETRY,
            handler=anything,
        )


def test_a_read_may_still_be_declared_as_needing_approval() -> None:
    """Which is why approval_required is stored rather than derived from
    side_effect."""
    escalated = ToolDefinition(
        spec=GET_PROJECT_STATUS_TOOL,
        required_scope="project.status.read",
        approval_required=True,
        timeout_seconds=5.0,
        retry=NO_RETRY,
        handler=anything,
    )

    assert escalated.side_effect == "read" and escalated.approval_required is True


# --------------------------------------------------------------------------
# Retry budgets
# --------------------------------------------------------------------------


def test_only_the_flaky_read_gets_more_than_one_attempt(
    registry: ToolRegistry,
) -> None:
    """So a retry showing up in a trace is always a deliberate policy and never
    a default nobody chose."""
    budgets = {
        name: registry.get(name).retry.max_attempts for name in registry.names()
    }

    assert budgets["get_project_status_flaky"] == 2
    assert all(
        attempts == 1
        for name, attempts in budgets.items()
        if name != "get_project_status_flaky"
    )


def test_the_write_is_never_retried(registry: ToolRegistry) -> None:
    """A retried create is how one approved risk becomes three, and the
    approval was for one."""
    assert registry.get("create_risk").retry.max_attempts == 1


def test_the_default_policy_is_a_single_attempt() -> None:
    """Retrying is the exception and has to be asked for by name."""
    assert NO_RETRY.max_attempts == 1


def test_a_budget_of_zero_attempts_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_a_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ToolDefinition(
            spec=GET_PROJECT_STATUS_TOOL,
            required_scope="project.status.read",
            approval_required=False,
            timeout_seconds=0.0,
            retry=NO_RETRY,
            handler=anything,
        )


# --------------------------------------------------------------------------
# Handlers are bound, and bound to the store they were given
# --------------------------------------------------------------------------


def test_each_definition_carries_the_code_that_runs(registry: ToolRegistry) -> None:
    """A callable, not a name to look up later: a name fails at call time in
    production, a missing callable fails at import."""
    assert callable(registry.get("list_risks").handler)


def a_writable_copy(target: Path) -> MockErp:
    """A store over a fresh copy of the fixture, so a write lands nowhere
    shared."""
    target.write_text(
        DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return MockErp.load(target)


def test_two_registries_do_not_share_a_store(tmp_path) -> None:
    """So one test's write cannot be seen by another's read.

    Two files, not one: with a write that really persists, two stores over one
    file would pass this on stale in-memory lists -- the isolation being
    asserted is between stores, and it has to survive a reload."""
    mine = build_default_registry(a_writable_copy(tmp_path / "mine.json"))
    yours = build_default_registry(a_writable_copy(tmp_path / "yours.json"))

    mine.get("create_risk").handler(
        CREATE_RISK_TOOL.validate_arguments(
            {"project_id": "atlas", "title": "mine", "severity": "low"}
        )
    )
    listed = yours.get("list_risks").handler(
        LIST_RISKS_TOOL.validate_arguments({"project_id": "atlas"})
    )

    assert "mine" not in listed.summary
