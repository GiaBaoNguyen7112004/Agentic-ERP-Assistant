"""A port is only a boundary if nothing in the runtime reaches past it."""

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentic_erp_assistant.runtime.ports import (
    DocumentRetrieverPort,
    ToolGatewayPort,
    ToolOutcome,
)
from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.evidence import EvidenceSnippet


# --------------------------------------------------------------------------
# Fakes. Note what is missing: neither inherits from the protocol, and neither
# imports anything from rag/ or tools/. That is the whole argument for
# structural typing here.
# --------------------------------------------------------------------------


class FakeRetriever:
    """Ten lines, no index, no I/O -- what a node gets to test against."""

    def __init__(self, snippets: Sequence[EvidenceSnippet] = ()) -> None:
        self.snippets = tuple(snippets)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int) -> Sequence[EvidenceSnippet]:
        self.calls.append((query, limit))
        return self.snippets[:limit]


class FakeGateway:
    def __init__(self, outcome: ToolOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, Mapping[str, Any], str, str]] = []

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        actor: str,
        approval: ApprovalDecision,
    ) -> ToolOutcome:
        self.calls.append((tool_name, arguments, actor, approval))
        return self.outcome


def snippet(locator: str) -> EvidenceSnippet:
    return EvidenceSnippet(
        source_id="sprint-12-report.md", locator=locator, text="M2 slipped."
    )


def succeeded(tool_name: str) -> ToolOutcome:
    """A minimal successful outcome. Its own rules are tested with the type,
    in tests/state/test_tool_outcome.py."""
    return ToolOutcome(
        tool_name=tool_name, status="ok", summary="M2 is on track.",
        source_ids=("milestone:M2",),
    )


# --------------------------------------------------------------------------
# Conformance is structural: an implementation never imports the port
# --------------------------------------------------------------------------


def test_a_retriever_conforms_without_inheriting() -> None:
    """rag/ implements this without importing runtime/, which is what keeps the
    dependency direction pointing inward."""
    retriever = FakeRetriever()

    assert isinstance(retriever, DocumentRetrieverPort)
    assert DocumentRetrieverPort not in FakeRetriever.__mro__


def test_a_gateway_conforms_without_inheriting() -> None:
    gateway = FakeGateway(succeeded("get_project_status"))

    assert isinstance(gateway, ToolGatewayPort)
    assert ToolGatewayPort not in FakeGateway.__mro__


def test_something_without_the_method_does_not_conform() -> None:
    class NotARetriever:
        def query(self, text: str) -> None: ...

    assert not isinstance(NotARetriever(), DocumentRetrieverPort)


def test_the_runtime_check_sees_methods_only_and_not_signatures() -> None:
    """Worth pinning so nobody reads an isinstance pass as a contract check.
    The signature is enforced by the type checker and by the call sites here,
    not at runtime."""

    class WrongSignature:
        def search(self) -> None: ...

    assert isinstance(WrongSignature(), DocumentRetrieverPort)


# --------------------------------------------------------------------------
# The call shapes the runtime relies on
# --------------------------------------------------------------------------


def test_the_result_type_is_the_one_the_state_layer_defines() -> None:
    """Re-exported, not redefined: the tool layer builds these and must not
    have to import runtime/ to name what it returns."""
    from agentic_erp_assistant.state.tool_outcome import ToolOutcome as Defined

    assert ToolOutcome is Defined


def test_a_caller_typed_against_the_port_works_with_the_fake() -> None:
    retriever: DocumentRetrieverPort = FakeRetriever([snippet("3.1"), snippet("3.2")])

    results = retriever.search("How is M2 tracking?", limit=1)

    assert [result.tag for result in results] == ["[sprint-12-report.md#3.1]"]


def test_limit_is_keyword_only() -> None:
    """No default and no positional form: how much evidence to pull is a
    context-budget decision the runtime records, not one a backend hides."""
    retriever = FakeRetriever([snippet("3.1")])

    with pytest.raises(TypeError):
        retriever.search("q", 1)  # type: ignore[misc]


def test_finding_nothing_is_an_answer_and_not_an_exception() -> None:
    """'no evidence' is a routed outcome; an exception would make the trace
    record a fault instead."""
    assert FakeRetriever().search("q", limit=5) == ()


def test_the_gateway_is_told_who_asked_and_what_the_approver_said() -> None:
    """The last boundary in front of a write does not get to assume someone
    upstream checked."""
    gateway = FakeGateway(succeeded("close_milestone"))

    gateway.execute(
        "close_milestone", {"milestone_id": "M2"}, actor="bao", approval="approved"
    )

    assert gateway.calls == [
        ("close_milestone", {"milestone_id": "M2"}, "bao", "approved")
    ]


@pytest.mark.parametrize("missing", ["actor", "approval"])
def test_neither_audit_argument_can_be_omitted(missing: str) -> None:
    gateway = FakeGateway(succeeded("close_milestone"))
    kwargs: dict[str, Any] = {"actor": "bao", "approval": "approved"}
    del kwargs[missing]

    with pytest.raises(TypeError):
        gateway.execute("close_milestone", {}, **kwargs)


# --------------------------------------------------------------------------
# The acceptance criterion, checked against the source rather than by review
# --------------------------------------------------------------------------


RUNTIME_PACKAGE = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_erp_assistant" / "runtime"
)

REPLACEABLE_PARTS = (
    "agentic_erp_assistant.rag",
    "agentic_erp_assistant.tools",
    "agentic_erp_assistant.erp",
    "agentic_erp_assistant.web",
    "agentic_erp_assistant.llm.adapters",
)
"""The packages a port exists to stand in for, plus the layer the runtime must
never point at. Naming any of them from inside runtime/ means a node reached
past its boundary."""


def imported_modules(source: Path) -> set[str]:
    """Every module named by an import statement in one file."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_nothing_in_the_runtime_imports_a_replaceable_part() -> None:
    """The unit's acceptance criterion, as a check instead of a habit: no node
    may import a concrete retriever or gateway, only the protocols. Written
    against the whole package so it keeps holding as nodes are added."""
    offenders: dict[str, set[str]] = {}

    for source in sorted(RUNTIME_PACKAGE.rglob("*.py")):
        reached_past = {
            module
            for module in imported_modules(source)
            for part in REPLACEABLE_PARTS
            if module == part or module.startswith(part + ".")
        }
        if reached_past:
            offenders[source.name] = reached_past

    assert offenders == {}


def test_the_guard_would_actually_catch_an_offender() -> None:
    """A check that cannot fail proves nothing about the code it guards."""
    offending = "from agentic_erp_assistant.rag.hybrid import HybridRetriever\n"
    tree = ast.parse(offending)
    module = next(
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert any(
        module == part or module.startswith(part + ".") for part in REPLACEABLE_PARTS
    )
