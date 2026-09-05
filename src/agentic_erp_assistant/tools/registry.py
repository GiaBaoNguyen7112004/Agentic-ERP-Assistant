"""The registry is the control plane: policy is declared here, never at a call site.

Every fact the gateway needs about a tool -- who may call it, whether a human
must approve it, how long it may take, how many attempts it gets, and what code
runs -- is on one :class:`ToolDefinition`. The gateway therefore contains no
``if tool_name == ...`` anywhere: adding a tool is a row here, and changing a
policy is an edit a reviewer can see in a diff rather than a branch buried in an
execution path.

Where each fact lives, and why it is not stored twice
-----------------------------------------------------

The provider-facing half of a tool -- name, description, argument schema --
already exists as :class:`~agentic_erp_assistant.llm.tools.ToolSpec`, which is
what gets rendered into a function-calling schema. A definition holds a spec
rather than copying out of it, so there is one name and one argument contract.

:attr:`ToolDefinition.side_effect` is derived from ``spec.mutating`` rather
than stored beside it. They are the same fact, and it is the fact the
transition guard already reads before letting a write execute; a second copy
would be the one that is wrong the day someone edits only one of them.

The required arguments are not listed here either. ``spec.arguments`` is a
pydantic model that already declares them, and strict function calling forces
every property to be required, so a second list would add nothing but a chance
to disagree with the first.

:attr:`ToolDefinition.approval_required` *is* stored, because it is not
derivable. Every write needs approval -- enforced below -- but a read may need
it too: an export of a whole ledger is read-only and still worth stopping for a
human. Deriving it would make that policy unexpressible.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.erp.mock import MockErp
from agentic_erp_assistant.llm.tools import (
    CREATE_RISK_TOOL,
    GET_BUDGET_SUMMARY_TOOL,
    GET_PROJECT_STATUS_FLAKY_TOOL,
    GET_PROJECT_STATUS_TOOL,
    GET_SPRINT_PROGRESS_TOOL,
    LIST_RISKS_TOOL,
    ToolSpec,
)
from agentic_erp_assistant.tools.handlers import build_handlers, Handler

__all__ = [
    "build_default_registry",
    "NO_RETRY",
    "RetryPolicy",
    "SideEffect",
    "ToolDefinition",
    "ToolRegistry",
    "UnknownTool",
]


SideEffect = Literal["read", "write"]
"""What a tool does to the world. Derived from the spec; see the module doc."""


class RetryPolicy(BaseModel):
    """How many attempts a tool gets, and how long the waits between them are.

    Per tool rather than global, because the right budget depends on what is
    behind the tool: a read against a flaky endpoint is worth repeating, and a
    write is not worth repeating blindly at all -- a retried create is how one
    approved risk becomes three.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=1, ge=1)
    """Total attempts, not retries. ``1`` means the call is made once and any
    failure is reported as it is -- the default, because retrying is the
    exception and should have to be asked for by name."""

    base_delay_seconds: float = Field(default=0.5, ge=0.0)
    """The first wait, before jitter."""

    max_delay_seconds: float = Field(default=8.0, ge=0.0)
    """The ceiling the doubling is clamped to."""


NO_RETRY = RetryPolicy()
"""One attempt, no waiting. Shared so the common case reads as a decision
rather than as a constructor nobody looked at."""


class UnknownTool(KeyError):
    """The registry was asked for a tool it does not have.

    A ``KeyError`` because that is what it is, and named because the gateway
    has to tell "this tool does not exist" apart from any other lookup failure
    in order to report it as an outcome rather than crash a turn.
    """


@dataclass(frozen=True)
class ToolDefinition:
    """One tool, and every policy that applies to running it.

    A frozen dataclass rather than a pydantic model because it holds a callable
    handler, and the value of validating a function object is nil next to the
    cost of teaching a model to accept one.
    """

    spec: ToolSpec
    """The provider-facing declaration: name, description, argument model, and
    the ``mutating`` flag everything below reads."""

    required_scope: str
    """The entitlement an actor must hold to call this at all.

    Checked against ``ToolRequest.scopes`` by the gateway, before approval is
    even considered: approval decides whether a permitted call should happen
    now, and it cannot grant an entitlement its holder never had.
    """

    approval_required: bool
    """Whether a human must decide before this runs. Not derivable; see the
    module docstring."""

    timeout_seconds: float
    """How long one attempt may take. Declared per tool because the budget is a
    property of what is behind the tool, not of the runtime."""

    retry: RetryPolicy
    """The attempt budget. See :class:`RetryPolicy`."""

    handler: Handler
    """The code that runs, already bound to its data store.

    A callable, not a name to look up later. A name is a lookup that fails at
    call time, in production, with a registry that looked complete; a callable
    fails at import, where a missing handler is a syntax-level mistake.
    """

    @property
    def name(self) -> str:
        """The tool's name, from the spec. Never a second copy."""
        return self.spec.name

    @property
    def side_effect(self) -> SideEffect:
        """``"write"`` if the tool changes ERP data, else ``"read"``.

        A view over ``spec.mutating``, which is what the runtime's transition
        guard reads. One stored fact, two vocabularies.
        """
        return "write" if self.spec.mutating else "read"

    def __post_init__(self) -> None:
        if not self.required_scope.strip():
            raise ValueError(
                f"tool {self.name!r}: required_scope must name an entitlement; "
                f"a blank one would be a permission check that always passes"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"tool {self.name!r}: timeout_seconds must be positive, got "
                f"{self.timeout_seconds}"
            )
        if self.side_effect == "write" and not self.approval_required:
            raise ValueError(
                f"tool {self.name!r}: a write that runs unattended cannot be "
                f"declared here -- approval_required must be True when the "
                f"tool mutates ERP data"
            )


class ToolRegistry:
    """Every tool the runtime can execute, by name.

    Not the same set as
    :data:`~agentic_erp_assistant.llm.tools.DEFAULT_TOOLS`, which is what the
    model is *offered*. A tool can be executable without being advertised --
    ``get_project_status_flaky`` is -- and the two lists answer different
    questions, so they are deliberately not one list.
    """

    def __init__(self, definitions: tuple[ToolDefinition, ...]) -> None:
        by_name: dict[str, ToolDefinition] = {}
        for definition in definitions:
            if definition.name in by_name:
                raise ValueError(
                    f"tool {definition.name!r} is declared twice; one of the "
                    f"two policies would silently win"
                )
            by_name[definition.name] = definition
        self._by_name = by_name

    def get(self, name: str) -> ToolDefinition:
        """The definition for ``name``.

        Raises:
            UnknownTool: No tool by that name. The model naming a tool that
                does not exist is a routed, recorded failure at the gateway --
                not something this class should paper over with ``None``.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise UnknownTool(name) from None

    def names(self) -> tuple[str, ...]:
        """Every registered name, in declaration order."""
        return tuple(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)


def build_default_registry(erp: MockErp | None = None) -> ToolRegistry:
    """The six tools, with their policy, bound to one ERP store.

    ``erp`` is injectable so a test can hand in its own data and so two tests
    cannot see each other's writes. Defaults to the repo fixture.

    The scopes are namespaced by what they grant rather than by who holds them
    (``project.risk.write``, not ``pm``), so an entitlement can be read without
    knowing the role table -- and so adding a role never means editing a tool.
    """
    store = erp if erp is not None else MockErp.load()
    handlers = build_handlers(store)

    def read(
        spec: ToolSpec, scope: str, *, retry: RetryPolicy = NO_RETRY
    ) -> ToolDefinition:
        return ToolDefinition(
            spec=spec,
            required_scope=scope,
            approval_required=False,
            timeout_seconds=5.0,
            retry=retry,
            handler=handlers[spec.name],
        )

    return ToolRegistry(
        (
            read(GET_PROJECT_STATUS_TOOL, "project.status.read"),
            # The only tool with a budget above one attempt. Its backend fails
            # intermittently and a second try genuinely fixes it; every other
            # read gets one attempt, so a retry in a trace is always a
            # deliberate policy and never a default nobody chose.
            read(
                GET_PROJECT_STATUS_FLAKY_TOOL,
                "project.status.read",
                retry=RetryPolicy(max_attempts=2),
            ),
            read(GET_SPRINT_PROGRESS_TOOL, "project.sprint.read"),
            read(GET_BUDGET_SUMMARY_TOOL, "project.budget.read"),
            read(LIST_RISKS_TOOL, "project.risk.read"),
            ToolDefinition(
                spec=CREATE_RISK_TOOL,
                required_scope="project.risk.write",
                approval_required=True,
                timeout_seconds=5.0,
                # One attempt, on purpose. A retried write is how one approved
                # risk becomes three, and the approval was for one.
                retry=NO_RETRY,
                handler=handlers["create_risk"],
            ),
        )
    )
