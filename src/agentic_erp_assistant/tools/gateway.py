"""One ordered path from a request to an outcome, and the order is the safety.

:meth:`ToolGateway.execute` is seven steps, in this order and no other:

1. find the tool in the registry
2. validate the arguments against its declaration
3. check the actor holds the tool's scope
4. stop for approval, if the tool needs one
5. write the audit row for a gated call
6. run the handler, inside its retry budget and its timeout
7. emit a trace event

Steps 3 and 4 come before step 6, and that is the whole point of writing this
as one function. A gateway that checked permission after execution, or asked
for approval after calling the handler, would have already changed ERP data by
the time it refused -- and every test of the refusal would still pass, because
the refusal is returned either way. The order is the invariant, so it is
written once, in one place a reader can follow top to bottom, and the tests
assert on what did *not* happen as much as on what came back.

No branching on tool names
--------------------------

There is no ``if tool_name == "create_risk"`` here and there never should be.
Every policy this function applies is read off the
:class:`~agentic_erp_assistant.tools.registry.ToolDefinition`, so adding a tool
is a row in the registry and changing a rule is an edit a reviewer sees. A
special case written here would be a policy that exists in exactly one code
path and in no document.

Nothing escapes
---------------

Every failure comes back as a :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`,
including a handler that raised and a tool that does not exist. The turn has to
record that the attempt happened; an exception unwinding into the graph would
leave that record unwritten, and the one call nobody can account for would be a
call that may or may not have changed something.
"""

import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ValidationError

from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.state.events import EVENT_DETAIL_MAX_CHARS, TraceEvent
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest
from agentic_erp_assistant.tools.audit import AuditSink, InMemoryAuditLog
from agentic_erp_assistant.tools.models import (
    ARGUMENTS_SUMMARY_MAX_CHARS,
    AuditRow,
    ToolError,
    ToolStatus,
    TransientToolError,
)
from agentic_erp_assistant.tools.registry import ToolDefinition, ToolRegistry, UnknownTool

__all__ = ["GATEWAY_NODE", "ToolGateway"]

logger = logging.getLogger(__name__)

GATEWAY_NODE = "tool_gateway"
"""What this layer calls itself in a trace event.

A name, not a class reference, so the trace stays readable after a refactor
renames the class.
"""

_VALUE_MAX_CHARS = 40
"""How much of one argument value reaches the audit line before it is elided."""


def _summarize(request: ToolRequest) -> str:
    """Render a call as one line an approver and an auditor can both read.

    Values are included, not just keys: "create_risk(project_id, title,
    severity)" tells an auditor nothing about what changed, which is the only
    thing they came to find out. The protection is the cap, both per value and
    on the whole line -- a tool that takes a credential as an argument is the
    thing to fix, and no renderer can make that safe.
    """
    parts = []
    for key, value in request.arguments.items():
        rendered = str(value)
        if len(rendered) > _VALUE_MAX_CHARS:
            rendered = rendered[: _VALUE_MAX_CHARS - 1] + "…"
        parts.append(f"{key}={rendered}")

    line = f"{request.tool_name}({', '.join(parts)})"
    if len(line) > ARGUMENTS_SUMMARY_MAX_CHARS:
        line = line[: ARGUMENTS_SUMMARY_MAX_CHARS - 1] + "…"
    return line


@dataclass
class ToolGateway:
    """The one place a tool call becomes a tool execution.

    Satisfies :class:`~agentic_erp_assistant.runtime.ports.ToolGatewayPort`
    structurally, without importing it -- see that module for why.

    Every collaborator is injected and defaulted, which is not politeness: the
    clock, the sleep and the jitter are what make "it retried twice and waited"
    something a test can assert on instead of live through.
    """

    registry: ToolRegistry
    """What may be called, and under what policy."""

    audit: AuditSink = field(default_factory=InMemoryAuditLog)
    """Where the record of a gated call goes."""

    on_event: Callable[[TraceEvent], object] | None = None
    """Called with each trace event as it happens.

    A hook rather than a trace store, for the reason
    :func:`~agentic_erp_assistant.llm.retry.retry_with_backoff` takes one: the
    event log lives on
    :class:`~agentic_erp_assistant.state.agent_state.AgentState`, and a gateway
    that wrote to it directly would need to import the runtime and would be
    holding a state object it has no business editing.
    """

    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    """The clock, injectable so an audit row is assertable."""

    sleep: Callable[[float], object] = time.sleep
    """How to wait between attempts. A test passes a no-op."""

    jitter: Callable[[], float] = random.random
    """The backoff jitter factor. A test pins it."""

    # -- the policy order --------------------------------------------------

    def execute(self, request: ToolRequest) -> ToolOutcome:
        """Run one call through every check, in order, and report what happened.

        Args:
            request: The call, with the actor, their scopes and the approval
                decision that the checks below consult.

        Returns:
            A :class:`ToolOutcome`. Never raises for a failed call -- see the
            module docstring.
        """
        # 1. Find the tool. A model naming one that does not exist is a routed,
        #    recorded failure, not a crash: the registry is the authority, and
        #    the turn still has to say what it tried.
        try:
            definition = self.registry.get(request.tool_name)
        except UnknownTool:
            return self._refused(
                request,
                status="failed",
                error=f"no tool named {request.tool_name!r} is registered",
            )

        # 2. Validate the arguments, before anything else looks at the call.
        #    ToolSpec's model forbids extras, so an argument the model invented
        #    raises here and no handler is ever reached.
        try:
            arguments = definition.spec.validate_arguments(request.arguments)
        except ValidationError as error:
            return self._refused(
                request,
                definition=definition,
                status="invalid_arguments",
                error=self._first_problem(error),
            )

        # 3. Permission. Before approval, deliberately: approval decides
        #    whether a permitted call should happen now, and it can never grant
        #    an entitlement its holder never had. One attempt, no retry -- a
        #    missing scope will still be missing on the second try.
        if definition.required_scope not in request.scopes:
            return self._refused(
                request,
                definition=definition,
                status="denied",
                error=(
                    f"actor {request.actor!r} does not hold "
                    f"{definition.required_scope!r}"
                ),
            )

        # 4. Approval, for the tools that need one, before the handler exists
        #    in this function's future at all.
        if definition.approval_required and request.approval != "approved":
            asked = request.approval == "denied"
            return self._refused(
                request,
                definition=definition,
                status="denied" if asked else "approval_required",
                error=(
                    "a human denied this call"
                    if asked
                    else f"{definition.side_effect} calls to "
                    f"{definition.name!r} need a recorded approval first"
                ),
            )

        # 5, 6, 7. Run it, record it, trace it.
        outcome = self._run(definition, arguments, request)
        self._write_audit_row(request, definition, outcome)
        self._emit(
            "tool_called" if outcome.status == "ok" else "failed",
            f"{definition.name} -> {outcome.status} in {outcome.attempts} "
            f"attempt{'s' if outcome.attempts != 1 else ''}",
        )
        return outcome

    # -- execution ---------------------------------------------------------

    def _run(
        self,
        definition: ToolDefinition,
        arguments: BaseModel,
        request: ToolRequest,
    ) -> ToolOutcome:
        """Call the handler inside its budget, and turn any raise into a status."""
        attempts = 0

        def attempt() -> Any:
            nonlocal attempts
            attempts += 1
            # The timeout bounds how long this gateway waits, not how long the
            # handler runs -- a thread cannot be cancelled. A real backend
            # client would carry the deadline itself; this stops one slow tool
            # from holding a turn open indefinitely, and says so in the outcome.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(definition.handler, arguments)
                try:
                    return future.result(timeout=definition.timeout_seconds)
                except FutureTimeout:
                    raise TransientToolError(
                        f"{definition.name} did not answer within "
                        f"{definition.timeout_seconds:g}s"
                    ) from None

        def on_retry(attempt_number: int, delay: float, error: Exception) -> None:
            self._emit(
                "retry_scheduled",
                f"{definition.name} attempt {attempt_number} failed "
                f"({type(error).__name__}); waiting {delay:.2f}s",
            )

        try:
            result = retry_with_backoff(
                attempt,
                max_attempts=definition.retry.max_attempts,
                base_delay_seconds=definition.retry.base_delay_seconds,
                max_delay_seconds=definition.retry.max_delay_seconds,
                sleep=self.sleep,
                jitter=self.jitter,
                on_retry=on_retry,
                retry_on=TransientToolError,
            )
        except TransientToolError as error:
            return ToolOutcome(
                tool_name=definition.name,
                status="transient_failure",
                error=str(error),
                attempts=attempts,
            )
        except ToolError as error:
            # Permanent by construction: the handler said this will fail the
            # same way next time, so the budget is not spent proving it.
            return ToolOutcome(
                tool_name=definition.name,
                status="failed",
                error=str(error),
                attempts=attempts,
            )

        return ToolOutcome(
            tool_name=definition.name,
            status="ok",
            summary=result.summary,
            source_ids=result.source_ids,
            attempts=attempts,
        )

    # -- recording ---------------------------------------------------------

    def _write_audit_row(
        self,
        request: ToolRequest,
        definition: ToolDefinition,
        outcome: ToolOutcome,
    ) -> None:
        """Record a gated call, whatever became of it.

        Only tools that require approval produce rows. An audit trail with one
        row per read is one where the writes are buried, and the writes are the
        reason it exists.

        The row is written after execution because it carries the real status,
        and a row saying "approved, then unknown" is not evidence. Nothing can
        execute and skip this: every path out of :meth:`_run` returns an
        outcome rather than raising.
        """
        if not definition.approval_required:
            return

        self.audit.record(
            AuditRow(
                occurred_at=self.now(),
                actor=request.actor,
                tool_name=definition.name,
                arguments_summary=_summarize(request),
                approval=request.approval,
                status=outcome.status,
                source_ids=outcome.source_ids,
            )
        )
        self._emit(
            "approval_recorded",
            f"{definition.name}: approval {request.approval}, "
            f"outcome {outcome.status}",
        )

    def _refused(
        self,
        request: ToolRequest,
        *,
        status: ToolStatus,
        error: str,
        definition: ToolDefinition | None = None,
    ) -> ToolOutcome:
        """Build the outcome for a call that never reached a handler.

        One attempt, always: nothing was tried more than once because nothing
        was tried at all, and an attempts count above one here would read as a
        retry that never happened.
        """
        outcome = ToolOutcome(
            tool_name=request.tool_name, status=status, error=error, attempts=1
        )
        if definition is not None:
            self._write_audit_row(request, definition, outcome)
        self._emit(
            "approval_requested" if status == "approval_required" else "failed",
            f"{request.tool_name} refused before execution: {status}",
        )
        return outcome

    def _emit(self, kind: str, detail: str) -> None:
        """Hand one trace event to whoever is listening, if anyone is."""
        if self.on_event is None:
            return
        self.on_event(
            TraceEvent(
                node=GATEWAY_NODE,
                kind=kind,  # type: ignore[arg-type]
                detail=detail[:EVENT_DETAIL_MAX_CHARS],
            )
        )

    @staticmethod
    def _first_problem(error: ValidationError) -> str:
        """One line naming the first thing wrong with the arguments.

        The whole pydantic report is several lines of schema paths, and it
        would land in an outcome that a model reads back. One field and one
        reason is what a caller can act on.
        """
        problem = error.errors()[0]
        where = ".".join(str(part) for part in problem["loc"]) or "arguments"
        return f"{where}: {problem['msg']}"
