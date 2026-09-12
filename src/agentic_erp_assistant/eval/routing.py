"""Sending every (prompt, case) pair through the real planner, and counting
what it chose.

Every number this produces is computed from a run, the same rule
``eval/retrieval.py`` follows: a provider error becomes one row with
``error`` set, never an aborted report. What is measured is the planner's
*first* decision only -- one model call, nothing executed -- because the
question this harness exists to answer is which contract routes correctly,
not what a whole turn would do.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_erp_assistant.eval.routing_cases import RoutingCase
from agentic_erp_assistant.eval.routing_prompts import RouterPrompt
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.tools import PLANNING_TOOLS
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState

if TYPE_CHECKING:
    from agentic_erp_assistant.composition.users import UserDirectory

__all__ = ["ComparisonReport", "PromptSummary", "RouterResult", "run_comparison"]

logger = logging.getLogger(__name__)

_HALLUCINATION_MARKER = "which was not offered"
"""Substring of Planner._unreadable's message when the model named a tool it
was not offered. Detected, not re-derived: the planner already knows this
happened, and asking it twice would be a second place the check could drift
from the one it mirrors."""


@dataclass(frozen=True)
class RouterResult:
    """One (prompt, case, repeat) decision, self-contained and typed."""

    prompt: str
    case_id: str
    repeat: int

    chosen_route: str | None
    """``None`` only when the row failed before a decision came back."""

    chosen_tool: str | None

    accepted_first_routes: tuple[tuple[str, str | None], ...]
    """A copy of the case's accepted set, so a reader of this one row does
    not have to look the case back up to know what would have counted."""

    matched: bool
    """Whether ``(chosen_route, chosen_tool)`` is in ``accepted_first_routes``."""

    hallucinated_tool: bool
    """Whether the model named a tool that was never offered."""

    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    error: str | None = None
    """Why the row has no decision, when it does not."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "case_id": self.case_id,
            "repeat": self.repeat,
            "chosen_route": self.chosen_route,
            "chosen_tool": self.chosen_tool,
            "accepted_first_routes": [list(pair) for pair in self.accepted_first_routes],
            "matched": self.matched,
            "hallucinated_tool": self.hallucinated_tool,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass(frozen=True)
class PromptSummary:
    """One prompt's numbers, across every case and repeat."""

    prompt: str
    total: int
    matches: int
    hallucinations: int
    cost_usd: float
    contract_length: int

    @property
    def match_rate(self) -> float:
        return self.matches / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "total": self.total,
            "matches": self.matches,
            "match_rate": round(self.match_rate, 4),
            "hallucinations": self.hallucinations,
            "cost_usd": round(self.cost_usd, 6),
            "contract_length": self.contract_length,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """The run, and the numbers derived from it -- nothing here can be set,
    so a report cannot disagree with the rows it came from."""

    results: tuple[RouterResult, ...]
    prompts: tuple[RouterPrompt, ...]

    def summary(self) -> tuple[PromptSummary, ...]:
        by_name = {prompt.name: prompt for prompt in self.prompts}
        summaries = []
        for name in dict.fromkeys(result.prompt for result in self.results):
            rows = [result for result in self.results if result.prompt == name]
            summaries.append(
                PromptSummary(
                    prompt=name,
                    total=len(rows),
                    matches=sum(1 for row in rows if row.matched),
                    hallucinations=sum(1 for row in rows if row.hallucinated_tool),
                    cost_usd=sum(row.cost_usd or 0.0 for row in rows),
                    contract_length=len(by_name[name].contract) if name in by_name else 0,
                )
            )
        return tuple(summaries)

    def route_distribution(self, prompt: str, case_id: str) -> dict[str, int]:
        """How often each ``route`` or ``route:tool`` was chosen, for one
        (prompt, case) pair across its repeats."""
        counts: dict[str, int] = {}
        for result in self.results:
            if result.prompt != prompt or result.case_id != case_id:
                continue
            key = (
                f"{result.chosen_route}:{result.chosen_tool}"
                if result.chosen_tool
                else str(result.chosen_route)
            )
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": [s.to_dict() for s in self.summary()],
            "results": [r.to_dict() for r in self.results],
        }


def _build_state(case: RoutingCase, user: Any) -> AgentState:
    return AgentState(
        request=case.request,
        actor=case.actor,
        project_code=user.project_code,
        trace_id=f"routing-eval-{case.case_id}",
        scopes=user.scopes,
    )


def run_comparison(
    prompts: Sequence[RouterPrompt],
    cases: Sequence[RoutingCase],
    gateway_factory: Callable[[str], LLMGateway],
    users: "UserDirectory",
    *,
    repeats: int = 5,
    pause_seconds: float = 0.0,
    sleep: Callable[[float], object] = time.sleep,
) -> ComparisonReport:
    """Run every (prompt, case) pair ``repeats`` times through the real planner.

    One :class:`~agentic_erp_assistant.llm.gateway.LLMGateway` per prompt (so
    each contract's telemetry, cost and budget are its own), one
    :class:`~agentic_erp_assistant.reasoning.planner.Planner` over it, offering
    the same :data:`~agentic_erp_assistant.llm.tools.PLANNING_TOOLS` production
    offers. Nothing executes: ``Planner.plan`` only, so a two-of-six write case
    costs one model call, never a real change to ERP data.

    Args:
        prompts: The contracts to compare.
        cases: The fixed set, identical for every prompt.
        gateway_factory: Builds one gateway from a contract string. The
            caller's to construct (a real ``OpenAIChatClient``, or a scripted
            fake in a test) -- this module never names a provider.
        users: Resolves each case's actor to a project and scopes, the same
            directory the web layer reads (``composition.users``).
        repeats: How many times each pair is run.
        pause_seconds: Waited before every call after the first. ``0.0`` --
            the default, and what every offline test uses -- runs back to
            back. A live comparison against an account with a tokens-per-
            minute limit sets this to stay under it: 90 calls at ~2.7k
            tokens each is well past a 30k TPM budget run back to back,
            observed live the first time this ran (ADR 0020's report notes
            the pause it was produced with).
        sleep: How to wait. Injectable so a test asserting on pacing does
            not have to.

    Returns:
        A :class:`ComparisonReport` with ``len(prompts) * len(cases) *
        repeats`` rows.
    """
    results: list[RouterResult] = []
    started_any_call = False
    for prompt in prompts:
        gateway = gateway_factory(prompt.contract)
        planner = Planner(gateway, tools=PLANNING_TOOLS)
        for case in cases:
            user = users.get(case.actor)
            accepted = tuple(sorted(case.accepted_first_routes))
            for repeat in range(repeats):
                if pause_seconds and started_any_call:
                    sleep(pause_seconds)
                started_any_call = True
                state = _build_state(case, user)
                try:
                    decision = planner.plan(state)
                except Exception as error:  # noqa: BLE001 - a failed row, not a failed report
                    logger.warning(
                        "prompt %s case %s repeat %d failed: %s",
                        prompt.name,
                        case.case_id,
                        repeat,
                        error,
                    )
                    results.append(
                        RouterResult(
                            prompt=prompt.name,
                            case_id=case.case_id,
                            repeat=repeat,
                            chosen_route=None,
                            chosen_tool=None,
                            accepted_first_routes=accepted,
                            matched=False,
                            hallucinated_tool=False,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=None,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                    continue

                pair = (decision.route, decision.required_tool)
                record = gateway.telemetry.records[-1] if gateway.telemetry.records else None
                results.append(
                    RouterResult(
                        prompt=prompt.name,
                        case_id=case.case_id,
                        repeat=repeat,
                        chosen_route=decision.route,
                        chosen_tool=decision.required_tool,
                        accepted_first_routes=accepted,
                        matched=pair in case.accepted_first_routes,
                        hallucinated_tool=(
                            decision.route == "fail"
                            and _HALLUCINATION_MARKER in decision.rationale
                        ),
                        input_tokens=record.input_tokens if record else 0,
                        output_tokens=record.output_tokens if record else 0,
                        cost_usd=record.cost_usd if record else None,
                    )
                )
    return ComparisonReport(results=tuple(results), prompts=tuple(prompts))
