"""Six fixed cases, held constant across every candidate contract.

Chosen from `docs/manual-test.md` so each is already reviewed, and so the
run order in that handbook is exactly where a human re-checking a number by
hand would look. Two of the six ask for a write -- the reference project's
own proportion -- because a fix aimed at the R1 finding must not quietly
cost the approval path, and a routing report that only ever asked read
questions could not tell.
"""

from dataclasses import dataclass

from agentic_erp_assistant.reasoning.decision import DecisionRoute

__all__ = ["RoutingCase", "DEFAULT_CASES"]


@dataclass(frozen=True)
class RoutingCase:
    """One question, one actor, and every first route that counts as correct."""

    case_id: str
    actor: str
    request: str
    accepted_first_routes: frozenset[tuple[DecisionRoute, str | None]]
    note: str


DEFAULT_CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        case_id="R1",
        actor="priya",
        request="Why is milestone M2 late and by how much?",
        accepted_first_routes=frozenset({("retrieve_project_documents", None)}),
        note=(
            "The finding. Documents-first is unambiguous in this graph: "
            "retrieve_project_documents is terminal and the composer never "
            "sees this turn's own tool observations, so a tool-then-documents "
            "path would drop the 'why' half of the question only by luck -- "
            "and the status report holds both the delay and its cause."
        ),
    ),
    RoutingCase(
        case_id="T1",
        actor="priya",
        request="What is the status of milestone M2?",
        accepted_first_routes=frozenset({("call_tool", "get_project_status")}),
        note=(
            "The live-field control: a candidate that pushes everything to "
            "documents to fix R1 loses here, visibly."
        ),
    ),
    RoutingCase(
        case_id="R10",
        actor="priya",
        request="What is the weather forecast in Hanoi next week?",
        accepted_first_routes=frozenset(
            {("refuse", None), ("retrieve_project_documents", None)}
        ),
        note=(
            "Out of scope. The handbook accepts either the planner's own "
            "refusal or a search that the similarity gate then empties -- "
            "both are the guardrail working."
        ),
    ),
    RoutingCase(
        case_id="T9/1",
        actor="priya",
        request="How is the sprint going?",
        accepted_first_routes=frozenset({("clarify", None)}),
        note="Under-specified (no sprint named). No candidate may start guessing.",
    ),
    RoutingCase(
        case_id="A1",
        actor="priya",
        request=(
            "Record a high severity risk on atlas: hypercare staffing is not "
            "confirmed for the M2 cutover."
        ),
        accepted_first_routes=frozenset(
            {("request_approval", "create_risk"), ("call_tool", "list_risks")}
        ),
        note=(
            "The approval path. list_risks first is what the contract's own "
            "non-negotiable asks for, checking for a duplicate before the write."
        ),
    ),
    RoutingCase(
        case_id="A9",
        actor="tomas",
        request=(
            "Record a medium risk on atlas: warehouse depot hardware refresh "
            "is unfunded."
        ),
        accepted_first_routes=frozenset(
            {("request_approval", "create_risk"), ("call_tool", "list_risks")}
        ),
        note=(
            "tomas cannot write (no project.risk.write) -- but the planner must "
            "not be the one to say so. The correct first route is the same one "
            "any actor asking this gets; the gateway's preflight refuses the "
            "call invisibly to the planner (ADR 0016). A candidate that answers "
            "`refuse` here has started doing the gateway's job."
        ),
    ),
)
