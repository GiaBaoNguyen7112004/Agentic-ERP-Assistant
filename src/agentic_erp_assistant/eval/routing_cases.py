"""Eight fixed cases, held constant across every candidate contract.

The original six are chosen from `docs/manual-test.md` so each is already
reviewed, and so the run order in that handbook is exactly where a human
re-checking a number by hand would look. Two of the six ask for a write --
the reference project's own proportion -- because a fix aimed at the R1
finding must not quietly cost the approval path, and a routing report that
only ever asked read questions could not tell.

R12/R13 were added for ADR 0025/0026/0027: the compound, multi-document
request that produced a false refusal, run once as priya (who cannot read
the Q3 budget summary) and once as wei (who can) -- the same request, two
entitlement snapshots, so a routing report can show the pair scores
identically on routing and declaration while `docs/manual-test.md` is
where the citation-level difference between them is actually checked.
"""

from dataclasses import dataclass

from agentic_erp_assistant.reasoning.decision import DecisionRoute
from agentic_erp_assistant.state.reply_contract import ReplyNeed

__all__ = ["RoutingCase", "DEFAULT_CASES"]


@dataclass(frozen=True)
class RoutingCase:
    """One question, one actor, every first route that counts as correct, and
    -- since ADR 0021 -- what the planner should *declare* it needs.

    ``expected_needs`` is scored separately from ``accepted_first_routes``
    (ADR 0020's routing match): a declaration and a routing choice are two
    different model outputs, from two different calls
    (:meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare` vs.
    :meth:`~agentic_erp_assistant.reasoning.planner.Planner.plan`), and a
    contract that routes correctly while declaring the wrong needs is a
    different failure than a contract that declares correctly and routes
    wrong -- one comparison should not average them into a single number.
    """

    case_id: str
    actor: str
    request: str
    accepted_first_routes: frozenset[tuple[DecisionRoute, str | None]]
    note: str
    expected_needs: frozenset[ReplyNeed] | None = None
    """What a correct declaration names, or ``None`` when this case is not
    scored on declaration at all -- a refusal or a write case, where any
    declaration the planner offers is a legitimate answer (see each case's
    own note)."""


DEFAULT_CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        case_id="R1",
        actor="priya",
        request="Why is milestone M2 late and by how much?",
        accepted_first_routes=frozenset({("retrieve_project_documents", None)}),
        note=(
            "The finding. gpt-4o routes this to the tool deterministically "
            "(ADR 0020: 0/15 across three contracts) -- the fix ADR 0021 ships "
            "is not a routing prompt, it is a declared contract "
            "({document_passage, erp_field}) that engine/nodes.py::think "
            "redirects an incomplete answer against structurally, after "
            "routing, not instead of it. This case still measures the "
            "*routing* match ADR 0020 measured; expected_needs measures the "
            "new declaration separately."
        ),
        expected_needs=frozenset({"document_passage", "erp_field"}),
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
        expected_needs=frozenset({"erp_field"}),
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
            "both are the guardrail working. Not scored on declaration: "
            "refusing needs nothing, and a declaration made before the model "
            "even decides the question is out of scope is not wrong either way."
        ),
    ),
    RoutingCase(
        case_id="T9/1",
        actor="priya",
        request="How is the sprint going?",
        accepted_first_routes=frozenset({("clarify", None)}),
        note="Under-specified (no sprint named). No candidate may start guessing.",
        expected_needs=frozenset({"erp_field"}),
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
            "non-negotiable asks for, checking for a duplicate before the write. "
            "Not scored on declaration: a write declares whatever it declares, "
            "per D1's rejection of a needs field on tool arguments -- the "
            "reply here is the record of what was written, not a claim the "
            "completeness check has any business holding to a fact pattern."
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
            "`refuse` here has started doing the gateway's job. Not scored on "
            "declaration, for the same reason A1 is not."
        ),
    ),
    RoutingCase(
        case_id="R12",
        actor="priya",
        request=(
            "Atlas project: pull the current budget and open risks from the ERP, "
            "then cross-check each open risk against the risk register CSV for "
            "its severity, check the Q3 budget summary PDF for whether "
            "contingency has been allocated for high-severity risks, and check "
            "the latest sprint report to see if any of those risks are already "
            "causing schedule slip. Summarize the full picture with sources for "
            "each claim."
        ),
        accepted_first_routes=frozenset(
            {
                ("call_tool", "get_budget_summary"),
                ("call_tool", "list_risks"),
                ("retrieve_project_documents", None),
            }
        ),
        note=(
            "The trace that motivated ADR 0025/0026/0027 (run-"
            "e4feb394274f42c288be90ed37ad8c8e): a compound, multi-document "
            "request that produced a false refusal claiming two of the three "
            "named documents -- both actually within priya's own scope -- were "
            "inaccessible. Live re-runs after the fix consistently open with an "
            "ERP tool; a search-first order that satisfies PLANNER_CONTRACT rule "
            "1's own reading is accepted too, since the request also names "
            "document work. Not scored on whether the reply correctly declines "
            "the Q3 budget summary (priya lacks project.docs.finance.read) -- "
            "this harness scores routing and declaration, not citation content; "
            "see docs/manual-test.md for that assertion."
        ),
        expected_needs=frozenset({"document_passage", "erp_field"}),
    ),
    RoutingCase(
        case_id="R13",
        actor="wei",
        request=(
            "Atlas project: pull the current budget and open risks from the ERP, "
            "then cross-check each open risk against the risk register CSV for "
            "its severity, check the Q3 budget summary PDF for whether "
            "contingency has been allocated for high-severity risks, and check "
            "the latest sprint report to see if any of those risks are already "
            "causing schedule slip. Summarize the full picture with sources for "
            "each claim."
        ),
        accepted_first_routes=frozenset(
            {
                ("call_tool", "get_budget_summary"),
                ("call_tool", "list_risks"),
                ("retrieve_project_documents", None),
            }
        ),
        note=(
            "R12's own case, as wei -- who holds project.docs.finance.read, so "
            "the Q3 budget summary PDF is genuinely in scope this time. Routing "
            "and declaration are expected to match R12's shape exactly; what "
            "differs is downstream of this harness -- wei's reply should cite "
            "budget-summary-q3 where priya's correctly cannot, a distinction "
            "the catalogue (ADR 0026) is what makes possible, and one this "
            "harness does not score. See docs/manual-test.md."
        ),
        expected_needs=frozenset({"document_passage", "erp_field"}),
    ),
)
