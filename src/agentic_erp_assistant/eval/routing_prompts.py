"""Three planner contracts, compared before any of them is kept.

gap-plan.md's Phase L (gap 13): the manual walkthrough found "Why is
milestone M2 late and by how much?" (R1) taking the retrieve-and-cite route
once in four tries, and the other three answering the ERP field alone,
dropping the explanation half of the question. `PLANNER_CONTRACT`'s rule 1
(a quoted explanation needs documents) and rule 2 (an ERP field needs a tool)
both match a compound question like that; nothing said which wins.

The selection rule, written here -- before ``eval/routing.py``'s comparison
is ever run, so this file cannot be edited to fit a number it did not like:

    Highest overall match rate across all six cases wins. A tie goes to the
    fewer hallucinated tool calls, then to the shorter contract. V1_DIRECT
    keeps its place (no promotion) unless a candidate beats it on the R1
    case specifically *and* loses on no other case -- a fix that helps the
    compound question at the cost of a case that already worked is not a
    fix.

V2_COMPOUND and V3_EVIDENCE_FIRST are candidates aimed at R1 from two
different angles: one adds a rule, the other reorders the existing ones.
Rules 3-5 and the three non-negotiables are identical in all three, so the
diff a reviewer needs is just rule 1 (and, in V3, rule 2 alongside it).

What was deliberately not built as a fourth candidate: a "policy-first"
variant that asks the planner to apply authorization or approval policy
before choosing a route. Rejected on the same grounds ADR 0016 already
settled -- authorization is the gateway's job, checked structurally, and a
planner that started refusing calls on the actor's behalf would be measured
on a job it must not have. The A9 case (`tomas`, who cannot write) is in
the comparison set specifically to catch a candidate that tries: the
correct route for `tomas` is still `request_approval` or `call_tool`, same
as any actor asking the same thing -- the refusal happens at the gateway,
invisibly to the planner, and a candidate that answers `refuse` there has
started doing a job that was never its to do.
"""

from dataclasses import dataclass

from agentic_erp_assistant.llm.prompts import PLANNER_CONTRACT

__all__ = ["RouterPrompt", "V1_DIRECT", "V2_COMPOUND", "V3_EVIDENCE_FIRST", "ALL_PROMPTS"]


@dataclass(frozen=True)
class RouterPrompt:
    """One candidate developer-block contract, named for the report."""

    name: str
    contract: str


V1_DIRECT = RouterPrompt("v1-direct", PLANNER_CONTRACT)
"""The production contract, imported rather than copied -- the baseline
cannot drift from what production actually sends, because it is not a
second string, it is the same one."""


_V2_RULE_1 = (
    "1. If the question needs something written down in a project document -- a\n"
    "decision, a commitment, an explanation, anything that has to be quoted -- call\n"
    "search_project_documents first. Facts that must be cited come from documents,\n"
    "not from memory. A question that asks both for a field and for the reason\n"
    "behind it (\"why ... and by how much\") is a document question: the ERP holds\n"
    "the number, never the explanation, and an answer that gives only the number\n"
    "is incomplete.\n"
)

V2_COMPOUND = RouterPrompt(
    "v2-compound",
    PLANNER_CONTRACT.replace(
        "1. If the question needs something written down in a project document -- a\n"
        "decision, a commitment, an explanation, anything that has to be quoted -- call\n"
        "search_project_documents first. Facts that must be cited come from documents,\n"
        "not from memory.\n",
        _V2_RULE_1,
    ),
)
"""V1 plus one sentence on rule 1, naming the compound-question shape
directly. The narrowest possible edit: nothing else in the contract moves."""


_V3_RULES_1_2 = (
    "1. Decide what kind of fact the reply needs before deciding where to get it.\n"
    "If any part of the question needs something quoted -- a decision, a\n"
    "commitment, an explanation of why something happened -- call\n"
    "search_project_documents first, even if the question also names a field the\n"
    "ERP holds. A reply that gives the field but not the explanation a\n"
    "compound question also asked for is incomplete.\n"
    "2. Only when nothing about the question needs a quote -- it asks purely for a\n"
    "field the ERP holds, such as a milestone's status, a sprint's burn-down, a\n"
    "budget, or the open risks -- call that tool with the identifier the user\n"
    "gave.\n"
)

V3_EVIDENCE_FIRST = RouterPrompt(
    "v3-evidence-first",
    PLANNER_CONTRACT.replace(
        "1. If the question needs something written down in a project document -- a\n"
        "decision, a commitment, an explanation, anything that has to be quoted -- call\n"
        "search_project_documents first. Facts that must be cited come from documents,\n"
        "not from memory.\n"
        "2. If the question asks for a field the ERP holds -- a milestone's status, a\n"
        "sprint's burn-down, a budget, the open risks -- call that tool with the\n"
        "identifier the user gave.\n",
        _V3_RULES_1_2,
    ),
)
"""Rules 1-2 restructured: name what kind of fact is needed, then route --
rather than V2's added sentence, this asks the model to reason about the
question's shape before matching either rule."""


ALL_PROMPTS: tuple[RouterPrompt, ...] = (V1_DIRECT, V2_COMPOUND, V3_EVIDENCE_FIRST)
