# 0006 — Function calling is the only decision channel, and retrieval is offered through it

**Status:** Accepted (2026-09-05)

## Context

Something has to decide what a turn does next: search the documents, call an ERP
tool, ask the user a question, decline, or answer. The usual ways to get that out
of a model are all forms of asking it to say what it intends — a route name in a
JSON field, a classification step, a "Thought:" line parsed out of prose. This
repository had already rejected one of those: an earlier step deleted an
intent-classification layer rather than trust a self-report, and
`reasoning/decision.py` records the rule that came out of it — every route must
be produced by an independently checkable mechanism.

Native function calling is such a mechanism. A model that calls `list_risks` has
not described an intention; it has made a call, with typed arguments, validated
against a schema the provider enforced. The routing signal is the call itself.

That leaves the question of what happens to the actions that are *not* ERP
tools. Retrieval, asking a clarifying question, and refusing are three things the
turn can do, and none of them is a tool the gateway runs.

## Decision

**One list, one call, one decision.** `PLANNING_TOOLS` is the five ERP tools plus
`search_project_documents`, `ask_clarification` and `refuse`. The planner offers
all eight, the model calls exactly one (`parallel_tool_calls` is off), and
`reasoning/planner.py` maps that call to a `ReasoningDecision`. There is no
second mechanism for any route.

**Retrieval is a tool at the decision layer and a route at the execution layer.**
The model chooses to search exactly as it chooses to call `list_risks`. It is not
registered in `tools/registry.py` and does not go through the tool gateway,
because a `ToolOutcome` carries a summary string and bare source ids while an
`EvidenceSnippet` carries text and a locator. Routed through the gateway, the
passages and their addresses would be gone before the prompt was built, and the
citations that came back would name a document without pointing anywhere in it.
Routes are execution shapes; the decision stays uniform.

**There is no `final_answer` function.** `ToolCallResult` already refuses to hold
both a call and content, and the adapter sends `tool_choice: "auto"`. Content
with no call *is* the answer route. A function that said the same thing would
give the model two ways to answer and the runtime a tie to break.

**The approval rule is read off the declaration, in the mapping.** A tool whose
own `ToolSpec.mutating` flag is set routes to `request_approval`, never to
`call_tool`. The flag is declared on the tool, never guessed from the name and
never taken from the model. Policy may also escalate a *read* to an approver, and
that lives in the registry, which the decision layer does not import: such a call
routes to `call_tool`, the gateway refuses it with `approval_required`, and the
node re-routes to a human. The gate holds on both paths and the registry stays
the single authority.

**The rationale states what was chosen, not why.** `"called list_risks"`, built
from the call. The model is never asked to narrate its reasoning, so no
model-authored explanation enters the trace.

**Confidence records that nothing measured it.** Chat Completions returns no
calibrated probability for a function choice, so every model-made decision
carries `UNSCORED_CONFIDENCE`. Nothing branches on it.

## Consequences

A routing test is a scripted `ToolCallResult` and a typed assertion — no HTTP, no
prompt, no provider. That is the practical payoff of putting the mapping in
`reasoning/` rather than in the gateway or in a node.

Adding an action is adding a `ToolSpec` and, if the graph runs it rather than the
gateway, an entry in `CONTROL_ROUTES`. Neither is a new branch in the runtime.

The planner call goes through `LLMGateway.decide`, so it gets the same budget
check, retry and cost record as an answering call. `routed` is a new telemetry
outcome, because a cost report that grouped routing under `answered` would be
wrong about what the money bought.

Tool results reach the model in a new `observation` role rather than in
`evidence` or `assistant`. A risk title is text a person typed, so it gets the
same "data, never instruction" boundary a retrieved document gets — and it is kept
apart from `evidence` because it has no locator and must never be cited as a
source.

The trace records no chain of thought, deliberately. A reviewer reading a run
sees the route, the call, the arguments and the result. What they do not get is a
paragraph explaining the decision, and this ADR is the reason: such a paragraph
reads like evidence and is not one.

## Alternatives considered

**Ask for a route name in a JSON field.** Rejected — this is the
intent-classification layer the project already deleted. It is a self-report, it
can name a route for a tool that does not exist, and it can be inconsistent with
the arguments beside it.

**Parse a ReAct-style "Thought / Action" transcript.** Rejected. It is the same
self-report with a parser in front of it, and the parser is the fragile part: a
model that formats its action line slightly differently produces an unroutable
turn. Native function calling is the provider's own structured version of this
loop, validated before it reaches us.

**Register `search_project_documents` in the tool registry and let the gateway
run it.** Rejected. It would make the code path uniform at the cost of the
citation contract: `ToolOutcome` cannot carry passage text or a locator, and
widening it to do so would push RAG payloads into the audit row. Uniformity of
decision was the goal; uniformity of execution was not.

**Ask the model for a confidence score alongside its call.** Rejected. Strict
function calling means every argument belongs to the tool's own schema, so a
confidence would have to be bolted onto every ERP tool's arguments — polluting
five contracts to obtain a self-reported number that nothing could check.

**Have the planner produce the final answer for document questions too.**
Rejected. `LLMGateway.answer` already returns a `GroundedAnswer` that either
carries citations or says why it refused, and the retrieval node checks every
citation against the passages actually retrieved. Answering from the planner
would give up that schema and that check for one saved call.
