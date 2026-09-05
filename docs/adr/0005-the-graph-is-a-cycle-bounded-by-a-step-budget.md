# 0005 — The graph is a cycle, bounded by a step budget rather than by missing edges

**Status:** Accepted (2026-09-05)

## Context

Before this step the transition table let a tool call go to `answer` or to
`fail` and nowhere else. That made the graph a single shot: one decision, one
action, one reply. It was safe in the way a machine with no moving parts is
safe, and it could not do the thing the assistant is for. "What could go wrong
on atlas, and is the budget covering it?" needs two calls and a reply composed
from both. "Has M2 slipped?" may need a document search, and then a tool call
the search revealed was the real question.

The runtime being built here is a reason-act loop: decide, act, observe, decide
again. Adding the observe-and-decide-again edge is what makes it one. It is also
what makes non-termination possible for the first time — a planner that keeps
choosing the same tool, a tool whose result never satisfies the question, a
routing bug that produces a state nobody ends.

So the question is not whether to allow the cycle. It is what stops it.

## Decision

**A `think` route, and edges back to it from `call_tool` and from
`retrieve_project_documents`.** Thinking is a route rather than an implicit
return to the unrouted start, so every re-plan is a declared edge in `ALLOWED`
and appears in the trace as its own step. `think -> think` is deliberately not
an edge: a planning step that produces no action is a planner bug, and the table
should say so rather than permit a spin.

**A successful tool call routes back to `think`, not to `answer`.** This costs a
second model call per tool call and buys the loop: the model sees what its own
action returned and decides whether that is enough. An edge straight to `answer`
would leave the cycle technically present and never taken, which is the worst of
both — the machinery of a loop and the behaviour of a single shot.

**`WorkflowRuntime.max_steps` is the ceiling, and it lives on the engine.**
`AgentState.step_count` counts node executions; `max_steps` limits them. A run
that reaches the limit without a terminal state is forced to `fail` with the
typed `FailureMode` member `max_steps_exceeded` and a `run_failed` trace event.
The default is 8 — a search, a tool call, an answer and the thinking between
them come to five.

**A retry is not an edge.** When a tool comes back `rate_limited` and the retry
budget allows another attempt, the node returns a state on the *same* route with
`retry_count` raised, and the engine runs that node again. `call_tool ->
call_tool` stays out of the table. The attempt is therefore counted by the step
budget, appears in the trace as its own step, and needs no edge that would also
permit loops nobody intended.

## Consequences

The step budget is now load-bearing rather than defensive. It is the only thing
between a stuck planner and a run that never returns, which is why it is checked
before every node execution rather than at the end, and why exhausting it
produces a failure with the full trace attached instead of an exception.

`run_failed` is a separate `EventKind` from `failed`. `failed` means a node
looked at what happened and assigned a reason; `run_failed` means no node did.
A reviewer counting how runs end has to be able to tell "the tool broke" from "we
lost control of the graph", and one of those is a bug in this repository.

The pause is checked before the budget. Waiting on a human is not work, and
charging steps for it would fail a turn that nothing is wrong with.

Cost per turn went up. A read tool call is now two model calls, not one: the
planner picks the tool, and the planner is asked again with the result in front
of it. That is the price of the loop and it is visible in the telemetry as two
`routed` records.

The guard is proved twice in the tests, because it has two shapes. A planner
that keeps calling the same tool is the realistic hang; an injected node that
returns its argument unchanged is the shape no real node can produce. Testing
only the second would leave the guard proved against a situation that cannot
occur.

## Alternatives considered

**Keep the single-shot graph.** Rejected: it cannot answer questions that need
two actions, and it makes the step budget decorative — a guard that can only fire
on a bug in code that has no loop is a comment, not a control.

**Let the tool node answer directly, and re-plan only when it fails.** Rejected.
It puts the "is this enough?" judgement in a node, which is where the planner's
job would then be done badly and untestably: the node has no view of the
question, only of one result.

**Bound the loop with a wall-clock timeout instead of a step count.** Rejected.
A timeout is non-deterministic — the same run passes on a fast day and fails on a
slow one — and a trace that says "took too long" does not say what was looping.
Steps are countable, reproducible in a test, and name the node that kept firing.

**Bound it with a spend limit in tokens or dollars.** Rejected as the wrong
first control, though a reasonable second one. It stops the same runaway more
expensively and answers a different question: how much a turn cost, not whether
it terminated. The gateway already records cost per call; a budget over that
belongs with the telemetry, not in the loop.

**Let the node retry a rate-limited call internally.** Rejected: it spends time
outside the counter that is supposed to bound the turn, and hides the attempt
from the trace. The retry is a step, so it should be counted as one.

**Add `think -> think` so a planner can refine.** Rejected. There is nothing to
refine against — no new observation arrives between two consecutive thoughts, so
the second one is either the same decision or a differently-sampled one, and a
graph that permits resampling until it likes the answer is not one whose
decisions can be audited.
