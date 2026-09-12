# 0019 — A repeated call is refused a re-run, and the planner is forced to answer instead

**Status:** Accepted (2026-09-12)

## Context

The manual walkthrough (`docs/manual-test.md`, commit `638adbc`) found A11
genuinely broken: `orion.lead` asks for a medium risk to be recorded, an
approver says yes, `create_risk` executes and writes it — and the turn never
replies. The trace showed why: after the write returned `ok`, `think()`
re-ran the planner as it always does after a tool call, and the model chose
`list_risks(project_id=orion)` — a call that had already succeeded once,
before the write, as `PLANNER_CONTRACT`'s own instruction to check for a
duplicate before calling `create_risk`. It did this three more times before
the eight-step budget ended the run with `max_steps_exceeded` and no answer.
The write, the approval, the audit trail, and the project binding were all
correct throughout; only the reply was missing.

Two things in the prompt made this more likely than it needed to be, not less.
`PLANNER_CONTRACT`'s non-negotiables said "never say or imply that anything
has been recorded" about `create_risk` — true before approval, and never
retracted after it, so the model had no sanctioned way to say what the
observation in front of it already confirmed. The existing repeat rule ("do
not repeat a call... if it failed... repeating it will fail the same way")
only reasoned about failure; it gave no reason not to repeat a call that had
*succeeded*, which is exactly what happened.

`docs/gap-plan.md`'s Phase J, written before this ADR, proposed closing the
gap with two guards: force tools off unconditionally on the very first
`think()` after any successful write, and force them off on a detected
repeat. The first was implemented, tested against the live stack, and
reverted in the same sitting — see Decision, and the alternative it lost to.

## Decision

**A decision is compared against what already succeeded this turn, not
against what happened last.** `engine/nodes.py::_repeats_a_success(state,
decision)` renders the decision's tool and arguments with
`state/tool_request.py::summarize_tool_call` — the same renderer the tool
gateway already uses for `AuditRow.arguments_summary`, now stamped onto every
`ToolOutcome` as its own `arguments_summary` field — and checks it against
every prior observation with status `ok`. A match means the planner is
re-asked once, immediately, with tools withheld; anything else proceeds as
before, including a second write that has never been attempted this turn.

**Tools are withheld by taking the option away, not by asking in prose.**
`ToolCallingClient.call_with_tools(..., allow_tools: bool = True)` sends
`tool_choice: "none"` on the wire when `False`; the tools are still offered
(the model may still need the definitions to make sense of its own prior
calls in `observations`), but the result is guaranteed content. `LLMGateway`
threads `allow_tools` through `decide()`/`call_tools()`, and
`Planner.plan(state, *, offer_tools: bool = True)` is the layer that turns
"was this call forced" into "was a tool call in the reply legal" — with
`offer_tools=False`, a tool call surviving anyway becomes
`Planner._unreadable` (`route="fail"`), because the real provider's
`tool_choice: "none"` cannot produce one, so seeing one at all means
something standing in for the client did not honor the request.

**A forced call that still names something fails as `planner_loop`, not
`max_steps_exceeded` and not `provider_failure`.** A dedicated `FailureMode`
member, checked structurally by `engine/nodes.py::think()` regardless of what
route the forced decision claims (belt-and-suspenders beside `Planner`'s own
guarantee — the graph does not trust a `PlannerPort` implementation to be the
only thing enforcing the contract it satisfies). `max_steps_exceeded` says
the budget ran out with the turn still undecided; `planner_loop` says a
repeat was refused before it ever had to run that far, in one extra model
call rather than eight.

**The prompt is edited too, but only as a supporting fix.** The
non-negotiable about `create_risk` now says what to do once an observation
confirms it: state exactly what was recorded, and stop checking. The repeat
rule now names success as well as failure. Both cost nothing and make the
common case better sooner, but neither is what makes the turn end — the
guard above is.

## What was tried and rejected, live

The gap plan's first design forced tools off unconditionally on the very
first `think()` after *any* successful write, reasoning that a write is
consequential enough to deserve an immediate, unconditional answer. Implemented
and run against the full suite, it broke
`tests/engine/test_orchestrator.py::test_a_resumed_turn_can_pause_again_and_waits_anew`:
a turn that asks for two different risks to be recorded in sequence is
*supposed* to reach `request_approval` a second time right after the first
write succeeds, with a different tool call and different arguments. Forcing
tools off after every write would have refused that legitimate second
proposal along with A11's unproductive repeat — the rule could not tell "a
new action" from "the same action again" because it never looked.

The repeated-call guard replaces it entirely, not alongside it: it is both
narrower (only an *identical* call is refused, so two different writes are
untouched) and sufficient (A11's actual shape was `list_risks` repeating a
call already sitting in `observations`, caught on the first repeat). No
`ToolOutcome.mutating` field was needed either — an earlier draft added one
to know whether the *last* observation was a write, purely to support the
now-rejected design; removed along with it, since nothing else reads it.

## Consequences

`ToolOutcome` gains `arguments_summary: str = ""`, a defaulted field (no
`STATE_VERSION` bump — see that constant's own docstring on what does and does
not require one). `tools/gateway.py::_summarize` moved to
`state/tool_request.py::summarize_tool_call`, generalized from taking a whole
`ToolRequest` to taking `(tool_name, arguments)`, because `engine/nodes.py`
needed to render a decision that has neither yet — `state/` is where both
`ToolRequest` and `ToolOutcome` already live, and neither `tools/` nor
`engine/` may import the other's concrete module, only `state/`.
`ARGUMENTS_SUMMARY_MAX_CHARS` moved with it; `tools/models.py` re-exports both,
unchanged for every existing caller.

Every fake standing in for `DecisionModel` (the planner's model-layer port)
now needs an `allow_tools: bool = True` parameter, the same one-time tax ADR
0016 describes for `preflight`. Fakes standing in for `ToolCallingClient` (the
provider port) need no such change: `allow_tools` is omitted from the call
entirely when `True`, the same convention `on_delta` already established, so
every client fake written before this ADR stays valid.

Verified live: the A11 request as `orion.lead`, approved by `sponsor`, now
replies naming the recorded risk and project (`step_count=5`, `failure=none`)
instead of ending with no answer. gpt-4o did not happen to repeat the call on
either live attempt made after the prompt edit landed — itself suggestive
that the prompt fix helps in the common case — so the guard's structural
firing is proven by `tests/engine/test_think_after_write.py` (a fake that
repeats a call, and a fake that ignores `offer_tools=False` entirely) rather
than by a live trace that shows `tool_choice=none` in `model_calls.detail`.
A live repeat remains possible and remains caught; it was simply not observed
on the two attempts made.

## Alternatives considered

**A smaller step budget after a write.** Rejected: it turns the symptom into
a different symptom — the run still ends with no answer, just sooner, and the
trace still reads `max_steps_exceeded` rather than naming what actually
happened.

**A line in the observation itself** ("this has already been recorded, do
not check again"). Rejected: the observation already says `create_risk -> ok:
Recorded R-6 …` — that *was* the contradiction the prompt's own wording
created against a stale non-negotiable. Adding a second sentence to the data
the model reads is answering a prompt problem with more prompt, when the
actual defect was the two not agreeing with each other in the first place.

**A dedicated `post_write_answer` node.** Rejected: it would be one more node
for a decision `think()` already makes, with the only difference being which
tools it is handed. `GraphNodes`'s four ports and the transition table both
stay exactly as ADR 0005 and ADR 0006 left them.

**Detect a hallucinated tool name and a repeated call with the same
mechanism.** Considered and kept separate: `Planner._unreadable` already
handles "the model named a tool that does not exist," a decision-layer
problem the planner alone can see. A repeat is a *turn-level* fact — it
depends on `observations`, which the planner does not hold — so it belongs in
`engine/nodes.py`, where the state actually lives.
