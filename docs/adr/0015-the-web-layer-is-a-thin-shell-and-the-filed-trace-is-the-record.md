# 0015 — The web layer is a thin async shell, streamed text is a preview, and the filed trace is the record

**Status:** Accepted (2026-09-11)

## Context

Every layer below the web existed and was tested against fakes (fact 0 in the
end-to-end code plan): the engine, the tool gateway, retrieval, memory, and the
Postgres adapters. Nothing composed them against a real model, and nothing put
a browser in front of a person. The brief asked for a chat with a dev-only
actor switch (no login), streamed answers, and the agent's decisions,
citations, and trace shown alongside the reply — not after it, and not only on
request.

Three facts about what already existed shaped every decision below, and none
of them were negotiable without rewriting layers this plan was not asking to
touch:

1. **The engine is synchronous and blocking.** `WorkflowRuntime.run` loops
   over nodes; each node makes blocking HTTP calls (a model, Qdrant,
   Postgres). An async web framework calling it directly would block the
   event loop for the length of a turn — every other request queued behind
   it.
2. **The answering call is structured output.** `GroundedAnswer` arrives as
   one JSON object. "Stream the answer" cannot mean forwarding provider
   fragments verbatim — half of a citation object mid-flight is not
   something a chat bubble should render.
3. **Ports are per-turn by design.** `RetrievalService.for_context`,
   `MemoryService.for_scope`, and `RunTelemetry(trace_id, store)` are all
   bound to one actor, one project, or one run. The composition root
   therefore has to build the port graph per request, over resources shared
   across the process.

## Decision

**FastAPI + uvicorn, one worker** (D1). One worker because the rate limiter,
the flaky-tool counter, and the mock ERP's write lock are all in-process state
(ADR 0004 already records this limit for the rate limiter specifically); a
second worker would give each of them an unsynchronized copy, silently
doubling every budget. SSE is hand-written (`web/protocol.py`'s
`encode_sse`) rather than a library: it is one-directional, `fetch` already
reads it natively, and the event vocabulary needed to be a typed contract
either way.

**The engine stays synchronous.** A turn runs on a worker thread via
`loop.run_in_executor`, never on the event loop. `web/stream.py`'s
`TurnStream` is the one object that thread and the event loop both touch —
every callback (`step`, `trace_event`, `delta`, `reset`) hands its event to
the loop with `call_soon_threadsafe` rather than touching the queue directly,
because an `asyncio.Queue` is not thread-safe and this is the one
cross-thread handoff in the whole turn.

**Streamed tokens are a preview; the final `answer` event is authoritative**
(D4). `llm/streaming.py`'s `JsonStringFieldExtractor` decodes the `answer`
field's string value out of the structured-output JSON as it arrives, one
character state machine, never a general JSON parser (a prefix of JSON is not
JSON). The grounding check (`_ungrounded`) and the `Sources:` trailer both run
only after the *complete* reply is parsed and validated, so a citation to a
source that was never retrieved turns what was streamed into a refusal — and
the client has to show that refusal, not the words it already rendered.
`ui/src/turnReducer.ts` enforces the rule on the client: once `answer`
arrives, a renderer shows `answer.text`, never the accumulated preview.

**Streaming is bound at the gateway, not threaded through the engine's
ports** (D5). `LLMGateway.stream: AnswerStreamSink | None`, a constructor
field. `PlannerPort`, `AnswerComposerPort`, and `DecisionModel` are
unchanged. A second `LLMGateway` — same client, same budget, no sink — is
built for memory work in every turn (`composition/turn.py`), so a memory
proposal can never stream into the chat, structurally, not by convention.

**Composition is assembled per request, over resources shared per process**
(D6). `composition/resources.py`'s `AppResources` holds the expensive,
connection-bearing things built once (the HTTP client, the Qdrant clients,
the lexical index, the ERP store, a shared rate limiter); `composition/
turn.py`'s `build_turn` assembles the rest — the retriever bound to one
actor's `RetrievalContext`, the memory scope bound to one session, the
telemetry sink stamped with one trace id — new every call, because fact 3
above leaves no other honest place to build them. `composition/` imports the
engine and the trace ports; nothing below it imports back, and `web/` imports
`composition/`, never the reverse.

**Cancellation is the client giving up, not the engine stopping** (D11). A
"Stop" button aborts the client's `fetch`; the turn the server started keeps
running to completion and the trace is still filed. Cooperative cancellation
inside the engine — a node checking a cancellation flag between steps — is a
named follow-up (known gap 2), not something this plan's scope justified
building for a single-user dev demo.

**The read model is plain SQL, not new methods on the write-side ports**
(D12). `persistence/postgres_queries.py::EvidenceQueries` reads the same
nine tables a turn already writes to, through its own connection. The
alternative — adding `pending_approvals()`, `sessions()`, and so on to
`trace/ports.py::PauseStore` or `TraceStore` — would give every fake standing
in for those ports in an engine test a method the engine itself never calls;
a screen's listing need is not a fact the orchestrator depends on.

**Dev toggles live in exactly one place** (D13). `DEV_TOOL_RATE_LIMIT`,
`DEV_FLAKY_STATUS`, `DEV_MAX_STEPS`, and `DEV_HISTORY_TURN_LIMIT` are read
only by `composition/settings.py`, logged loudly at startup
(`Settings.log_dev_toggles`), and never referenced by `engine/`, `tools/`, or
`context/` directly. They exist because the registry's real budgets — 30
reads/60s, an eight-step turn, a six-turn window — are none of them things a
person clicking around a browser demo can exhaust in a reasonable session,
and a scripted test reaches the same paths with a fake clock instead.

**The frontend is React + TypeScript + Vite** (D2), per the brief. A pure
reducer (`turnReducer`) over the typed SSE union is the testable core;
everything else — the three-column layout, the components — is presentation
built on top of it. `tests/web/test_protocol_drift.py` reads
`ui/src/protocol.ts` as text and checks its `type: "..."` literals against
`web/protocol.py`'s `EVENT_TYPES`, so a renamed event fails `pytest`, not a
browser console three files and one language away.

## Consequences

A turn's true record is the filed trace (`runs`, `trace_events`,
`audit_rows`, `model_calls`), not the stream a particular browser tab
happened to see. Killing a tab mid-turn, per D11, still leaves a complete
`runs` row — proved live: the write-and-approve scenarios in
`docs/manual-test.md` read the trace back through `/api/runs/{trace_id}`
after the fact and it matches what streamed, but the stream was never the
source of truth to begin with.

Gateway-internal events (a retry, the preflight's `approval_recorded`) reach
the live stream (`TraceRow.source == "tool_gateway"`, `seq: null`) but not
the database — `ToolGateway.on_event` was already unwired before this plan,
and wiring it to the *stream* was enough to satisfy the brief without also
wiring a second sink into Postgres, which is known gap 3.

`STATE_VERSION = 2` (ADR 0017, required for the web layer to bind a tool call
to an actor's project) means a v1 `runs.state` or `pauses.state` refuses to
load; development handles this with a `TRUNCATE` (`docs/manual-test.md`
§1.3), which is acceptable because nothing was ever written under v1 outside
development.

Two bugs were found only once a real browser was driving this stack, not by
any test written against fakes: `Citation.source_id`'s JSON schema carried no
field-level guidance beyond its bare name, and gpt-4o occasionally wrote the
whole rendered `[doc#locator]` tag into it instead of the plain document id,
turning a genuinely grounded answer into a false refusal (fixed by adding
`description=` to both `Citation` fields); and the three-column CSS grid gave
its children no bounded height, so the composer could be pushed below the
viewport as a conversation grew — the classic flexbox scrolling gotcha
(`flex: 1` cannot shrink without `min-height: 0` on the flexing child), fixed
alongside a client-state bug where recording a freshly-generated session id
after sending a message reused the `session_selected` action, whose case is
built for genuine navigation and clears the message list as a documented side
effect of that. Both are why "the build passes" is not "the feature works":
this project's own finishing checklist requires loading the page and using
it, and this plan is the reason that rule exists.

## Alternatives considered

**An async engine.** Rejected: it would touch every node and every existing
test for a property the web layer can supply at its own edge, for a
single-process, single-worker dev tool that does not need node-level
concurrency.

**WebSocket instead of SSE.** Rejected: the traffic is one-directional
(server events to the client; decisions travel as separate POST requests),
and SSE is exactly that shape over plain `fetch`, with automatic reconnection
semantics a WebSocket would have to reimplement for no benefit this brief
asked for.

**A sink threaded through the engine's ports**, so `WorkflowRuntime` or
`GraphNodes` would carry it directly. Rejected: it would widen the engine's
four-port surface for a browser-only concern, and a fake standing in for
`AnswerComposerPort` in an engine test would need to know about streaming to
remain a valid fake.

**Streaming the raw structured-output JSON to the client**, and parsing it
there. Rejected: it would move the grounding check and the citation
boundary — a security-relevant judgment about what the model is allowed to
assert — into the browser, the one place this project's access rules do not
reach.

**Simulated streaming of an already-finished answer** (write the whole reply,
then drip-feed it to the client). Rejected as dishonest about what is
happening: a person watching a reply "stream" in that shape is not watching
the model think, and the first real token would still take as long to arrive
as the whole answer does today.

**Server-side rendering, or a dependency-free browser-JS UI.** Rejected: the
brief asks for React specifically, and — independent of that — a pure
reducer over typed events is unit-testable in a way a template render is
not; `turnReducer.test.ts` is the proof this plan leaned on throughout.
