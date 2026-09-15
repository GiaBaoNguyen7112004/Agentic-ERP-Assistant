# CLAUDE.md

Mini Agentic ERP Assistant — a browser chat assistant for project-delivery operations
(sprints, budget, risk) backed by a from-scratch agent runtime. This is a graded final
project: architecture, grounding, safety, and evidence are assessed, and the author must
be able to defend every trade-off verbally.

## Hard constraints (never violate)

1. **The core runtime is written in this repo, from scratch.** Graph execution, typed
   state transitions, memory policy, context construction, tool execution, retry,
   approval routing, and trace capture must be project code.
2. **No agent framework may provide that core.** LangGraph, CrewAI, Agno, AutoGen,
   LangChain agents, LlamaIndex agents and equivalents must not appear in the
   orchestration, agent loop, memory manager, tool router, retry engine, or graph
   runtime. Do not add them as dependencies without the user explicitly asking; if asked,
   confirm the usage is outside the assessed core (benchmarking, experiments in a
   separate directory) and say so in the commit.
   Plain SDKs are fine: the provider SDK (`anthropic`), an HTTP client, a vector/BM25
   library, a web framework.
3. **Every factual answer about project documents carries citations.** An answer derived
   from RAG without resolvable source references is a defect, not a rough edge.
4. **Write actions require explicit in-context approval.** No tool that mutates ERP data
   executes without a routed approve/deny decision recorded in the trace.
5. **Every request emits a trace.** Node transitions, prompts, tool calls, retries,
   approvals, token/latency counts. Traces are the audit evidence and are as important as
   the feature itself.
6. **LLM access goes through a typed provider port.** No provider SDK import outside the
   adapter layer, no behavior smuggled into free-form prompt strings that should be typed
   state.

## Stack

- Python (see `.python-version`; currently 3.14), managed by **uv**. `pyproject.toml` uses
  the `uv_build` backend; package lives under `src/agentic_erp_assistant/`.
- Entry point: `agentic-erp-assistant` -> `agentic_erp_assistant:main`.

### Commands

```bash
uv sync                      # install/refresh the environment
uv run agentic-erp-assistant serve # start the web layer (FastAPI + the built React app)
uv run pytest                # tests (once pytest is a dev dependency)
uv add <pkg>                 # add a runtime dependency
uv add --dev <pkg>           # add a dev dependency

docker compose up -d qdrant                          # the vector store
uv run python scripts/ingest_documents.py --dry-run  # what would be embedded
uv run python scripts/ingest_documents.py            # embed and store, for real
uv run python scripts/run_retrieval_evaluation.py    # write the evidence report
uv run python scripts/run_routing_comparison.py      # three planner contracts vs six cases (ADR 0020), plus a declaration row per case (ADR 0021)
uv run python scripts/build_pdf_fixtures.py          # re-render the PDF fixture

docker compose up -d postgres                        # the evidence store
uv run python scripts/init_postgres.py               # create the nine tables (the dev database)
uv run python scripts/init_postgres.py --test        # a second, _test database -- ADR 0018
uv run pytest -m postgres                            # the SQL adapters, against the _test one only
uv run pytest -m live                                # a real OpenAI call; skipped without OPENAI_API_KEY
uv run python scripts/demo_memory_session.py         # two turns, the window, and what was kept
uv run python scripts/demo_pause_across_restart.py   # a pause survives a restart, for real

uv run python scripts/run_turn.py --actor priya "Why is milestone M2 late?"   # one real turn
uv run python scripts/run_turn.py --actor priya --approve <trace_id>          # resume a pause

npm --prefix ui install && npm --prefix ui run build # the React app -> web/static/
uv run agentic-erp-assistant serve                   # http://127.0.0.1:8000
```

Never edit `[project.dependencies]` by hand — use `uv add` so the lockfile stays in sync.

## Intended architecture

The package is a scaffold today. Build toward these boundaries; one concern per module,
and keep the dependency direction pointing inward (the engine never imports the web layer).

```
src/agentic_erp_assistant/
  engine/      graph engine, node contracts, typed state transitions, retry
               (named `engine/` because it *is* the graph engine; "runtime" named both
               the package and the execution concept, and the ambiguity cost more
               than the word was worth)
  state/       the reasoning state model (typed, serializable, versioned)
  reasoning/   the decision layer: what to do next (route, tool, approval) and
               why a turn failed -- typed fields, never prose
  memory/      the write gate, the task in flight, the session's residue, and
               the stores behind them. Store only what is useful later: the
               policy is a pure function and the model may only propose
  context/     context construction: what gets into the prompt and why
  llm/         provider port (protocol), contracts, prompts, tokens, retry,
               tool specs, cost telemetry, and the gateway that orders them
  llm/adapters/  vendor adapters (openai_chat.py); the only place a provider is named
  rag/         ingestion, chunking, index, retrieval, citation objects
  tools/       MCP-style tool boundary: typed schemas, router, approval gating
  erp/         mock ERP provider over project data, project-bound (ProjectErp, ADR 0017)
  guardrails/  input/output checks, refusal + escalation paths
  trace/       structured trace records, run store, export for evidence
  persistence/ the Postgres adapters behind trace/memory's ports, schema, and
               EvidenceQueries -- the plain-SQL read model web/ uses
  eval/        offline eval harness, datasets, scored runs
  composition/ where the environment becomes typed config (settings.py), the
               process-wide clients get built once (resources.py), and one
               turn's ports get assembled per request (turn.py) -- the one
               place engine/llm/tools/rag/memory's ports meet a real adapter
  web/         FastAPI + hand-written SSE over the engine, and the built React
               app (ui/) it serves -- ADR 0015
```

Design rules:

- **Ports are Protocols/ABCs, adapters are swappable.** LLM provider, retriever, ERP
  backend, and tool transport each sit behind an interface with at least one fake used in
  tests. Route/model selection is data, not `if` chains scattered through the runtime.
- **State is typed and explicit.** Nodes take state, return state (or a transition). No
  hidden globals, no mutation of shared dicts. Prefer dataclasses/pydantic models over
  free-form dicts anywhere state crosses a boundary.
- **Tools declare typed input/output schemas** and a `mutating: bool`-style flag that the
  router uses to force approval. A tool call that fails validation never reaches the
  implementation.
- **Retry is a runtime concern**, budgeted and traced — not a `try/except` sprinkled into
  each tool.
- **Streaming is end-to-end**: provider tokens -> runtime -> transport -> UI, with the
  trace still complete when a stream is cancelled.

## Working conventions

- Use standard-library `logging`, never `print`, outside the CLI entry point.
- Type-annotate public functions; keep modules import-light so tests run fast.
- Every non-trivial component ships with tests using fakes, not live provider calls. Live
  calls belong only in explicitly marked integration tests.
- Mock ERP/project data and RAG source documents live in a repo directory (e.g.
  `data/`), are small, synthetic, and readable in review — no real customer data.
- Secrets come from the environment only. Never commit an API key; never log prompt
  contents containing credentials.
- When a design decision is made (route policy, memory promotion rule, chunk strategy),
  record the reasoning in the commit message or an ADR under `docs/adr/` —
  the defense depends on being able to explain *why*, not just *what*.
- If work is a team submission, keep the contribution map current (owner per component,
  tests, evidence).

## Finishing a change

Do not report a task done until both steps below have actually run and passed. If a step
fails, fix it — reporting a green result you did not observe is worse than reporting a
red one.

### 1. Verify it builds — backend *and* frontend

Backend (Python has no build step, so force the equivalent):

```bash
uv run python -m compileall -q src   # syntax errors anywhere in the package
uv run python -c "import agentic_erp_assistant"   # import-time errors
uv run pytest -q                     # once tests exist
```

Add `uv run mypy src` / `uv run ruff check src` to this list as soon as those tools are
installed, and update this file when they are.

Frontend — React + TypeScript + Vite in `ui/` (see Open decisions):

```bash
npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build
uv run agentic-erp-assistant serve        # then open http://127.0.0.1:8000 and use it
```

The build writes to `src/agentic_erp_assistant/web/static/` (git-ignored) and the last
step is not optional: load the page, send a message as an actor with document access,
watch it stream and end with clickable citations, and confirm the browser console is
free of errors. A UI that was never loaded in a browser has not been verified —
`npm run build` succeeding proves the code compiles, not that it works.

### 2. Commit the step

Commit at every meaningful step — a component finished, a bug fixed, a decision made —
not once at the end of a long session. These commits are the project's readable history:
a later session reads `git log` to recover what was built and why, so the message must
carry the reasoning that is not visible in the diff.

```
<area>: <what changed, imperative, one line>

Why this approach and what was rejected. Constraints honored (typed port,
approval gating, trace coverage). Anything deliberately left for later.

Co-Authored-By: Claude <noreply@anthropic.com>
```

Rules:

- One logical change per commit; do not mix a refactor with a feature.
- `<area>` matches a module directory (`engine:`, `rag:`, `tools:`, `web:`, `eval:`).
- Never commit with failing checks, and never use `--no-verify`.
- Work on `dev` or a feature branch, never directly on `main`.
- Standing authorization: committing at these checkpoints is expected and does not need
  to be asked about each time. Pushing, force-pushing, opening PRs, and rewriting history
  still require an explicit request.

## Open decisions

Not yet chosen; ask before assuming, and update this file once settled.

- ~~Web layer~~ — settled: FastAPI + uvicorn, one worker (the rate limiter, the
  flaky-tool counter, and the ERP lock are in-process state, ADR 0004), hand-written
  SSE (`web/protocol.py`'s nine event types, `web/stream.py`'s thread → asyncio
  bridge) over the engine, which stays synchronous — a turn runs on a worker thread
  via `loop.run_in_executor`, never on the event loop. React 18+ TypeScript + Vite in
  `ui/`, building into `web/static/` (git-ignored); `ui/src/turnReducer.ts` is the
  pure, tested core every rendered turn goes through, and
  `tests/web/test_protocol_drift.py` fails the Python suite the day
  `web/protocol.py` and `ui/src/protocol.ts` disagree. `AgentState.project_code`
  is required (`STATE_VERSION = 2`, ADR 0017) and a write is put to a human only
  after the gateway's own checks already passed (ADR 0016) — see ADR 0015 for the
  web layer itself. `agentic-erp-assistant serve [--host] [--port] [--reload]`
  starts it; `scripts/run_turn.py --actor <name> [--session ...] "<question>"` (or
  `--approve`/`--deny <trace_id>`) proves the composition root from the terminal,
  no browser required. `data/users.json` is the dev-only actor directory (no
  login); `OPENAI_CONTEXT_WINDOW` is required in `.env` alongside `OPENAI_MODEL`;
  the `DEV_*` toggles (`DEV_TOOL_RATE_LIMIT`, `DEV_FLAKY_STATUS`, `DEV_MAX_STEPS`,
  `DEV_HISTORY_TURN_LIMIT`) exist only so a browser session can reach paths a
  scripted test reaches with a fake clock instead — see `.env.example` and
  `composition/settings.py`. `DEV_TRACE_MODEL_IO` is a fifth, differently-shaped
  toggle (`docs/trace-inspector-plan.md`): it streams a turn's model call
  prompts and replies to the trace panel live, through
  `llm/inspection.py::ModelCallInspector` — never persisted, off by default,
  and unrelated to any budget a person could otherwise exhaust by hand.
- ~~LLM provider~~ — settled: OpenAI Chat Completions, called with plain `httpx` in
  `llm/adapters/openai_chat.py` (no `openai` SDK anywhere). The model itself is **not** chosen by the
  repo: `OPENAI_MODEL` comes from `.env` with no default, and whatever model is set there
  also needs a reviewed row in `llm/pricing.py`. Routing policy behind the port is still
  open.
- ~~Retrieval backend and citation format~~ — settled: hybrid. Dense search is
  OpenAI embeddings in **Qdrant** (`docker-compose.yml`), the lexical half is a
  from-scratch BM25 in `rag/lexical.py`, and the two are combined with Reciprocal
  Rank Fusion. A citation is `[document_id#locator]` where the locator is the
  format's own address — `§3.2`, `p.4`, `row R-1` — and is the chunk id itself
  (ADR 0009). `OPENAI_EMBEDDING_MODEL` comes from `.env` with no default.
  Still open behind that: `MIN_COSINE_SIMILARITY` in `rag/retriever.py` is
  provisional until the evidence run measures the gap it should sit in, and the
  index router and graph slice (reference steps 13 and 14) are deferred. Since
  ADR 0026, the planner's own routing prompt (and only that prompt) carries a
  `DocumentCatalogue` -- every document this turn's actor is authorized to
  search, filtered from the manifest by the same `rag/access.py::
  is_authorized` check retrieval itself enforces -- so a refusal or a search
  choice is made against what actually exists and what this actor may open,
  never a guess from a tool description alone. A document outside the
  catalogue is never named to the model; the wording only tells it to say a
  named-but-absent document is inaccessible, never that it does not exist.
  Since ADR 0027, `search_project_documents` takes `queries: list[str]` (one
  to three, `SEARCH_QUERY_LIMIT` in `llm/tools.py`) instead of a single
  `query`, and `retrieve_and_answer` runs every one of them in the same node
  — `EVIDENCE_LIMIT` stays a per-query budget, not a total — unions the hits
  in query order and dedupes by citation tag (`GraphNodes._dedupe_by_tag`)
  before composing once. `ReplyContract.document_query` stays a single
  string: it is only the redirect fallback for a model that never searched
  at all (ADR 0021/0025), and one query is the honest size of that
  fallback. A state built before ADR 0027 still runs — `GraphNodes._queries`
  falls back to the legacy singular `query` key, then to the request itself.
- ~~Memory topology~~ — settled: Postgres holds the records (`memories`,
  `intents`, `memory_audit`), a second Qdrant collection indexes them for
  semantic recall, and graph memory is rejected because the required queries are
  lookups rather than traversals (ADR 0013). What is stored is decided by
  `memory/policy.py`, a pure function whose default is to refuse (ADR 0011), and
  memory reaches a prompt in its own role where it can never become a citation
  (ADR 0012). On top of that sits short-term memory (ADR 0014): `session_turns`
  keeps a verbatim, bounded window of the session's recent turns, they reach the
  prompt in a `history` role of their own — not policy-gated, defended
  structurally — and turns evicted from the window are folded into the session's
  `session_summary` through the compaction allow-list, with an audit row.
  `QDRANT_MEMORY_COLLECTION` and `MemoryService.required_scope` are
  the two configuration points. Since the 2026-09 memory refactor (ADR 0022)
  the model is told who it is talking to via a principal block in the system
  role, history is the authority on the conversation itself, and memory is
  only what the user established — the pure policy refuses absence claims,
  self-descriptions, restated replies and unstated preferences under the
  `not_established` rejection reason (ADR 0023), and a turn that refused,
  clarified or failed is never asked to propose at all. A `preference` is
  additionally replaced by *topic*, not only by the exact `(kind, key)` match
  every other kind uses: the proposer's `key` is not enforced stable across
  turns, so `_resolve_conflict` supersedes a live preference whose statement
  shares enough content words with a new one (`TOPIC_OVERLAP_RATIO`,
  preference-only) and keeps the record under the key already stored rather
  than the one just proposed (ADR 0024).
  Still open behind that: nothing infers when the task in flight has changed —
  `SessionMemory.start_intent`/`advance_intent`/`close_intent` are complete and
  are driven by the caller. The `think -> answer` route is held to the reply
  contract the planner declared before the graph ran (ADR 0021): a missing
  need is redirected once, and a reply still short after that is delivered
  marked `failure=incomplete_reply`, never silently. Since ADR 0025, `think ->
  refuse` is held to the same contract -- a refusal with a declared need still
  unmet is redirected exactly once, the same as an answer, except the
  refusal's own message is never kept as a fallback draft, so a redirected
  search that still finds nothing ends in an ordinary, now-tested
  `insufficient_evidence` refusal rather than delivering the model's untested
  claim. A turn with no declaration (`contract=None` — a replay, a hand-built
  state, a declarer that raised) is unchecked, and its trace says so plainly
  rather than looking indistinguishable from one that passed.
- ~~Trace persistence~~ — settled: Postgres (`persistence/schema.py`, nine tables,
  `docker compose up -d postgres && uv run python scripts/init_postgres.py`).
  `web/`'s read model (`persistence/postgres_queries.py::EvidenceQueries`) is
  plain SQL over those tables, deliberately not new methods on the write-side
  ports — a listing query is a screen's need, and widening a port to serve it
  would give every fake standing in for it in an engine test a method the engine
  never calls.
- ~~Eval report format~~ — settled: a JSON report under `evidence/`, one
  subdirectory per harness. `evidence/rag/retrieval-report.json` (hit rate,
  latency, the similarity gap) and `evidence/routing/routing-comparison-
  <date>.json` (ADR 0020: match rate, hallucination count, cost, and every
  row, per planner contract — joined by ADR 0021's declaration match rate,
  one row per case×repeat rather than per prompt, since the declaration
  call does not vary by which routing contract is under comparison) are
  both produced by a script under `scripts/` that never hand-writes a
  number into the file. The planner contract itself is whichever ADR 0020
  names — currently the unedited production `PLANNER_CONTRACT`; no
  candidate in that comparison beat it.
- Test database isolation (ADR 0018) and the `live` marker (ADR-adjacent,
  `tests/live/`): settled as of the gap-plan.md walkthrough.
  `POSTGRES_TEST_URL` points `uv run pytest -m postgres` at a `_test`-suffixed
  database, refused otherwise before a connection opens. `uv run pytest -m
  live` makes a real OpenAI call and is skipped without `OPENAI_API_KEY` --
  like `postgres`, its tests also run as part of a bare `uv run pytest -q`
  whenever the resource they need happens to be present (a real key, a
  reachable container); neither marker excludes itself from the default run,
  it just skips itself gracefully when the thing it needs is not there.
