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
uv run agentic-erp-assistant # run the CLI entry point
uv run pytest                # tests (once pytest is a dev dependency)
uv add <pkg>                 # add a runtime dependency
uv add --dev <pkg>           # add a dev dependency
```

Never edit `[project.dependencies]` by hand — use `uv add` so the lockfile stays in sync.

## Intended architecture

The package is a scaffold today. Build toward these boundaries; one concern per module,
and keep the dependency direction pointing inward (runtime never imports the web layer).

```
src/agentic_erp_assistant/
  runtime/     graph engine, node contracts, typed state, transitions, retry
  state/       the reasoning state model (typed, serializable, versioned)
  memory/      short-term + adaptive long-term memory, promotion/eviction policy
  context/     context construction: what gets into the prompt and why
  llm/         provider port (protocol), contracts, prompts, tokens, retry,
               tool specs, cost telemetry, and the gateway that orders them
  llm/adapters/  vendor adapters (openai_chat.py); the only place a provider is named
  rag/         ingestion, chunking, index, retrieval, citation objects
  tools/       MCP-style tool boundary: typed schemas, router, approval gating
  erp/         optional ERP provider plugin over mock project data
  guardrails/  input/output checks, refusal + escalation paths
  trace/       structured trace records, run store, export for evidence
  eval/        offline eval harness, datasets, scored runs
  web/         HTTP/SSE surface + the browser chat UI (accessible, responsive)
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
  record the reasoning in the commit message or a short note under `docs/decisions/` —
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

Frontend — the exact command depends on the web stack (see Open decisions); use whichever
applies and record the real command here once the stack lands:

```bash
npm run build          # or: npx tsc --noEmit   (bundled/TypeScript UI)
```

If the UI stays dependency-free browser JS with no build step, the check is: start the
server, load the chat page, and confirm the browser console is free of errors and a
message round-trips. A UI that was never loaded has not been verified.

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
- `<area>` matches a module directory (`runtime:`, `rag:`, `tools:`, `web:`, `eval:`).
- Never commit with failing checks, and never use `--no-verify`.
- Work on `dev` or a feature branch, never directly on `main`.
- Standing authorization: committing at these checkpoints is expected and does not need
  to be asked about each time. Pushing, force-pushing, opening PRs, and rewriting history
  still require an explicit request.

## Open decisions

Not yet chosen; ask before assuming, and update this file once settled.

- Web layer: no HTTP framework or JS tooling is present yet. Default suggestion is a
  Python framework serving SSE plus a dependency-free browser UI, matching the
  Python-only repo — confirm before scaffolding.
- ~~LLM provider~~ — settled: OpenAI Chat Completions, called with plain `httpx` in
  `llm/adapters/openai_chat.py` (no `openai` SDK anywhere). The model itself is **not** chosen by the
  repo: `OPENAI_MODEL` comes from `.env` with no default, and whatever model is set there
  also needs a reviewed row in `llm/pricing.py`. Routing policy behind the port is still
  open.
- Retrieval backend (embedding store vs. lexical vs. hybrid) and citation format.
- Trace persistence (files vs. SQLite) and eval report format.
