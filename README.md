# Mini Agentic ERP Assistant

A browser chat assistant for project-delivery operations — sprints, budget,
risk — backed by an agent runtime written from scratch. No LangGraph, CrewAI,
AutoGen, or equivalent framework provides the graph engine, state model,
memory policy, context builder, tool router, retry engine, or trace capture:
all of that is project code under `src/agentic_erp_assistant/`, and the only
third-party pieces are plain SDKs and infrastructure (an HTTP client, Qdrant,
Postgres, FastAPI, React).

This is a graded final project. Architecture, grounding, safety, and evidence
are what is assessed, so the repo is built to be *defended*: every
load-bearing decision has an ADR under [`docs/adr/`](docs/adr/README.md),
every request emits a persisted trace, and `CLAUDE.md` lists the hard
constraints the code is held to.

## What it does

An actor (a synthetic project member from `data/users.json`) types a question
in the browser. One turn then:

1. **Plans** — the model is asked, through function calling only (ADR 0006),
   what to do next: search documents, call an ERP tool, answer, clarify, or
   refuse. Before the graph runs, the planner also *declares* what a complete
   reply needs, and the graph holds the reply to that declaration (ADR 0021,
   0025).
2. **Grounds** — document questions go through hybrid retrieval: OpenAI
   embeddings in Qdrant fused with a from-scratch BM25 index by Reciprocal
   Rank Fusion. Access is enforced *before* ranking on both paths (ADR 0008),
   and every factual answer carries citations of the form
   `[document_id#locator]` — `§3.2`, `p.4`, `row R-1` — that resolve to the
   real chunk (ADR 0009).
3. **Acts** — ERP reads (`get_project_status`, `get_sprint_progress`,
   `get_budget_summary`, `list_risks`) run unattended within a rate limit;
   the one write (`create_risk`) is validated by the gateway first, then
   pauses the turn for an explicit human approve/deny (ADR 0016). The pause is
   persisted and survives a server restart.
4. **Remembers** — a verbatim window of recent turns reaches the prompt as
   history (ADR 0014); long-term memory is written only *after* the turn by a
   pure policy whose default is to refuse (ADR 0011, 0023), and memory can
   never masquerade as a citation (ADR 0012).
5. **Records** — node transitions, prompts, tool calls, retries, approvals,
   token and latency counts are streamed to the UI's trace panel over SSE and
   filed in Postgres. The filed trace is the record; the streamed text is a
   preview (ADR 0015).

The graph is a bounded cycle (ADR 0005): `think` hands control to
`retrieve_project_documents` or `call_tool` and back, and a turn that spends
its eight-step budget ends as `max_steps_exceeded` with the whole trace
attached rather than hanging.

## Quick start

### One command, from a fresh checkout

Requires Docker and an OpenAI key.

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, OPENAI_MODEL, OPENAI_CONTEXT_WINDOW,
# OPENAI_EMBEDDING_MODEL -- the comments in .env.example explain each

docker compose up --build
# open http://localhost:8000
```

This builds the app image (React UI, `uv.lock`-resolved venv, slim runtime —
ADR 0029), starts Qdrant and Postgres, applies the schema and ingests the
corpus through two idempotent one-shot services, then starts the app.

### The dev loop

Requires Python (see `.python-version`, currently 3.14) via
[uv](https://docs.astral.sh/uv/), Docker for the two stores, and Node ≥ 20.

```bash
cp .env.example .env            # and fill in the four OPENAI_* values
uv sync

docker compose up -d qdrant postgres
uv run python scripts/init_postgres.py       # nine tables
uv run python scripts/ingest_documents.py    # embed the corpus; re-runs are free

npm --prefix ui install && npm --prefix ui run build
uv run agentic-erp-assistant serve           # http://127.0.0.1:8000
```

For live-editing the frontend, run `npm --prefix ui run dev` (Vite on `:5173`,
`/api` proxied to `:8000`) alongside the server.

### Using it

There is no login. Switch actor with the sidebar `<select>`; the six synthetic
users in `data/users.json` are each bound to a project and a set of
entitlements:

| actor        | role                     | project | notable scopes                                      |
|--------------|--------------------------|---------|-----------------------------------------------------|
| `priya`      | Delivery lead            | atlas   | all reads, `project.risk.write`, `approvals.decide` |
| `wei`        | Finance business partner | atlas   | `project.docs.finance.read` (the budget PDF)        |
| `tomas`      | Warehouse engineer       | atlas   | docs, status, sprint reads only                     |
| `sponsor`    | Approver                 | atlas   | docs, status, `approvals.decide`                    |
| `orion.lead` | Delivery lead (Orion)    | orion   | the other project — sees none of Atlas              |
| `guest`      | No entitlements          | atlas   | nothing                                             |

Things worth trying:

- As `priya`: *"Why is milestone M2 late?"* — watch it search, stream, and end
  with clickable citations.
- As `tomas`: *"What is the Q3 budget variance?"* — the budget summary is
  finance-only, so the turn refuses — the model is never even shown that
  document exists (ADR 0026).
- As `priya`: *"Record a low risk on atlas: ..."* — the turn pauses on an
  approval card; approve or deny it, and find the decision in the trace.
- As `guest`: anything — every tool and document is out of scope.

`docs/manual-test.md` is the full scenario handbook: each case names the
actor, the request, the expected route and trace shape, and the SQL that
proves it in the evidence store.

## Proving it from the terminal

`scripts/run_turn.py` drives the same composition root the web layer uses, so
the whole system can be exercised without a browser:

```bash
uv run python scripts/run_turn.py --actor priya "Why is milestone M2 late?"
uv run python scripts/run_turn.py --actor priya "Record a low risk on atlas: ..."
uv run python scripts/run_turn.py --actor priya --approve <trace_id>   # or --deny

uv run python scripts/demo_memory_session.py        # two turns, the window, what was kept
uv run python scripts/demo_pause_across_restart.py  # a pause survives a restart
```

## Architecture

```
browser (ui/, React + TS)
   │  SSE: typed events (web/protocol.py ⇄ ui/src/protocol.ts, drift-tested)
   ▼
web/          FastAPI, thin async shell; a turn runs on a worker thread
   ▼
composition/  .env -> typed Settings; process-wide clients built once;
              one turn's ports assembled per request
   ▼
engine/       WorkflowRuntime: apply nodes until terminal, paused, or out of budget
   ├── reasoning/   planner (function calling), reply contract, failure classification
   ├── context/     what enters the prompt and why: budget, history, memory, catalogue
   ├── llm/         provider port + gateway (retry, tokens, cost); adapters/ names OpenAI
   ├── rag/         manifest, chunking, Qdrant + BM25, RRF fusion, access, citations
   ├── tools/       typed specs, registry, gateway (scope, rate limit, approval, retry)
   │     └── erp/   mock ERP over data/erp/project.json, bound to the actor's project
   ├── memory/      write gate (pure policy), session window, summary, intents
   └── trace/       structured records, run telemetry
   ▼
persistence/  Postgres adapters behind the trace/memory ports + EvidenceQueries
state/        AgentState and every typed value that crosses a boundary
eval/         retrieval and routing harnesses; scripts/ write evidence/
```

Design rules the code follows, and the defense rests on:

- **Ports are Protocols, adapters are swappable.** LLM provider, retriever,
  ERP backend, trace and memory stores each sit behind an interface with a
  fake used in tests. The provider SDK is never imported outside
  `llm/adapters/`.
- **State is typed and explicit.** Nodes take `AgentState` and return a new
  one; routes and failure modes are closed `Literal` sets, not prose.
- **Tools declare typed schemas and `mutating: bool`**, and the router forces
  approval from that flag. A call that fails validation never reaches a
  handler; a write gets one attempt, never a retry (a retried write is how
  one approved risk becomes three).
- **Retry, rate limits, and the step budget are runtime policy**, budgeted
  and traced, not `try/except` sprinkled into tools.
- **Every request emits a trace**, and the trace is as much the deliverable
  as the answer.

## Configuration

Everything comes from `.env` (see `.env.example`, which documents every
variable). The model is deliberately not chosen by the repo: `OPENAI_MODEL`,
`OPENAI_CONTEXT_WINDOW` and `OPENAI_EMBEDDING_MODEL` have no defaults, and
whatever chat model is set needs a reviewed row in `llm/pricing.py` or cost
estimation raises rather than guessing.

The `DEV_*` toggles (`DEV_TOOL_RATE_LIMIT`, `DEV_FLAKY_STATUS`,
`DEV_MAX_STEPS`, `DEV_HISTORY_TURN_LIMIT`, `DEV_TRACE_MODEL_IO`) exist only so
a browser session can reach paths — rate-limit refusal, retry-then-succeed,
step-budget exhaustion, history eviction, live prompt inspection — that a
scripted test reaches with a fake clock. Leave them blank for a normal run.

## Testing

```bash
uv run python -m compileall -q src
uv run python -c "import agentic_erp_assistant"
uv run pytest -q                       # the fake-backed suite

docker compose up -d postgres
uv run python scripts/init_postgres.py --test
uv run pytest -m postgres              # the SQL adapters, against a _test database only (ADR 0018)
uv run pytest -m live                  # one real OpenAI call; skipped without OPENAI_API_KEY

npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build
```

Both markers also run as part of a bare `uv run pytest -q` whenever the
resource they need is present, and skip themselves gracefully otherwise.
`tests/web/test_protocol_drift.py` fails the Python suite the day
`web/protocol.py` and `ui/src/protocol.ts` disagree.

## Evidence

Numbers in the repo are produced by scripts, never typed by hand:

```bash
uv run python scripts/run_retrieval_evaluation.py   # evidence/rag/retrieval-report.json
uv run python scripts/run_routing_comparison.py     # evidence/routing/routing-comparison-<date>.json
```

- `evidence/rag/` — hit rate, latency, and the cosine-similarity gap the
  retrieval threshold sits in, over the golden cases in `eval/golden_cases.py`.
- `evidence/routing/` — three planner contracts against six cases (ADR 0020):
  match rate, hallucination count, cost, every row, plus the reply-contract
  declaration match rate per case (ADR 0021). The production contract is
  whichever ADR 0020 names; no candidate has beaten it.

## Data

Everything under `data/` is small, synthetic, and readable in review — one
fictional programme (Atlas, an ERP rollout) plus a second project (Orion) that
exists to prove project isolation.

- `data/documents/` — the corpus: Markdown, HTML, CSV, and a rendered PDF, with
  `manifest.json` as the single control plane for access (`project_code` +
  one `required_scope` per document). Adding a document is one file plus one
  row; no code changes.
- `data/erp/project.json` — what the mock ERP serves. A test fixture guards
  that this and the documents cannot drift apart.
- `data/users.json` — the dev-only actor directory.

## Where to read next

- [`docs/adr/`](docs/adr/README.md) — 29 decision records; start with 0005
  (the graph is a bounded cycle), 0006 (function calling is the only decision
  channel), 0011 (memory is written by a pure policy), 0016 (a write is put to
  a human only after the gateway agrees it could run), and 0021 (the planner
  declares what a reply needs).
- [`docs/manual-test.md`](docs/manual-test.md) — the browser scenario handbook.
- [`CLAUDE.md`](CLAUDE.md) — the hard constraints and the module-by-module
  intent, kept current as decisions settle.
