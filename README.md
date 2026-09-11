# Mini Agentic ERP Assistant

A browser chat assistant for project-delivery operations (sprints, budget,
risk) backed by an agent runtime written from scratch: no LangGraph, CrewAI,
or equivalent framework provides the graph engine, memory policy, tool
router, or retry logic. See `CLAUDE.md` for the hard constraints this project
is graded against, and `docs/adr/` for the reasoning behind every load-bearing
decision.

## Quick start

Requires Python (see `.python-version`) via [uv](https://docs.astral.sh/uv/),
Docker, and Node ≥ 20.

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, OPENAI_MODEL, OPENAI_CONTEXT_WINDOW,
# OPENAI_EMBEDDING_MODEL -- see the comments in .env.example

uv sync

docker compose up -d qdrant postgres
uv run python scripts/init_postgres.py
uv run python scripts/ingest_documents.py

npm --prefix ui install
npm --prefix ui run build

uv run agentic-erp-assistant serve
# open http://127.0.0.1:8000
```

Switch actor with the sidebar `<select>` — there is no login; the six
synthetic users in `data/users.json` are the dev-only cast, each bound to a
project and a set of entitlements. Ask a document question to see it stream
and cite a source; ask `create_risk`-shaped question ("record a risk that
...") as an actor holding `project.risk.write` to see the approval flow.

For a live-edit loop on the frontend: `npm --prefix ui run dev` (Vite on
`:5173`, `/api` proxied to `:8000`) alongside `uv run agentic-erp-assistant
serve`.

## Proving it from the terminal, no browser required

```bash
uv run python scripts/run_turn.py --actor priya "Why is milestone M2 late?"
uv run python scripts/run_turn.py --actor priya "Record a low risk on atlas: ..."
uv run python scripts/run_turn.py --actor priya --approve <trace_id>
```

## Testing

```bash
uv run python -m compileall -q src
uv run python -c "import agentic_erp_assistant"
uv run pytest -q                                    # the fake-backed suite
docker compose up -d postgres && uv run python scripts/init_postgres.py
uv run pytest -m postgres                           # the SQL adapters, for real

npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build
```

`docs/manual-test.md` is the scenario handbook: every case names the request,
the expected route and trace shape, and how to verify it in the database or
the UI. `docs/e2e-code-plan.md` is the phase-by-phase build log this codebase
was implemented from.

## Layout

See `CLAUDE.md`'s "Intended architecture" for the package-by-package
breakdown and the design rules each one follows.
