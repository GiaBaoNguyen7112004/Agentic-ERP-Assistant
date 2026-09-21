# 0029 — The app ships as a built image, and one command brings up the whole system

**Status:** Accepted (2026-09-15)

## Context

Until now `docker-compose.yml` defined only the two stores, qdrant and
postgres, and the application was always run with `uv` on the host. That was
the honest arrangement while the web layer was still landing — the compose
file said as much: *"When the web layer lands, an app service is a decision
to record then."*

It landed (ADR 0015). What is left is the question a fresh checkout poses:
a reviewer — or the grading run — should be able to bring the whole system
up without reading this repository first. Today that means: install uv and
Python 3.14 and Node, `uv sync`, `npm --prefix ui install && npm run build`,
`docker compose up -d qdrant postgres`, wait for both healthchecks, apply
the schema, ingest the corpus, then `serve`. Six steps in a specific order,
each a place to stall.

The constraints the decision has to respect:

- **The `/app` layout is load-bearing.** Data paths (`data/users.json`,
  `erp/project.json`, `documents/manifest.json`) are computed as
  `Path(__file__).resolve().parents[3] / "data"` from inside the package
  (composition/settings.py, erp/mock.py, rag/manifest.py). Any container
  layout must put the editable install's `__file__` exactly three levels
  under its data directory, or startup breaks.
- **No agent framework provides the core** (constraint 2), so the image is
  just the project's own venv — no framework base image, no framework CLI.
- **Secrets come from the environment only**, never the repo — the image
  must not bake in `.env`.
- **workers=1** (ADR 0004): the rate limiter, flaky-tool counter and ERP
  lock are in-process state, so no horizontal scaling is offered, even
  accidentally.
- The dev loop stays as it is: during development the thing changed every
  minute must not be behind an image rebuild.

## Decision

Three stages in the repo-root `Dockerfile`, and three services in compose:

**Stage 1 — the React UI** on `node:22-alpine`: `npm ci`, `npm run build`.
Vite's configured output path (`ui/vite.config.ts` writes to
`../src/agentic_erp_assistant/web/static` relative to `ui/`) puts the built
app at a path copied straight into the runtime stage. The image builds its
own UI rather than mounting or copying the host's: `web/static` is
git-ignored local state, and `ui/node_modules` is hundreds of MB of context
bloat — `.dockerignore` excludes both, so the node stage, not the checkout,
is the source of truth in the image.

**Stage 2 — the venv** on `python:3.14-slim`, resolved by `uv sync --frozen
--no-dev` from `uv.lock`, third-party deps first (so the layer is cached
until the lock changes) and the project itself second. `uv` installs the
project *editable*, because the lock pins it `source = { editable = "." }` —
which is exactly what keeps `__file__` at `/app/src/agentic_erp_assistant/`
and the `parents[3]`-relative data paths working. The venv and the runtime
stage use the same base image, so the interpreter symlinks inside the venv
resolve identically after the copy.

**Stage 3 — runtime** on `python:3.14-slim`: non-root `appuser`, the venv,
`src/`, `data/`, `scripts/`, and the built UI. The command is the ordinary
entry point, `agentic-erp-assistant serve --host 0.0.0.0 --port 8000` —
`--host 0.0.0.0` is required because the entry point defaults to
`127.0.0.1`, which is unreachable from outside the container; `--port` is
stated even though it is the default, so the image's command and its
`EXPOSE`/port mapping cannot drift apart silently. The healthcheck probes
`/api/health`, whose 200 proves the uvicorn lifespan finished — the lexical
index was built from Qdrant and Postgres connected — so it is a readiness
probe, not a port check, and it is answered with the venv's own `python`
because the slim image carries no `curl`.

**Compose ordering.** Two one-shot services, `postgres-init` and `ingest`,
share the app image (the scripts and the venv they run against already ship
in it — no second image to keep in step). `postgres-init` applies the
schema once postgres is healthy; `ingest` waits for qdrant *and* for
`postgres-init`, then embeds the corpus; the `app` service waits for both
stores' healthchecks *and* for both one-shots'
`service_completed_successfully`. Both one-shots are idempotent by
construction — the schema is `CREATE TABLE IF NOT EXISTS` in one
transaction, and ingest skips unchanged documents by content hash — so
every later `docker compose up` re-runs them as cheap no-ops rather than
needing a flag to skip them.

**Environment.** The one-shots and the app get their provider settings
(`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_CONTEXT_WINDOW`,
`OPENAI_EMBEDDING_MODEL`) from `.env` via `env_file` with
`required: false`, so a checkout without a key still brings the two stores
up. The in-network addresses — `http://qdrant:6333` and the postgres URL —
are set in `environment:` deliberately, because it overrides `env_file`:
a URL pointed at the host's published `localhost` port is unreachable from
inside the compose network's DNS, and silently serving a container pointed
at nothing would be worse than the explicit override.

## Alternatives rejected

**Host `uv` against published ports, forever** (the status quo): no rebuild,
but bring-up stays a six-step runbook, and the deliverable is only provably
runnable on machines set up like the author's.

**One container running everything** — app, qdrant, postgres in one image:
violates what the stores' separate lifecycle is for. Evidence and vectors
must survive the app process being replaced (the named volumes already
record that promise); one container bundling them means replacing the app
restarts the evidence store.

**Synchronizing at startup from inside the app process** — run
`init_postgres` and ingest in the server's lifespan: schema application and
a spend-money embedding run become things a server restart silently does.
The whole point of the init script's docstring is that an init run is
something an operator *sees*. A one-shot service is that same deliberate
act, just ordered by compose instead of by hand.

**A separate dev image or a volume-mounted source tree**: the dev loop
already has the right shape — `uv run` on the host against the published
ports, no image between the edit and the run. The dev commands stay in the
compose header comments unchanged; `docker compose up --build` is the
fresh-checkout path, not the development one.

**Installing `curl` for the healthcheck**: extra image surface for
something the venv's `python` already does; `urllib` in a `python -c` is
the whole probe.

## Consequences

- A fresh checkout with Docker, uv and Node is *not* required — Docker
  alone suffices: `docker compose up --build`, then
  `http://localhost:8000`. The build runs `npm ci` and `uv sync --frozen`
  inside the image, so the host needs no Node and no Python at all.
- `.env` must still exist with `OPENAI_MODEL`, `OPENAI_CONTEXT_WINDOW` and
  `OPENAI_EMBEDDING_MODEL` filled in for a *useful* bring-up — the app
  without a model key serves its healthcheck but cannot answer. The
  `.env.example` contract is unchanged and the compose file adds no default
  for any of them.
- Changing `pyproject.toml`'s `readme` (README.md), moving `src/` or
  `data/`, or changing vite's output path breaks the image in ways the
  Dockerfile comments now pin; the Dockerfile is the record of the
  `/app` layout being load-bearing, with ADR 0029 as the why.
- `uv:latest` (stage 2's copied binary) is the one un-pinned tag in the
  build; it is a builder-stage tool whose output is the lockfile-resolved
  venv, so a drift there produces a resolution that either matches
  `uv.lock --frozen` or fails the build loudly — it never quietly changes
  what runs. Pinned on the same day something makes the pin cheap to
  maintain.
- The one-shots share the `agentic-erp-assistant:latest` tag, so the three
  `build:` stanzas are the same context three times; BuildKit's cache makes
  the repeats cheap, and the tag is the dedup that matters at run time.