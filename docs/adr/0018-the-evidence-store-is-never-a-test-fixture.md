# 0018 — The evidence store is never a test fixture

**Status:** Accepted (2026-09-12)

## Context

CLAUDE.md's constraint 5 says every request emits a trace, and that traces
are the audit evidence — "as important as the feature itself." The manual
walkthrough of `docs/manual-test.md` (commit `638adbc`) is that evidence for
this project: sixty-five scenarios run against the real model, the real
retriever, and the real Postgres, with the trace of each one read back and
checked against the handbook's own SQL.

Its own finishing check erased it. `CLAUDE.md`'s "Finishing a change" section
requires `uv run pytest -q` before a task is reported done, run with
`docker compose up -d postgres` up (the walkthrough needed it for the live
turns anyway). Five modules under `tests/persistence/` — `test_schema.py`,
`test_postgres_adapters.py`, `test_postgres_conversation.py`,
`test_postgres_memory.py`, `test_postgres_queries.py` — each opened their own
`database` fixture with `connect()` and no URL, which resolves to
`POSTGRES_URL`, falling back to the compose default when unset. That is not a
second, private database for the tests; it is `postgresql://agentic_erp:
agentic_erp@localhost:5432/agentic_erp`, the exact URL the running server
writes every trace to. `store_connection`'s per-test setup then issued
`TRUNCATE runs, trace_events, audit_rows, model_calls, pauses, memories,
intents, memory_audit, session_turns CASCADE` — sixty-six times, once per
postgres-marked test. Running the mandated finishing check on a repo with the
dev container up leaves one row (`test_schema.py`'s own `run-schema-test`)
where sixty-five recorded turns had been.

Nothing about this is a Postgres adapter bug: every one of the tests it
wiped is correct on its own terms, proving exactly what it says it proves.
The mistake is in what database it proved it against.

## Decision

**A URL is either the dev store or the test one, and a name check is the
only thing standing between them, checked before a connection is even
opened.** `persistence/connection.py` gains `POSTGRES_TEST_URL` /
`test_url_from_environment()`, parallel to `POSTGRES_URL` /
`url_from_environment()` in every respect but which variable it reads and
which compose default it falls back to
(`…agentic_erp_test` against `…agentic_erp`). `assert_test_database(url)`
extracts the database segment and raises `StoreConfigurationError` — a new
error, deliberately not `StoreConnectionError` — unless it ends in `_test`.
The distinction the two errors carry is the one that matters here:
`StoreConnectionError` means "unreachable", a deployment fact nothing in the
caller can fix; `StoreConfigurationError` means "reachable, and wrong to be
reaching" — the one mistake a connection attempt would never itself surface,
because the connection succeeds and the truncation is what tells you too
late.

**One fixture, in `tests/persistence/conftest.py`, replaces the five.** It
calls `test_url_from_environment()`, then `assert_test_database` —
uncaught, so a misconfigured URL fails the run rather than skipping it — then
`connect()`, which *is* caught: a missing test database is not a
configuration mistake, it is a setup step not yet run, and the message names
the command that runs it. Every module keeps its own `store_connection`
fixture (the truncate-then-yield each test wants); only the connection
underneath it is now shared and guarded.

**`scripts/init_postgres.py --test`** creates the database and applies the
schema, so "point tests somewhere safe" is a command, not an instruction to
create a database by hand correctly. `CREATE DATABASE` cannot run inside a
transaction block, which is exactly what an autocommit connection (already
`connect()`'s only mode) is for; the script connects to the *dev* database
first to issue it, since the same compose user owns both, then reconnects to
the new one to apply the schema — the identical two-step `init_postgres.py`
without `--test` already does, once with one database instead of two.

## Consequences

The evidence store is now something a test suite runs against, not something
it can reach at all without deliberately asking for the other one. A future
sixth `tests/persistence/` module inherits the shared fixture by using
`database` as a parameter name — there is no longer a `connect()`-with-no-URL
pattern to copy by accident.

Re-verified live, both directions: with the dev container up and one real run
recorded, `uv run pytest -q` (1628 tests) and `uv run pytest -m postgres` (66
tests) both pass, and `SELECT count(*) FROM runs` on the dev database is
identical before and after. Pointing `POSTGRES_TEST_URL` at the dev database
directly — the exact mistake this ADR exists to catch — turns every one of
those 66 tests into an immediate `StoreConfigurationError` at fixture setup,
and the dev database's row count is again unchanged afterward.

`docs/manual-test.md`'s sixty-five traces from `638adbc` are not recovered by
this change — they were already gone before it landed, and Postgres does not
undelete. What this closes is the recurrence: the walkthrough's findings
(gap-plan.md, Phases J–L) can now be re-verified live without a routine
`pytest -q` erasing the proof a second time.

## Alternatives considered

**Transactional tests, rolled back instead of truncated.** Rejected: the
adapters under test commit as part of what they are (autocommit is the whole
point of `connect()`, per its own docstring), and `TRUNCATE … CASCADE` is
what gives each test a store that looks freshly initialized rather than
polluted by whatever ran before it. Wrapping every test in an uncommitted
transaction would mean testing a code path — commit — that never actually
runs.

**A `--keep-evidence` flag, or a guard the operator has to remember to pass.**
Rejected for the same reason ADR 0016 rejected a web-layer-only pre-check: a
guard that depends on every future invocation remembering to ask for safety
is not safety, it is a habit, and this ADR exists because the habit already
failed once, silently, on the project's own mandated finishing check.

**Skip the postgres-marked suite instead of isolating it**, e.g. only run
`pytest -m postgres` deliberately, never as part of `pytest -q`. Rejected:
`-m postgres` tests are already opt-in by marker and CLAUDE.md's own
Commands block lists `uv run pytest -m postgres` as something to run "for
real" — the problem was never that they ran, it was where they pointed.

**A single `POSTGRES_URL` with a runtime flag distinguishing test runs from
server runs.** Rejected: one variable serving two purposes is exactly the
failure mode `QDRANT_MEMORY_COLLECTION`'s own doc comment (`.env.example`)
already warns against for a different pair of concerns — a test run and a
server run needing different values for the same name means one of them
gets the wrong value the moment somebody sets the variable for the other.
