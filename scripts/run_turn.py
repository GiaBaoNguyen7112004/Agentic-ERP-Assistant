"""One real turn from the terminal, with the trace beside it.

Proves the composition root wires to a real answer before any web layer
exists: a real actor from data/users.json, real retrieval, a real model call,
a real Postgres connection, streamed tokens to stdout as they arrive, and the
trace read back afterward through the same EvidenceQueries a screen will use.

    uv run python scripts/run_turn.py --actor priya --session s1 \\
        "Why is milestone M2 late?"

A paused write is resumed the same way an approver would settle it -- as the
original requester, with the approver's identity recorded separately:

    uv run python scripts/run_turn.py --actor priya --approve run-<hex>
    uv run python scripts/run_turn.py --actor sponsor --deny run-<hex>

Requires ``docker compose up -d qdrant postgres``, a completed
``scripts/ingest_documents.py``, and a filled-in ``.env`` (see
``.env.example``).
"""

import argparse
import logging
import sys
import uuid

from agentic_erp_assistant.composition.resources import AppResources, ResourcesError
from agentic_erp_assistant.composition.settings import SettingsError
from agentic_erp_assistant.composition.turn import build_turn, initial_state
from agentic_erp_assistant.composition.users import UnknownUser
from agentic_erp_assistant.engine.orchestrator import ApprovalAlreadySettled
from agentic_erp_assistant.engine.workflow import is_paused
from agentic_erp_assistant.llm.ports import LLMClientError
from agentic_erp_assistant.persistence.connection import StoreConnectionError
from agentic_erp_assistant.persistence.postgres_pause import PostgresPauseStore

logger = logging.getLogger("run_turn")


class ConsoleStream:
    """Prints reply tokens as they arrive; everything else is printed once,
    after the run ends, from the trace that was actually filed -- the same
    evidence a screen would read, not a live narration of internals."""

    def __init__(self) -> None:
        self._started = False

    def step(self, state) -> None:  # noqa: ARG002 - the observer's contract
        pass

    def trace_event(self, event) -> None:  # noqa: ARG002
        pass

    def delta(self, text: str) -> None:
        if not self._started:
            print("assistant> ", end="", flush=True)
            self._started = True
        print(text, end="", flush=True)

    def reset(self) -> None:
        if self._started:
            print(" [retrying...]", flush=True)
        self._started = False


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", help="the question to ask")
    parser.add_argument(
        "--actor", required=True, help="who to act as -- a name from data/users.json"
    )
    parser.add_argument(
        "--session", default=None, help="session id; generated if omitted"
    )
    parser.add_argument("--approve", metavar="TRACE_ID", help="approve a pending pause")
    parser.add_argument("--deny", metavar="TRACE_ID", help="deny a pending pause")
    return parser.parse_args(argv)


def _print_events(events) -> None:
    print("\n--- trace events ---")
    for event in events:
        print(f"  {event.node:22s} {event.kind:20s} {event.detail}")


def _print_model_calls(queries, trace_id: str) -> None:
    records, totals = queries.model_calls(trace_id)
    print("\n--- model calls ---")
    for record in records:
        delta = ""
        if record.input_tokens:
            drift = record.estimated_input_tokens - record.input_tokens
            pct = drift / record.input_tokens * 100
            delta = f" est={record.estimated_input_tokens} drift={drift:+d} ({pct:+.1f}%)"
        print(
            f"  {record.model} {record.outcome} attempts={record.attempts} "
            f"in={record.input_tokens} out={record.output_tokens} "
            f"cost={record.cost_usd}{delta}"
        )
    print(
        f"  totals: {totals.count} call(s), {totals.input_tokens} in / "
        f"{totals.output_tokens} out tokens, cost_usd={totals.cost_usd} "
        f"({totals.unpriced} unpriced)"
    )


def _print_outcome(final, queries) -> None:
    print(f"\n\nroute={final.route} terminal={final.terminal} approval={final.approval}")
    if final.failure != "none":
        print(f"failure={final.failure} detail={final.error_detail}")
    if final.response:
        print(f"reply: {final.response}")
    if is_paused(final):
        print(
            f"\nPAUSED on {final.tool_name}. Resume with:\n"
            f"  uv run python scripts/run_turn.py --actor <approver> "
            f"--approve {final.trace_id}\n"
            f"  uv run python scripts/run_turn.py --actor <approver> "
            f"--deny {final.trace_id}"
        )
    _print_events(queries.events(final.trace_id))
    _print_model_calls(queries, final.trace_id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments(argv)

    try:
        resources = AppResources.from_env()
    except (SettingsError, ResourcesError, LLMClientError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        connection = resources.connect()
    except StoreConnectionError as error:
        print(f"error: {error}", file=sys.stderr)
        resources.close()
        return 2

    stream = ConsoleStream()
    try:
        if arguments.approve or arguments.deny:
            trace_id = arguments.approve or arguments.deny
            approved = arguments.approve is not None

            try:
                approver = resources.users.get(arguments.actor)
            except UnknownUser:
                print(f"error: unknown actor {arguments.actor!r}", file=sys.stderr)
                return 2
            if not approver.can_approve:
                print(
                    f"error: {arguments.actor!r} does not hold approvals.decide",
                    file=sys.stderr,
                )
                return 1

            pending = PostgresPauseStore(connection).pending(trace_id)
            if pending is None:
                print(f"error: no pending pause for {trace_id!r}", file=sys.stderr)
                return 1
            # The turn resumes as the original requester -- their scopes and
            # project are what the resumed call is checked against -- with
            # the approver's identity recorded separately, in decided_by.
            requester = resources.users.get(pending.actor)

            turn = build_turn(
                resources,
                user=requester,
                session_id=pending.session_id or "",
                trace_id=trace_id,
                connection=connection,
                stream=stream,
            )
            try:
                final = turn.orchestrator.resume(
                    trace_id, approved=approved, decided_by=approver.actor
                )
            except ApprovalAlreadySettled as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
        else:
            if not arguments.message:
                print(
                    "error: a message is required unless --approve/--deny is given",
                    file=sys.stderr,
                )
                return 2

            try:
                user = resources.users.get(arguments.actor)
            except UnknownUser:
                print(f"error: unknown actor {arguments.actor!r}", file=sys.stderr)
                return 2

            session_id = arguments.session or f"sess-{uuid.uuid4().hex[:12]}"
            trace_id = f"run-{uuid.uuid4().hex}"

            turn = build_turn(
                resources,
                user=user,
                session_id=session_id,
                trace_id=trace_id,
                connection=connection,
                stream=stream,
            )
            state = initial_state(
                user, session_id=session_id, trace_id=trace_id, message=arguments.message
            )
            final = turn.orchestrator.handle(state)

        _print_outcome(final, turn.queries)
        return 0
    except LLMClientError as error:
        print(f"\nerror: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
