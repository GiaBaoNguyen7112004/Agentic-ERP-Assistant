"""Prove a pending approval survives a restart, against the real database.

The story in one script: a turn asks to create a risk, pauses on the human,
and the process "dies". A brand-new process -- a new connection, new stores,
a new orchestrator, nothing carried in a variable -- starts up, finds the
pending approval in Postgres, approves it, and the write happens: in the ERP
store, in the file on disk, and in the audit table. A second approval then
raises, because a decision settles once.

No LLM key is needed: the planner is a scripted model, the composer a canned
answer. What is real is everything the persistence work is for -- the engine,
the tool gateway, the approval policy, and Postgres.

    docker compose up -d postgres
    uv run python scripts/init_postgres.py
    uv run python scripts/demo_pause_across_restart.py

The evidence stays in the database afterwards, deliberately: the trace id is
printed, and the SELECTs at the end of the run are the queries to read it
with.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from agentic_erp_assistant.engine.orchestrator import (
    ApprovalAlreadySettled,
    RunOrchestrator,
)
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, MockErp
from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.persistence import (
    PostgresAuditLog,
    PostgresPauseStore,
    PostgresTraceStore,
    StoreConnectionError,
    connect,
)
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.tools.registry import build_default_registry

logger = logging.getLogger("demo_pause_across_restart")

SCOPES = frozenset(
    {
        "project.status.read",
        "project.sprint.read",
        "project.budget.read",
        "project.risk.read",
        "project.risk.write",
    }
)

SNIPPET = EvidenceSnippet(
    source_id="m2-status.md",
    locator="p.2",
    text="Finance module cutover slipped two weeks after the vendor delay.",
)


class ScriptedModel:
    """The planner's provider, scripted: a write, then an answer."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls = 0

    def decide(self, question, evidence=(), observations=(), *, tools=()):
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class FakeRetriever:
    def search(self, query: str, *, limit: int):
        return (SNIPPET,)


class FakeComposer:
    def answer(self, question: str, evidence):
        return GroundedAnswer(
            answer="Risk recorded.",
            citations=[Citation(source_id="m2-status.md", locator="p.2")],
            grounded=True,
            confidence=0.9,
        )


def build_orchestrator(
    url: str | None, erp_path: Path, *results: ToolCallResult
) -> tuple[RunOrchestrator, MockErp]:
    """A process's worth of construction: its own connection, its own stores,
    its own engine. Called twice below; nothing is shared between the calls
    except the database and the ERP file -- which is the demo."""
    connection = connect(url)
    erp = MockErp.load(erp_path)
    runtime = WorkflowRuntime(
        retriever=FakeRetriever(),
        tools=ToolGateway(
            registry=build_default_registry(erp),
            audit=PostgresAuditLog(connection),
        ),
        planner=Planner(ScriptedModel(*results)),
        composer=FakeComposer(),
        sleep=lambda seconds: None,
    )
    orchestrator = RunOrchestrator(
        runtime, PostgresTraceStore(connection), PostgresPauseStore(connection)
    )
    return orchestrator, erp


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=None,
        help="the Postgres conninfo to connect to (default: POSTGRES_URL or "
        "the docker-compose default)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    trace_id = f"demo-pause-{uuid4().hex[:8]}"
    erp_path = Path(tempfile.mkdtemp()) / "project.json"
    erp_path.write_text(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    risks_before = len(MockErp.load(erp_path).risks)

    # -- process 1: run the turn, pause, and "die" -------------------------
    try:
        orchestrator, _ = build_orchestrator(
            arguments.url,
            erp_path,
            ToolCallResult.from_tool_call(
                tool_name="create_risk",
                arguments={
                    "project_id": "atlas",
                    "title": "vendor slipped, demo",
                    "severity": "low",
                },
            ),
        )
    except StoreConnectionError as error:
        logger.error("%s", error)
        return 2

    state = AgentState(
        request="Record the vendor risk.",
        actor="demo",
        trace_id=trace_id,
        scopes=SCOPES,
    )
    final = orchestrator.handle(state)
    if final.route != "request_approval":
        logger.error("the demo turn did not pause: route=%r", final.route)
        return 3
    logger.info("run %s paused on %s -- the process now 'dies'", trace_id, final.tool_name)
    del orchestrator, final  # nothing is carried across the restart but the ids

    # -- process 2: a fresh process finds the pause and answers it ----------
    restarted, erp = build_orchestrator(
        arguments.url, erp_path, ToolCallResult.from_content("Risk recorded.")
    )
    pending = PostgresPauseStore(connect(arguments.url)).pending(trace_id)
    if pending is None:
        logger.error("no pending approval found for run %s after restart", trace_id)
        return 4
    logger.info("restart found the pending approval: %s", pending.tool_name)

    resumed = restarted.resume(trace_id, approved=True)
    logger.info("resumed to route=%r terminal=%s", resumed.route, resumed.terminal)

    # -- the evidence, read back the way an operator would ------------------
    reread = MockErp.load(erp_path)
    on_disk = any(risk.title == "vendor slipped, demo" for risk in reread.risks)
    in_store = any(risk.title == "vendor slipped, demo" for risk in erp.risks)
    logger.info(
        "the write reached the store (%s) and the file on disk (%s): "
        "%d risk(s) before, %d after",
        in_store,
        on_disk,
        risks_before,
        len(reread.risks),
    )

    connection = connect(arguments.url)
    events = connection.execute(
        "SELECT count(*) FROM trace_events WHERE trace_id = %s", (trace_id,)
    ).fetchone()[0]
    audit = connection.execute(
        "SELECT tool_name, approval FROM audit_rows WHERE trace_id = %s", (trace_id,)
    ).fetchall()
    decision = connection.execute(
        "SELECT status, decision FROM pauses WHERE trace_id = %s", (trace_id,)
    ).fetchone()
    logger.info(
        "evidence for run %s: %d trace event(s), audit row(s) %s, pause row %s",
        trace_id,
        events,
        audit,
        decision,
    )
    connection.close()

    # -- and a decision settles once -----------------------------------------
    try:
        restarted.resume(trace_id, approved=True)
    except ApprovalAlreadySettled:
        logger.info("second approval refused: a decision settles exactly once")
    else:
        logger.error("a second approval was accepted -- that must never happen")
        return 5

    if not (resumed.terminal and on_disk and in_store):
        logger.error("the demo did not prove what it exists to prove")
        return 6

    print(f"\ndemo complete. read the evidence with:")
    print(f"  SELECT * FROM runs WHERE trace_id = '{trace_id}';")
    print(f"  SELECT * FROM trace_events WHERE trace_id = '{trace_id}';")
    print(f"  SELECT * FROM audit_rows WHERE trace_id = '{trace_id}';")
    print(f"  SELECT * FROM pauses WHERE trace_id = '{trace_id}';")
    return 0


if __name__ == "__main__":
    sys.exit(main())