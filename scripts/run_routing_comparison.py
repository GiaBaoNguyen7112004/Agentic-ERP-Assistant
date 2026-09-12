"""Run the three planner contracts against the six fixed cases, for real.

    uv run python scripts/run_routing_comparison.py --repeats 5

Writes `evidence/routing/routing-comparison-<date>.json`, the number
gap-plan.md's ADR 0020 applies its selection rule to. Costs
`len(prompts) * 6 * repeats` real planner calls -- printed before anything is
sent, because this is exactly the kind of request the budget check itself
exists to let a caller decide about in advance.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_erp_assistant.composition.settings import DEFAULT_USERS_PATH, Settings
from agentic_erp_assistant.composition.users import UserDirectory
from agentic_erp_assistant.eval.routing import ComparisonReport, run_comparison
from agentic_erp_assistant.eval.routing_cases import DEFAULT_CASES
from agentic_erp_assistant.eval.routing_prompts import ALL_PROMPTS
from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.ports import LLMClientError
from agentic_erp_assistant.llm.telemetry import InMemoryTelemetry

logger = logging.getLogger("run_routing_comparison")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evidence" / "routing"

APPROX_INPUT_TOKENS_PER_CALL = 2_700
"""What the manual walkthrough's routed calls actually cost (docs/manual-test.md
§6, E4) -- printed as an estimate before spending real money, not measured
here."""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repeats", type=int, default=5, help="how many times to run each pair"
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="comma-separated prompt names to include (default: all three)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="where to write (default: dated path)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=4.0,
        help=(
            "seconds between calls, to stay under a tokens-per-minute limit "
            "(default 4.0: ~15 calls/min at ~2.7k tokens each, under a 30k "
            "TPM budget, which a live run of this script hit at --pause 0)"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def summarize(report: ComparisonReport) -> None:
    logger.info("")
    logger.info("%-16s %6s %6s %8s %10s %10s", "prompt", "n", "match", "rate", "halluc.", "cost_usd")
    for summary in report.summary():
        logger.info(
            "%-16s %6d %6d %8.1f%% %10d %10.4f",
            summary.prompt,
            summary.total,
            summary.matches,
            summary.match_rate * 100,
            summary.hallucinations,
            summary.cost_usd,
        )

    logger.info("")
    logger.info("route distribution, per (prompt, case):")
    prompt_names = dict.fromkeys(result.prompt for result in report.results)
    case_ids = dict.fromkeys(result.case_id for result in report.results)
    for prompt_name in prompt_names:
        for case_id in case_ids:
            distribution = report.route_distribution(prompt_name, case_id)
            logger.info("  %-16s %-6s %s", prompt_name, case_id, distribution)

    logger.info("")
    logger.info(
        "declarations: %d row(s), %.1f%% matched (scored cases only)",
        len(report.declarations),
        report.declaration_match_rate() * 100,
    )
    logger.info("%-8s %-8s %-30s %-30s %s", "case", "repeat", "declared", "expected", "matched")
    for row in report.declarations:
        logger.info(
            "%-8s %-8d %-30s %-30s %s",
            row.case_id,
            row.repeat,
            ",".join(row.declared_needs) if row.declared_needs else "(error)" if row.error else "(none)",
            ",".join(row.expected_needs) if row.expected_needs else "(unscored)",
            row.matched,
        )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(message)s",
    )

    prompts = ALL_PROMPTS
    if arguments.prompts:
        wanted = set(arguments.prompts.split(","))
        prompts = tuple(prompt for prompt in ALL_PROMPTS if prompt.name in wanted)
        missing = wanted - {prompt.name for prompt in prompts}
        if missing:
            logger.error("unknown prompt name(s): %s", ", ".join(sorted(missing)))
            return 2

    routing_calls = len(prompts) * len(DEFAULT_CASES) * arguments.repeats
    declaration_calls = len(DEFAULT_CASES) * arguments.repeats
    logger.info(
        "%d prompt(s) x %d case(s) x %d repeat(s) = %d routing call(s) + "
        "%d declaration call(s) (one per case x repeat, not per prompt) = "
        "%d call(s) total, ~%d input tokens each",
        len(prompts),
        len(DEFAULT_CASES),
        arguments.repeats,
        routing_calls,
        declaration_calls,
        routing_calls + declaration_calls,
        APPROX_INPUT_TOKENS_PER_CALL,
    )

    try:
        settings = Settings.from_env()
    except Exception as error:  # noqa: BLE001 - reported, not a traceback
        logger.error("%s: %s", type(error).__name__, error)
        return 3

    users = UserDirectory.load(DEFAULT_USERS_PATH)

    try:
        client = OpenAIChatClient(model=settings.model)
    except LLMClientError as error:
        logger.error("%s: %s", type(error).__name__, error)
        return 3

    def gateway_factory(contract: str) -> LLMGateway:
        return LLMGateway(
            client,
            context_window=settings.context_window,
            planner_contract=contract,
            telemetry=InMemoryTelemetry(),
        )

    try:
        report = run_comparison(
            prompts,
            DEFAULT_CASES,
            gateway_factory,
            users,
            repeats=arguments.repeats,
            pause_seconds=arguments.pause,
        )
    finally:
        client.close()

    summarize(report)

    output = arguments.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        output = DEFAULT_OUTPUT_DIR / f"routing-comparison-{stamp}.json"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit(),
        "model": settings.model,
        "repeats": arguments.repeats,
        "prompts": [prompt.name for prompt in prompts],
        **report.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger.info("")
    logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
