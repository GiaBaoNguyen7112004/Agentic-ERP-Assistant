"""The package entry point: ``agentic-erp-assistant``.

One subcommand today. ``serve`` starts the web layer -- FastAPI, uvicorn,
one worker (ADR 0004: the rate limiter, the flaky-tool counter, and the ERP
lock are all in-process state, and a second worker would give each of them
its own, unsynchronized copy).
"""

import argparse
import logging
import sys


def _serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    from agentic_erp_assistant.composition.settings import Settings, SettingsError

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Fail loudly before uvicorn even binds a socket: a misconfigured .env
    # is an operator problem best reported in one line at startup, not on
    # the first request a browser makes. AppResources.from_env() -- the
    # heavier check (a real API key, a non-empty Qdrant collection) -- runs
    # in create_app()'s lifespan and fails the same way, once uvicorn is
    # already reporting startup errors to the console.
    try:
        Settings.from_env().log_dev_toggles()
    except SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    uvicorn.run(
        "agentic_erp_assistant.web.app:create_app",
        factory=True,
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        workers=1,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-erp-assistant")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve", help="run the web layer (FastAPI + the built React app)"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--reload", action="store_true", help="restart on source changes (development only)"
    )

    arguments = parser.parse_args(argv)

    if arguments.command == "serve":
        raise SystemExit(_serve(arguments))

    parser.print_help()
    raise SystemExit(0 if arguments.command is None else 2)
