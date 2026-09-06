"""Ingest the project documents into Qdrant.

The only script in this project that spends money, so it is written to make that
visible before it happens: ``--dry-run`` loads the corpus, works out which
documents have actually changed, chunks them and reports what embedding them
would cost, without making a single request.

    docker compose up -d qdrant
    uv run python scripts/ingest_documents.py --dry-run
    uv run python scripts/ingest_documents.py

Re-running it is safe and cheap. Documents whose content has not changed are
skipped, so a second run immediately after the first makes no embeddings request
at all.
"""

import argparse
import logging
import sys
from pathlib import Path

from agentic_erp_assistant.llm.ports import LLMClientError
from agentic_erp_assistant.llm.tokenizer import count_tokens
from agentic_erp_assistant.rag.chunking import (
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    chunk_documents,
)
from agentic_erp_assistant.rag.embeddings import OpenAIEmbeddingsClient
from agentic_erp_assistant.rag.ingest import DEFAULT_BATCH_SIZE, ingest
from agentic_erp_assistant.rag.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestEntryError,
    changed_documents,
    load_documents,
)
from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex, VectorIndexError

logger = logging.getLogger("ingest_documents")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="the corpus manifest (default: the repo corpus)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="override QDRANT_COLLECTION for this run",
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=CHUNK_TOKENS, help="chunk budget"
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=OVERLAP_TOKENS, help="chunk overlap"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="chunks per embeddings request",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be embedded, and make no request",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def dry_run(arguments: argparse.Namespace) -> int:
    """Say what a real run would embed, and what it would cost, spending nothing.

    Still talks to Qdrant, because "what has changed" is a question only the
    index can answer. It never constructs an embeddings client, so it runs with
    no API key at all -- which is what makes it usable as a check before anyone
    has one.
    """
    documents = load_documents(arguments.manifest)
    index = QdrantVectorIndex.from_env(collection=arguments.collection)
    try:
        stored = index.document_hashes()
    finally:
        index.close()

    changed = changed_documents(documents, stored)
    logger.info(
        "%d document(s) in the manifest, %d already indexed, %d to embed",
        len(documents),
        len(documents) - len(changed),
        len(changed),
    )
    for document in changed:
        logger.info("  would embed %s", document.document_id)

    if not changed:
        return 0

    # The chunk budget is in tokens of the embedding model, and a dry run has no
    # client to ask for its name. The corpus tokenizer falls back cleanly for an
    # unknown model, so an approximate count here is honest as long as it is
    # labelled approximate.
    model = "text-embedding-3-small"
    chunks = chunk_documents(
        changed,
        model=model,
        chunk_tokens=arguments.chunk_tokens,
        overlap_tokens=arguments.overlap_tokens,
    )
    tokens = sum(count_tokens(chunk.text, model=model) for chunk in chunks)
    requests = -(-len(chunks) // arguments.batch_size)
    logger.info(
        "%d chunk(s), about %d token(s), in %d request(s) -- estimated with the "
        "%s tokenizer, not the configured one",
        len(chunks),
        tokens,
        requests,
        model,
    )
    return 0


def live_run(arguments: argparse.Namespace) -> int:
    """Embed and store for real."""
    documents = load_documents(arguments.manifest)
    index = QdrantVectorIndex.from_env(collection=arguments.collection)
    client = OpenAIEmbeddingsClient()

    try:
        report = ingest(
            documents,
            vector_index=index,
            embeddings=client,
            chunk_tokens=arguments.chunk_tokens,
            overlap_tokens=arguments.overlap_tokens,
            batch_size=arguments.batch_size,
        )
    finally:
        client.close()
        index.close()

    logger.info(
        "done: %d of %d document(s) embedded (%d skipped, %d removed), "
        "%d chunk(s) in %d request(s), %d prompt token(s), %d dimensions, model %s",
        report.documents_embedded,
        report.documents_seen,
        report.skipped,
        report.documents_removed,
        report.chunks_embedded,
        report.requests,
        report.prompt_tokens,
        report.dimensions,
        report.model,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return dry_run(arguments) if arguments.dry_run else live_run(arguments)
    except ManifestEntryError as error:
        logger.error("the corpus is not loadable: %s", error)
        return 2
    except VectorIndexError as error:
        # Already phrased for an operator by the adapter, so it is reported as
        # it stands rather than wrapped in a traceback nobody needs to read.
        logger.error("%s", error)
        return 4
    except LLMClientError as error:
        # Configuration and provider failures both land here, and both are
        # operator problems with a message that already says what to fix.
        logger.error("%s: %s", type(error).__name__, error)
        return 3


if __name__ == "__main__":
    sys.exit(main())
