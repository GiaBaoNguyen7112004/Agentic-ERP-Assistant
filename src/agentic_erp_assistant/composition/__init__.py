"""Where the process-wide port graph is assembled, once, over the environment.

Nothing under ``engine/``, ``llm/``, ``tools/``, ``rag/``, or ``memory/`` reads
an environment variable or opens a network connection at import time -- every
one of those packages is a set of ports and pure logic, built from arguments a
caller supplies. This package is the caller. ``settings.py`` turns the
environment into typed configuration; ``resources.py`` builds the process-wide
clients and indexes those settings describe; ``turn.py`` assembles the ports
one turn needs, over those shared resources, per request.

Import direction: this package imports the core packages and nothing imports
it back except ``web/`` (the browser surface), ``scripts/`` (one-shot tools),
and the package entry point. The core stays testable with fakes and no
environment, whatever grows on top of it.
"""
