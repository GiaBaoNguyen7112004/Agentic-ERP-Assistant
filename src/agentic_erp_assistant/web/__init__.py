"""The HTTP/SSE surface over the engine, and the React app it serves.

A thin async shell over a synchronous engine (ADR 0015): the engine's four
ports and the composition root under it are untouched by anything in this
package. ``protocol.py`` is the typed event vocabulary the browser and the
server agree on; ``stream.py`` bridges one turn's blocking worker thread to
an async response; ``service.py`` is the one place a request becomes a call
into :mod:`agentic_erp_assistant.composition`; ``app.py`` is the FastAPI
application and its routes.
"""
