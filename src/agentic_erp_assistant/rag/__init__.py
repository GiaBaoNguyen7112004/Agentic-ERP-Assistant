"""Retrieval: turning a shelf of project documents into citable passages.

The package is organised around one obligation. Every factual answer this
assistant gives about a project document has to arrive with a reference a
reviewer can open, and a reference is only openable if something upstream kept
the address of the passage instead of throwing it away. So the pipeline is built
address-first:

``manifest`` says which documents exist and who may read them, ``loaders`` turn
each format into blocks that still know where they came from, ``chunking`` packs
blocks into token-budgeted chunks that inherit that address, ``access`` decides
who may see a chunk, ``embeddings`` and ``vector_index`` handle the dense half of
search, ``lexical`` the sparse half, ``fusion`` combines them, and ``retriever``
hands the result to the graph as
:class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` objects whose
``source_id`` and ``locator`` are exactly what the answer will cite.

Nothing here imports ``engine``. The retriever satisfies
:class:`~agentic_erp_assistant.engine.ports.DocumentRetrieverPort` structurally,
which is the whole reason that port is a Protocol.
"""
