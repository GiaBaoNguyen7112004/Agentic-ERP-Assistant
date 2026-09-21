"""Vendor adapters: the only modules in the package that name a provider.

Everything directly under ``llm/`` is provider-neutral -- the port, the response
contracts, prompt construction, token counting, retry. Everything in here knows
one vendor's wire format and translates it into those types.

Nothing is re-exported from this package, and ``llm/__init__.py`` deliberately
does not import it. Two consequences, both wanted:

* Importing :mod:`agentic_erp_assistant.llm` pulls in no HTTP client and no
  provider dependency. A core that imported its provider on every import would
  be neutral by convention only.
* Choosing a vendor is a visible line of code. ``from
  agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient``
  names the decision at the construction site, where a reviewer -- and the
  composition root that will eventually own it -- can see it.

A test walks the AST of every module directly under ``llm/`` and fails if any of
them imports ``httpx``, ``dotenv``, ``openai`` or ``requests``, so this boundary
is checked rather than trusted.
"""
