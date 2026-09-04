# 0002 — `ContextBuilder` takes a model with no default

**Status:** Accepted (2026-09-04)

## Context

`ContextBuilder` measures candidate text to decide what fits in the prompt. Every
measurement is made under some model's encoding, and the same string is a
different number under a different one. So the builder needs a model name.

A default would be convenient — `ContextBuilder()` in a test, in a notebook, in a
quick script. The pressure to add one is real and will recur.

The repository has already answered this question once, in a neighbouring layer.
`LLMGateway.context_window` is a required field with no default, and its
docstring gives the reason: a default would be one model's number silently
applied to whichever model `.env` happens to name.

## Decision

`ContextBuilder.model` is required. There is no default, and there is no fallback
to an environment variable read inside the builder.

The related rule, enforced structurally rather than by documentation:
`ContextCandidate` has no `token_count` field. A caller cannot tell the builder
how large its text is; the builder measures `text` itself. A count that cannot be
supplied cannot be stale.

## Consequences

Constructing a builder forces the caller to say whose arithmetic it wants, at the
same place it already configures the client — so the two cannot silently disagree
about which model is in play.

Tests must name a model. They already do: `tests/context/test_builder.py` uses
`gpt-4o` and computes its expectations with `tiktoken` directly, and one test
pins the behaviour under a model name `tiktoken` has never heard of, so the
fallback path is exercised rather than assumed.

The cost is a slightly noisier construction site. That is the intended trade:
this project's failure mode is a budget decision that looks correct and was made
with the wrong encoding, and that failure is invisible until a provider rejects a
request the runtime believed it had already checked.

`ContextCandidate` gaining no size field means a caller with an expensive
tokenization cannot pass a cached count in. If that becomes a real cost, the
answer is a cache inside the counter — keyed by text and model, where it stays
correct — not a number travelling beside the text it describes.

## Alternatives considered

**Default to the tokenizer's fallback encoding.** Rejected. It would make every
misconfiguration produce a plausible number instead of an error, which is the
precise shape of the bug the budget check exists to catch.

**Read `OPENAI_MODEL` from the environment inside the builder.** Rejected. It
would put provider configuration inside a runtime module, and CLAUDE.md keeps
provider naming in the adapter layer. It would also make the plan depend on
ambient state, so the same candidates could produce different plans on two
machines — the opposite of the reproducibility this component is for.

**Allow a caller-supplied `token_count` on `ContextCandidate`, documented as an
override for expensive cases.** Rejected. Every such field is correct on the day
it is written and wrong the first time someone edits the text beside it.
