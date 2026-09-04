# 0001 — One tokenizer authority for both the budget check and the context plan

**Status:** Accepted (2026-09-04)

## Context

Two layers need to know how large a piece of text is, and they ask for different
reasons.

`llm/gateway.py` asks *before sending*: does this request, plus the reserve held
back for the reply, fit the model's context window? If not it refuses locally and
raises `ContextWindowExceeded`, having spent nothing.

`context/builder.py` asks *while assembling*: this candidate wants space in the
prompt — is there room for it, or does it get excluded `budget_exceeded`?

The obvious way to build the second one is to give the context package its own
`tokenizer.py`. The step specification this work came from is written that way,
as `context/tokenizer.py` alongside `context/builder.py`.

## Decision

The context package does not implement token counting. `ContextBuilder` depends
on the `TokenCounter` protocol already declared in `llm/tokenizer.py`, and takes
a `TiktokenCounter` by default.

`llm/tokenizer.py` already satisfies the specification's stated requirements:
`count_tokens(text, *, model)` built on real `tiktoken`, with an `o200k_base`
fallback for model names the per-model table does not carry. The worked example
in the specification reproduces against it exactly — `'policy note'` → 2,
`'secret document'` → 2, `'milestone status'` → 3, `'old preference'` → 2, under
`gpt-4o` and under the fallback alike.

## Consequences

There is one answer to "how big is this text?" in the runtime, so the number the
builder plans against and the number the gateway checks are the same arithmetic
on the same encoding. When a trace shows a request that was refused, the refusal
has one source.

The two layers still measure different *things*, and this ADR does not pretend
otherwise. `ContextPlan.used_tokens` is the raw sum of the included candidates'
text — no chat framing, no role names — because the acceptance criterion is that
it be independently verifiable by re-encoding those strings. The gateway's
estimate does include per-message framing. The caller bridges them:

```
budget_tokens = context_window - output_reserve - fixed_block_cost
```

The builder budgets only the variable part, because the variable part is the only
part it is allowed to drop. This is written into `context/builder.py`'s module
docstring so the gap between the two numbers is documented where someone
comparing them will find it, rather than discovered as a suspected bug.

A future provider-side counter — one that trades a round trip for an exact count
— swaps in at both sites at once, because both go through the same protocol.

## Alternatives considered

**A second `context/tokenizer.py`, as specified.** Rejected. Two implementations
of the same measurement can drift, and the failure is not loud: the builder
assembles a prompt it believes fits, the gateway measures it differently and
refuses it, and the trace shows two numbers with no way to tell which one is the
authority. Following the specification literally here would have produced exactly
the class of silent disagreement the rest of the specification is written to
prevent.

**A thin `context/tokenizer.py` that re-exports the `llm` one.** Rejected as a
seam that costs more than it saves — it looks like an abstraction boundary while
being none, and the next person to need a change has to discover that editing it
edits nothing.

**Moving the tokenizer out of `llm/` into a shared module.** Deferred. It is a
reasonable end state if a third consumer appears, but today it would be a rename
that touches working, tested code to satisfy a symmetry argument. The dependency
direction (`context` → `llm`) does not violate any boundary rule in CLAUDE.md.
