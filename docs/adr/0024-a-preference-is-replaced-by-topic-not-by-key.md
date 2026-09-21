# 0024 — A preference is replaced by topic, not only by key

**Status:** Accepted (2026-09-15)

## Context

The dev database held two live preferences about how budget numbers should be
shown, three minutes apart: `budget_reporting_format` ("The user prefers
budget numbers to be reported in thousands of USD.", 22:39 UTC) and
`budget_reporting_currency` ("... in thousands of VND.", 22:42 UTC). Both were
live (`superseded_at IS NULL`, `supersedes = {}`). The turn that wrote the
second one had the first one recalled into its own prompt —
`memory_recalled 3 memor(y|ies) recalled` — and the audit row still said
`write / new preference`: the policy never saw a conflict. The same pattern
had already happened once that day for a different preference (`atlas_reply_
prefix` / `project_atlas_prefix_request`, both "prefix replies about Atlas
with 'ATLAS 2026'", worded two ways). One turn later, asked for the budget
again, the assistant recalled all four preferences at once and replied with a
raw `$187,200` — the newer instruction was not honoured because its own
contradiction was sitting next to it in the prompt.

`memory/policy.py::_resolve_conflict` decides `update` vs `write` on an exact
`(kind, key)` match. `key` is `MemoryProposal.key`, free text the proposer
invents fresh each call — the field description asks for stability ("a later
version of the same fact must arrive under the same key") but nothing
enforces it, and two independent calls at `temperature=0.0` produced two
different keys for one preference. The twin rule ADR 0023 added
(`_resolve_conflict`'s duplicate check, any key, same statement) does not
reach this case either: the statements differ ("USD" vs "VND"), by design,
because a *changed* fact under a different key is meant to be a write — that
rule is `test_a_changed_statement_under_another_key_is_still_a_write`, and it
is the intended behaviour for `fact`. For `preference` it is the bug: a
person has one preference about a given subject, and a second one on the same
subject is not a new fact sitting beside the first, it is the same preference
restated.

The proposer could not do better even if the model tried: `_render_memory`
showed the recalled preference's statement and its date, never its key, so
"reuse the key" was an instruction with no way to be followed.

## Decision

**D1 — the proposer is shown each memory's key, in its own prompt only.**
`_render_memory` gained a keyword-only `with_keys: bool = False`; only
`build_memory_messages` passes `True`, so a recalled preference reads
`(2026-09-14, key: budget_reporting_format) The user prefers ...` in the
memory-proposal prompt, and `MEMORY_CONTRACT` gained one sentence asking the
model to reuse a listed memory's key when it is proposing a newer version.
The answering, planner and promotion builders keep the default — an answer
must never mention a key, since it is not a source and reciting one would
look like a citation of nothing — so `eval/routing.py`'s byte-identity
comparison (ADR 0020) is unaffected.

**D2 — a preference on the same topic is an update, whatever key it arrived
under.** `_resolve_conflict` gets a fourth branch, `preference`-only: a live
in-scope preference whose statement shares at least `TOPIC_OVERLAP_RATIO`
(0.7) of its content words with the candidate, measured in the smaller of the
two directions (so a short statement contained inside a long one does not
merge on the strength of the longer one alone), is treated as the same
preference restated, and superseded. Same lexical machinery
(`_content_words`, `_overlap`) the source-ownership check already used; no
embedding, no model call, the function stays pure.

0.7 was chosen against the real pairs, measured with the shipped function:
the USD/VND pair scores 0.86, the two ATLAS-prefix statements score 0.86, and
the nearest real negatives — "budget numbers in thousands of USD" against
"replies written in Vietnamese" — score 0.29, and against the ATLAS-prefix
statement, 0.14. 0.7 sits in that gap with margin on both sides. It is not
free of an over-merge case: "budget numbers reported in thousands of USD"
against "budget numbers reported per sprint" scores 0.71 and would be treated
as the same topic, merging a currency preference with a cadence preference.
Accepted (see Consequences).

Preference-only, deliberately: a `fact` or `decision` that changes one word
can be a genuinely different fact ("milestone M2's owner is Priya" /
"milestone M3's owner is Priya" would also score well above 0.7), and merging
those would silently lose one. A preference is "how this person wants to be
worked with" *on a subject*, and word overlap is a reasonable proxy for the
subject; two preferences on one subject that disagree cannot both be
honoured, which is exactly the state two live rows put the assistant in.

**D3 — the record keeps the key that was already there.** When D2 fires, the
stored record is written under the *existing* key, not the one the proposer
invented this turn (`MemoryDecision` gained an optional `key` field, valid
only on `update`, applied in `SessionMemory._judge_proposals` before
`memory_id` is derived). Without this the key would drift to a new value on
every rewrite, and the exact-key rule could never match again on its own —
D2 would have to fire every single time. In practice the model routinely
reuses the key D1 shows it (confirmed live, see Evidence), so the exact-key
path fires directly and D2 is the backstop for when it does not.

**D4 — the audit says which rule fired.** The `update` reason names both
keys when D2 fires (`"supersedes 1 live preference(s) on the same topic under
'budget_reporting_format' (86% shared); proposed under
'budget_reporting_currency'"`), distinct from the exact-key wording
(`"supersedes 1 live preference record(s) for '<key>'"`) already in place.
No new trace event or table column — `memory_written`/`memory_audit.reason`
already carry `decision.reason`.

The conflict check still runs last, after every attack and content rule, and
both of its update paths (exact-key, topic) inherit that ordering: a
candidate that also trips `instruction_like` is refused before either can
fire, so a poisoned statement about the same topic as a live preference
cannot arrive as an update and take its place.

## Evidence

Dev database repaired by hand once (`UPDATE memories SET superseded_at =
now() WHERE memory_id IN ('mem-33e82a6b...', 'mem-dcdb70a6...')` — the older
half of each duplicate pair), then verified live with `gpt-4o`,
2026-09-15 (`docs/manual-test.md` M9–M11):

- New chat, "What's the remaining budget?" → `memory_recalled 2` (not 4);
  reply `ATLAS 2026: ... 187,200,000 VND out of the approved 480,000,000
  VND.` — the newer preference, honoured, no raw dollar figure.
- Same session, "From now on, report budget numbers in thousands of USD." →
  `memory_written update (supersedes 1 live preference record(s) for
  'budget_reporting_currency')`. The model reused the exact key D1 showed it,
  so the exact-key path fired directly; D2's topic-overlap branch was not
  exercised by this run, by construction (D1 working as intended), and is
  covered instead by the unit fixtures (`tests/memory/junk_fixtures.py::
  DRIFTED_PREFERENCE_PAIRS`, `test_policy.py`, `test_service.py`) built from
  the two pairs the dev database actually held with drifted keys.
- New chat, same question, as priya → `memory_recalled 2`; one live budget
  preference (USD). Switched to wei → `memory_recalled 0`, no ATLAS prefix:
  preferences do not cross actors (`ACTOR_BOUNDED_KINDS`).

## Consequences

- **A preference on one subject can only ever be stated once per actor.**
  That is the intended behaviour, not a limitation: two live, contradicting
  preferences on the same subject is the defect this ADR fixes.
- **The lexical rule over-merges at the edges**, as shown by the 0.71 example
  above. Accepted for the reason D2 gives: a preference told twice costs a
  repeated sentence next time the user restates it; two live preferences
  silently contradicting each other — the state actually observed — costs a
  reply that honours neither. A reviewer can always see which pair merged and
  by how much, because the audit reason names the share.
- **D3 means a preference's key can still change once**, on the turn D2 (not
  the exact-key rule) first catches a drift — after that it is stable, since
  every later rewrite adopts the same stored key. A key already visible to a
  person reading the audit trail can therefore shift exactly once per
  preference; it is not exposed anywhere a user would see it directly.
- **The rule is not evaluated against a scored eval set.** Like
  `MIN_COSINE_SIMILARITY` in `rag/retriever.py`, `TOPIC_OVERLAP_RATIO` is
  provisional until enough real preference pairs exist to run a routing-
  comparison-style report against; for now it is defended by the two real
  pairs, two real negatives, and the fixtures built from them.

## Alternatives considered

- **Semantic near-duplicate detection through the memory index (Qdrant).**
  One more embedding call per candidate, and a cosine threshold with no
  evidence run behind it yet — the same reason `MIN_COSINE_SIMILARITY` is
  still provisional. Would also push I/O into `policy.decide`, whose value is
  that it has none (ADR 0011).
- **Making the proposer declare a `replaces_memory_id` argument.** Still a
  model-chosen identity; fixes nothing on the turn the model gets it wrong,
  which is precisely the turn this ADR is about.
- **One live preference per actor, full stop.** Wrong on its face: language,
  rounding and a reply prefix are three independent preferences, and a
  person may hold all three at once.
- **Applying the topic-overlap rule to `fact` and `decision` as well.**
  Rejected in Decision D2: the false-merge cost for those kinds is losing a
  fact, not repeating one, and that asymmetry is the reason this rule stops
  at `preference`.

See `fix-memory-key-drift-plan.md` (repo root) for the phased implementation
and the fixtures. Related: ADR 0011 (the pure policy and its default to
refuse), ADR 0023 (memory is what the user established).
