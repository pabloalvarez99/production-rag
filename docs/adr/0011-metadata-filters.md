# ADR 0011 — Metadata filters: allowlist-only, fail closed, exact keyword match

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0001](0001-hybrid-qdrant.md) (hybrid retrieval on Qdrant and
  the payload indexes this depends on),
  [ADR 0002](0002-langgraph-query.md) (the query graph the filter travels through)

## Context

`retrieval.filters.allowed_fields` has been in `configs/default.yaml` since M2 with
a `status: DECLARED ONLY` comment, and the payload indexes that would make a filter
cheap (`doc_id`, `source`, `tags`) have existed just as long. Nothing read the key.
`POST /v1/query` rejected a `filters` field as an unknown one, which was honest and
also meant the config file described a capability the system did not have.

That gap is the problem this ADR closes, and it is worse than a missing feature. A
reviewer who reads the profile top to bottom sees an allowlist, four field names,
and a comment about keeping them in sync with the payload indexes. The reasonable
conclusion — that filters work and are constrained — is wrong. A configuration file
that reads like an implementation is the same failure class this repository already
guards against for `/metrics` and `generation.stream`: a declared knob presented as
a live one.

Filters also matter for a reason retrieval alone does not cover. Generation can
only cite what retrieval returned, so a filter is not a convenience over the result
list — it decides what evidence the answer is allowed to rest on. "Answer this from
the handbook only" and "answer this from anything indexed" are different questions,
and a system that silently treats them as one produces an answer that is confidently
wrong and indistinguishable from a right one.

## Decision

Wire `filters` through the library, the API and both CLIs, as an **allowlist-only,
fail-closed, exact-keyword** contract.

### 1. Allowlist-only, and a field outside it is an error

A field not in `retrieval.filters.allowed_fields` is rejected. Over HTTP that is
**422** with a typed detail object carrying `error_type: filter_not_allowed`; from
either CLI it is **exit code 2** with the same slug in the last stdout line.

The alternative — dropping an unknown field and answering anyway — is the failure
this ADR exists to prevent. The caller asked one question, got the answer to
another, and has nothing in the response to tell them apart. Rejecting is also the
behaviour the endpoint already had for an unknown *body* field, so the two agree.

`error_type` is a stable slug rather than an exception class name because it is the
part clients branch on. `filter_not_allowed` and `filter_invalid_value` are the two
values; the message is for humans and may be reworded.

### 2. Empty filters is exactly the old path

`filters` omitted, `null`, or `{}` all mean "no filter", and the code takes the same
branch it took before this feature existed — no Qdrant expression is built, and the
retrieval summary reports `applied: false`. This is what makes the change safe to
ship against existing goldens: an unfiltered run is byte-identical to an M6 one.

### 3. Public field names are mapped explicitly, and `source` is not `source_path`

The one pair a reader gets wrong:

| Public API field | Payload key | What it holds | Indexed |
|---|---|---|---|
| `source` | `source` | first path segment under the corpus root, e.g. `sample` | yes |
| `title` | `title` | document title from front matter, first H1, or file stem | **no** |
| `tags` | `tags` | front matter tags, list-valued | yes |
| — | `source_path` | full corpus-relative path, e.g. `sample/01-hybrid-search.md` | no |

`source_path` is deliberately **not** filterable and is not silently answered
against `source`: the two select different document sets, and a caller who asked
for one and got the other has no way to notice.

The mapping is a table in code (`PUBLIC_TO_PAYLOAD`), not the allowlist itself,
because the two answer different questions. The allowlist says what a *deployment*
permits. The table says what the *payload* can express. A filter needs both, so
adding a name to `allowed_fields` cannot conjure a filterable field — which is why
`created_at` was removed from the shipped default in this change: no ingest run has
ever written that key, so it could only ever have matched nothing, and an empty
result reads as a corpus gap rather than a configuration mistake.

### 4. Exact keyword matching only — no ranges, no substrings, no negation

One condition is "this field equals one of these values". Several values on one
field are OR; several fields are AND. That is the whole expression language.

Every allowlisted field is a keyword payload. A range or substring predicate over a
keyword index is exactly the payload scan the allowlist exists to keep out, and a
filter language is the kind of surface that grows one operator per request until it
is an unbounded query API with no owner. Widening it is another ADR.

Values must be strings. A number or a boolean would build a condition that never
matches — Qdrant compares a keyword index against keywords — and silently returning
nothing is the failure mode this whole ADR is about.

### 5. An unindexed filter works, and is never silent

`qdrant.payload_indexes` decides which filters stay O(log n). A field that is
allowlisted but unindexed is still permitted: it is the honest answer to the
question asked. It degrades to a payload scan, so it is logged at warning level as
`filter_field_unindexed` and named in the retrieval summary's `filters.unindexed`.

Permission and cost are separate lists on purpose. `title` ships allowlisted and
unindexed precisely so the warning path is exercised by the default profile rather
than only by a test.

### 6. Validated once, at the edge, before anything is built

`FilterPolicy.build` runs in the HTTP route before an embedder, a store or a model
is constructed, and again inside the retriever before the embedding call. Same
object, same rules — the second call is what protects a library caller who never
goes through HTTP, not a second implementation.

A rejected filter therefore costs no provider round-trip, which matters because a
malformed filter is a client bug and client bugs repeat.

### 7. The filter that was applied is on the result

`RetrievalResult.query_filter` and `QueryResult.filters` carry a summary — fields,
values, payload keys, and which were unindexed — rendered into the CLI's JSON and
the library result. Same argument as the rerank summary: a hit count measured under
a filter is not comparable with one measured without it, and an eval row that cannot
tell the two apart will be misread.

## Consequences

**What this buys.** The config file now describes the system. A reviewer reading
`retrieval.filters.allowed_fields` and then `POST /v1/query` finds the same
capability in both. Scoped questions ("from the handbook only") are answerable, and
the evidence a generated answer may rest on is controllable by the caller.

**What it costs.** A new public field on the request schema, and a new failure mode
callers must handle: a 422 that is not a schema violation. The typed `error_type` is
what keeps that handleable.

**What it does not do.** It is not authorization. An allowlist bounds which *fields*
a query may filter on; it says nothing about which *documents* a caller may see, and
nothing here should be read as multi-tenancy. Anyone who can reach the port can
still query the whole corpus with no filter at all — see [SECURITY.md](../../SECURITY.md)
and the platform project that owns the edge.

**What stays open.** Range predicates on a real `created_at` field would need both
an ingest change (nothing writes it) and a Qdrant range index, and would reopen the
expression-language question in section 4. Not attempted here.

## Alternatives considered

**Accept any payload key.** Rejected. It turns the query endpoint into an unbounded
scan API over whatever ingest happened to write, and the first unindexed field in
production is a latency incident with no owner.

**Drop unknown fields and answer anyway.** Rejected, and it is the central decision
of this ADR. It is the only option here that can produce a wrong answer the caller
cannot detect.

**A JSON string argument on the CLI instead of repeatable `--filter`.** Rejected:
JSON on a PowerShell command line needs quoting that no runbook reader gets right
first time, and this project's scripts are PowerShell.

**Validate in the route only.** Rejected. A library caller — the eval harness, a
notebook — would then bypass the allowlist entirely, and the CLI would need a third
copy of the rules.

**Filter after retrieval, in the retriever.** Rejected. Post-filtering a top-k list
returns fewer hits than asked for whenever a match ranked below the cut, and quietly
changes what `top_k` means. Qdrant filters before the search; the in-memory store
does the same, which is what makes the offline tests meaningful.
