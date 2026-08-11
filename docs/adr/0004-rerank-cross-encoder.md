# ADR 0004 — Cross-encoder rerank after fusion, fail-open, three providers

- **Status:** Accepted
- **Date:** 2026-08-10
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0001](0001-hybrid-qdrant.md) (hybrid retrieval on Qdrant),
  [ADR 0002](0002-langgraph-query.md) (the query graph that will host this stage)

## Context

M2 fuses a dense branch and a sparse branch with reciprocal rank fusion. RRF is
deliberately scale-free: it sums `1/(k + rank)` over the branches that returned
a document, so cosine similarity and BM25 scores never have to be calibrated
against each other, and nothing drifts as the corpus grows.

That property is bought with a specific loss. **RRF sees rank, never magnitude.**
A document that is overwhelmingly the best match for the question contributes
exactly what a merely adequate one at the same rank contributes. Worse, both
branches score a document *against the query as a bag of signals*: the dense
branch compares two independently-produced vectors, the sparse branch sums
per-term weights. Neither ever reads the query and the passage together.

The consequence is the failure everyone recognises from a first RAG build: the
top 12 are all plausibly on-topic, and the passage that actually answers the
question is fourth. Recall is fine. Precision at the top is not. No prompt
wording fixes that, because generation only ever sees the order it is handed —
and with a context budget, retrieval order is also truncation order.

A cross-encoder is the standard instrument for exactly this: one model, one
forward pass per `(query, passage)` pair, with full attention across both. It
cannot be precomputed and it cannot be indexed, which is why it is a reranking
stage over a short candidate list rather than the retriever itself.

## Decision

Add a **rerank stage between fusion and everything downstream**, with three
interchangeable providers, disabled by default, and fail-open.

### 1. The stage sits after fusion, never instead of it

Retrieval stays hybrid and RRF stays the fusion. The reranker consumes the fused
candidate list and reorders it; it never queries Qdrant and never introduces a
document that retrieval did not return.

The division of labour is the point: **retrieval optimises recall, rerank
optimises precision.** They are different problems with different instruments,
and collapsing them is how a system ends up unable to say which one regressed.
A chunk that is not in the candidate list is unrecoverable — see `input_top_k`
below, which is the only knob that changes what "unrecoverable" means.

### 2. `input_top_k` (40) is larger than `top_k` (6), and that asymmetry is the design

The reranker is fed **more** candidates than it keeps. Feeding it exactly the
number that survives makes it a no-op sorter of an already-final list; feeding it
40 lets it promote the passage RRF buried at rank 30 into position 2.

Cost scales linearly with `input_top_k` — 40 candidates is 40 forward passes, or
40 passages on the wire for a hosted provider — so this is the stage's price
dial, and it is set where recovery is still possible and latency is bounded.
`input_top_k` is capped by what fusion actually returned: raising it above
`retrieval.dense_top_k + retrieval.sparse_top_k` buys nothing, because fusion can
only pass on what the branches produced.

### 3. Three providers, one interface: `fake`, `local`, `cohere`

| `--rerank` | Config `rerank.provider` | Model | Needs | Live in M3 | What it is for |
|---|---|---|---|---|---|
| `fake` | `fake` | none — query-term overlap in pure Python | nothing | **yes, always** | plumbing: CI, offline laptops, contract tests |
| `local` | `local-cross-encoder` | `BAAI/bge-reranker-base` via sentence-transformers | the `sentence-transformers` dependency + a ~1.1 GB model download, CPU | **yes, when installed** | real reranking with no per-query spend and no data leaving the machine |
| `cohere` | `cohere` | `rerank-english-v3.0` | the `cohere` package, `COHERE_API_KEY`, HTTPS per query | **yes, when keyed** | hosted swap: no model to ship, no cold start, billed per search |

Both real providers import lazily and are optional dependencies by design: nobody
should need torch installed, or a Cohere account, to run `--rerank off` or
`--rerank fake`. A missing package raises a `RerankError` naming what to install
rather than an `ImportError` three frames down. Today `sentence-transformers`
arrives with the `rag` extra; carving out dedicated `rerank` / `rerank-cohere`
extras is a `pyproject.toml` change and belongs to whoever owns that file.

Two more `--rerank` values exist and are not providers: `off` (the default —
M2 behaviour unless you ask for more) and `auto`, which reads `rerank.enabled`
and `rerank.provider` from the YAML. `auto` is what makes the config file the
single switch for a deployment while the flag stays the switch for one run.

`fake` deserves the same warning the fake embedder carries, and for the same
reason. It scores each candidate by the share of distinct query terms its text
contains — deterministic across runs and processes, so a test can assert an
order — but it **models nothing**. It is a cruder version of what BM25 already
computed one stage earlier, so on a real corpus it will barely move the fusion
order, and it will never do what a cross-encoder does. It exists so the rerank
stage is exercisable end to end with no credentials, no download, and no network:
the flag path, the candidate-count arithmetic, the `fail_open` branch, the
emitted hit fields and the JSON contract are all real under it. Its *ranking* is
not a quality signal, and no number measured with it is ever reported as one.

The honest summary: **the rerank plumbing is live; rerank quality is live only on
`local` or `cohere`.** The same shape as the fake embedder in M1/M2 — which is
the point of keeping the shape consistent.

`local` is the default *choice* for the project (it is what the README's
architecture claims) because a cross-encoder that runs on CPU turns reranking
into a fixed infrastructure cost rather than a per-query bill, and keeps the
corpus on the machine. `cohere` stays a supported swap rather than a fork:
identical interface, different latency and cost profile, and the sensible option
when a deployment target cannot host a model.

### 4. `fail_open: true` — a reranker error degrades ordering, never availability

If the reranker raises, times out, or returns a malformed response, the stage
logs the failure and returns **fusion order**, unchanged. The query succeeds.

This is a deliberate asymmetry against how M2 treats a *missing capability*: a
collection without a `sparse` named vector aborts, loudly, because the system
would otherwise be silently unhybrid. Rerank is different in kind — it is an
*improvement* over a result that is already correct-in-kind. Losing it costs a
few points of ordering quality; failing the request costs the answer.

The failure is never silent. Every result reports whether reranking ran, and a
degraded result says so explicitly, so "our nDCG dropped last Tuesday" is
answerable from logs rather than from a bisect.

`fail_open: false` exists for the deployment that would rather 5xx than serve
un-reranked results. It is not the default, and choosing it means accepting that
a Cohere outage is an outage.

### 5. Every reranked hit keeps its pre-rerank rank

A hit produced by a reranked run carries `pre_rerank_rank` (its position in
fusion order) and `rerank_score` (the cross-encoder score), alongside the M2
`branch_ranks` and `branch_scores` that are already there. Both keys are absent
when no reranker ran, so an M2-era consumer of the JSON keeps working unchanged.

The result also carries a `rerank` object — `{applied, reranker, candidates,
error}` — present even when nothing reranked. "Was this ranking reranked, by
what, over how many candidates, and if not why not" has to be answerable from
the response a caller already holds; a rerank that quietly stopped happening
looks exactly like a rerank that is not helping.

Ties break on the pre-rerank rank. A reranker that cannot separate two passages
leaves them in the order retrieval chose rather than shuffling them, which is
also what keeps the output deterministic — an eval number depends on that.

Without those two fields, "the reranker moved this to the top" and "fusion had it
at the top all along" are the same observation, and the stage's contribution is
unmeasurable. With them, the ordering delta is computable per query from the
output the CLI already prints. This is the same reasoning that put per-branch
ranks on a fused hit in M2, applied one stage later.

### 6. Still no generation

M3 reorders passages. It does not answer. There is no `POST /v1/query`, no LLM
call on the query path, no citation rendering — those are M4. The reranker is
the last stage of retrieval, not the first stage of generation.

## Consequences

**Positive**

- Precision at the top becomes a separately tunable, separately measurable
  property. `hit@k` can be reported pre- and post-rerank on the same run, which
  is the only way to say what the stage is worth on this corpus.
- The whole path stays runnable offline and in CI through the `fake` provider,
  so rerank does not become the milestone that breaks the no-credentials
  guarantee.
- The hosted-vs-local decision is a config value, not a rewrite.
- `fail_open` keeps the system's availability independent of a third-party
  reranker's uptime.

**Negative / accepted**

- **Latency.** A cross-encoder over 40 candidates is the slowest stage in the
  retrieval path — hundreds of milliseconds on CPU for `bge-reranker-base`,
  versus tens of milliseconds for the Qdrant round trip. Reranking is opt-in
  partly for this reason.
- **A model to ship.** `local` means a ~1.1 GB download on first use and a
  warm-up on first query. In a container this belongs in the image or in a
  mounted cache; a cold container that downloads a model on its first request is
  a latency incident waiting to happen.
- **A second vendor.** `cohere` adds an API key, a rate limit, and a per-search
  cost to a system that otherwise talks to one provider.
- **`fake` can mislead a reader.** A rerank stage that "runs" everywhere invites
  the exact overclaim this repository is written against. Mitigated by saying so
  in every surface that reports a reranked result, not only here.
- **The candidate ceiling is now the hard limit.** With rerank on, a relevant
  chunk outside `input_top_k` is unreachable by construction. That moves the
  first question on a bad result from "is the ranking wrong?" to "was it even a
  candidate?" — which is why the reported counts include how many candidates the
  stage actually saw.

## Alternatives considered

**No rerank; tune fusion weights instead.** `dense_weight` and `sparse_weight`
are global constants — they cannot express "for *this* query, the lexical match
is the answer". Tuning them by intuition on a 14-item golden set is how a fusion
config stops being explainable. Rejected: it addresses a per-query problem with a
per-corpus knob.

**Rerank with an LLM (listwise, "sort these 40 passages").** Better quality
ceiling, but it is a generation call: expensive, non-deterministic, slow, and it
would put an LLM on the query path one milestone before the milestone that
introduces one. Revisit after M4, when there is a judge to measure it against.

**Skip the `fake` provider; make rerank real-only.** Simpler code, but it makes
the stage untestable without either a credential or a 1.1 GB download, which
breaks the offline-CI property that has held since M1. Rejected.

**Rerank before fusion, per branch.** Reranking each branch's list separately
doubles the cost and then throws the scores away at fusion, which is rank-based.
Rejected as strictly worse.

**Retrieve fewer candidates and skip the stage.** Equivalent to accepting that
precision at the top is whatever RRF happens to give, which is the problem this
ADR exists to address.
