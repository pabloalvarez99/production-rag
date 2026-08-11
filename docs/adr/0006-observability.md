# ADR 0006 — Observability: structured logs and per-node timings first, tracing optional

- **Status:** Accepted
- **Date:** 2026-08-11
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0002](0002-langgraph-query.md) (the graph whose nodes are
  the timing boundaries), [ADR 0004](0004-rerank-cross-encoder.md) (a stage that
  degrades silently unless it reports), [ADR 0005](0005-grounded-generation.md)
  (the stage whose failures are *content*, not status codes),
  [ADR 0003](0003-eval-strategy.md) (what measures quality offline, which this
  is not)

## Context

Through M3 a question was one Qdrant round trip. As of M4 a single question is an
embedding call, two vector searches, an optional rerank provider call, an LLM
call, marker resolution and two guardrail checks. Each of those has its own
latency, its own way of failing, and — for the last three — its own way of
producing a *wrong-looking answer while every component reports success*.

That is the specific problem this milestone addresses. In a RAG system the
interesting failures are not exceptions:

- an answer is slow, and "slow" could be any one of five stages;
- an answer is thin, because the context budget truncated the tail rather than
  because retrieval missed;
- the reranker has been failing open for a week, so ordering quality dropped and
  nothing errored;
- the model is emitting markers that resolve to nothing, and they are stripped
  before anyone sees them.

None of these raise. All of them are ordinary log lines and numbers the system
either records or does not.

At the same time the request path now carries material that must never be
recorded: the prompt contains corpus text verbatim, and provider error bodies can
quote the whole prompt back. An observability layer is the most likely place in
the system for customer data to leak, because its whole purpose is to copy
internal state somewhere else.

So three questions have to be answered together, and answering them by reaching
for a vendor SDK answers none of them:

1. **What is always recorded**, on every request, with no configuration and no
   third party?
2. **What may a caller ask to see**, given that the caller is not necessarily
   trusted?
3. **What is never recorded anywhere**, regardless of log level?

Alternatives considered: OpenTelemetry spans with a collector as the baseline;
a vendor tracing SDK (Langfuse, LangSmith) called from inside the library
functions; verbose `DEBUG`-level logging of everything, filtered at the
aggregator; per-request sampling of full prompt/response pairs into a trace
backend by default.

## Decision

**Structured logs plus per-node timings are the baseline signal and carry no
vendor. Tracing is an opt-in export, never a dependency. Prompt and passage text
are never a signal at all.**

### 1. Per-node timings are always collected, never sampled, never optional

Every graph node is wrapped in a stopwatch and writes `timings_ms[node]` onto the
state object ([ADR 0002](0002-langgraph-query.md) makes the nodes the timing
boundaries — a node is an adapter around one stage, so stage latency and node
latency are the same number by construction). The library result carries the
dict; `QueryResult.to_dict()` renders it as `latency_ms` with a `total_ms`
alongside.

They are collected unconditionally because the cost is one `perf_counter()` pair
per stage and the alternative is asking an operator to reproduce a slow request
that has already happened. A timing breakdown that exists only when someone
thought to enable it is a breakdown that is missing exactly when it is needed.

A **per-node** breakdown rather than one total, because the first question about
a slow or wrong answer is *which stage*, and a total cannot answer it. The node
names are constants (`production_rag.graph.state.NODE_NAMES`), so a rename cannot
silently break a dashboard keyed on them.

### 2. The HTTP response stays narrow; `debug` widens shape, never secrets

`QueryResponse` remains `answer`, `citations`, `refused`, `refusal_reason`. The
collection name, the embedding model, the hit counts and the timings stay on the
library result and in the logs, keyed by request id. A public endpoint that
reports its own collection name and stage latencies is describing its interior to
anyone who asks.

`debug: true` on the request is the sanctioned widening — and the design
constraint that shapes it is that **`debug` is caller-controlled**. Anyone who
can call the endpoint can set it, so it cannot be treated as an authenticated
diagnostic channel. It may therefore expose only things that would be safe to
publish:

| Exposed under `debug` | Withheld regardless of `debug` |
|---|---|
| `timings_ms` per node, and the total | prompt text, system prompt, rendered blocks |
| `hits_retrieved` / `hits_used` | passage text beyond the citations already returned |
| the `rerank` summary (`applied`, `candidates`, `error`) | collection name, embedder model, provider identity |
| `invalid_markers` — count and values | credentials, in any form, at any level |

The asymmetry is the point: `debug` answers "what did the system do", never
"what does the system know". A diagnostic surface that has to be protected is a
diagnostic surface that will be left on.

The exposed subset arrives as a single optional `diagnostics` object rather than
as fields spliced into the response body, so the four stable fields a client
parses never change shape depending on a flag, and a client that ignores
diagnostics needs no conditional at all.

The CLI is treated differently on one axis only: it is already running inside the
trust boundary, so `--debug` may print the whole library object. It still prints
no prompt text, because the CLI's output is what ends up pasted into a ticket.

### 3. Tracing is an export, and the system is complete without it

Langfuse is the configured provider (`observability.tracing`), read from three
environment variables **by name** — `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_HOST`. Tracing stays off unless it is enabled and all three resolve.

Three properties are non-negotiable, and they are what "export" means here:

- **Offline works.** Every command, the endpoint, the CLI, the test suite and the
  eval script run with no tracing configured and no network to a trace backend.
  A laptop with no credentials is the reference environment for this project, and
  an observability layer that breaks it would be the first thing switched off.
- **A trace failure is never a request failure.** The exporter is fail-open in
  the same sense as the reranker ([ADR 0004](0004-rerank-cross-encoder.md)): if
  the backend is down, slow or misconfigured, the request still answers and the
  degradation is logged. Losing a trace is losing a diagnostic; failing the
  request is losing the product.
- **The library does not import a vendor.** Tracing attaches at the seam the
  request id already occupies — the pipeline boundary — not inside
  `production_rag.generation` or `production_rag.retrieval`. A vendor client
  called from a retrieval function makes the retrieval function untestable
  without that vendor, which is the inversion [ADR 0002](0002-langgraph-query.md)
  refuses for the graph framework and is refused here for the same reason.

Langfuse rather than raw OpenTelemetry for the first target because the unit
being inspected is a *generation* — prompt, model, tokens, latency, and the
answer's citations — and a generic span backend renders that as an untyped
attribute bag. OTel is not foreclosed: the seam is the same one, and the
structured logs are already vendor-neutral, which is the fallback path if the
hosted product is not acceptable for a deployment.

### 4. Prompts and passages are never logged, and that is not a level

`observability.logging.log_prompts` and `log_retrieved_text` default to `false`
and are documented as local-debugging switches only. The prompt contains corpus
text verbatim; a log aggregator with these on becomes a copy of the corpus with
none of its access controls, and — unlike the corpus — it is usually readable by
everyone with a dashboard login and retained for months.

Provider error bodies get the same treatment: summarised, never echoed, because
an upstream error can quote the request that produced it, which for a generation
call is the entire prompt. The CLI prints the exception *type* for the same
reason.

One deliberate exception, stated because it is one: the **question** is logged
(`query_completed`). It is user-supplied text and it is what makes a report
correlatable at all, but it is not corpus content, and a deployment handling
sensitive questions should treat that line as personal data and drop the field at
the aggregator rather than losing the event.

### 5. Metrics are declared, not live

`observability.metrics` describes a Prometheus exposition on `/metrics` with
buckets shaped for a RAG request (retrieval in tens of milliseconds, generation
in seconds). The keys stay in the config file to record the intended shape; the
endpoint is not wired, and the config file's status comments say so. A key in the
config is not a claim that the runtime reads it — the same rule the rest of that
file follows.

## Consequences

**Positive**

- "Which stage?" is answerable from an object the caller already has, for every
  request, without reproduction and without a backend.
- The three highest-value ops signals in a RAG system — timings, `invalid_markers`,
  `hits_used` vs `hits_retrieved` — are produced by the request path itself, cost
  nothing, and need no judge. They are the operational counterpart to the offline
  metrics of [ADR 0003](0003-eval-strategy.md), and deliberately not a substitute
  for them.
- A silent degradation becomes a countable one: a reranker failing open reports
  `rerank.error` on every result and a `rerank_failed_open` warning in the logs.
- No vendor is on the critical path. The default deployment has one fewer
  outbound dependency, and the offline demo keeps working.

**Negative**

- **Wall time only.** The timings are `perf_counter()` around a node, so a stage
  that fans out internally (two Qdrant branches inside `retrieve`) reports one
  number. Attributing a slow `retrieve` to the dense or the sparse branch still
  means running the retrieve command per mode.
- **No cost or token accounting.** The provider returns token counts on the
  generation call and nothing aggregates them; a per-query cost figure is a later
  milestone. Latency is not spend, and this milestone measures only the former.
- **`debug` is a deliberately weak channel.** Because it must stay safe for an
  untrusted caller, it cannot expose the things that would answer the hardest
  questions (which collection, which model, what was in the prompt). Those stay
  in the logs, which means real incident work still needs log access — the
  request id is the handle, not the response.
- **Tracing, when enabled, sends prompts and answers to a third party.** That is
  what a generation trace *is*. `sample_rate` bounds the volume, not the
  sensitivity. Enabling it is a data-processing decision about the corpus, not a
  toggle, and it belongs in the same review as any other processor.

**Neutral / follow-ups**

- Metrics (`/metrics`) and alert thresholds are not wired. The config block
  records the intended shape so the eventual implementation is a diff against a
  stated contract rather than a fresh invention.
- Latency under concurrency is still unmeasured ([evaluation](../evaluation.md)):
  per-request timings say nothing about behaviour under load, and load testing
  waits for a real deployment target.
- The request id is the join key between the logs, the response header and — when
  it is enabled — the trace. Anything added later (metrics exemplars, an OTel
  span) attaches at that same seam rather than inventing a second correlation
  scheme.
