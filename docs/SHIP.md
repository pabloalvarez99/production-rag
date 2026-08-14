# Ship notes

**Status: [v0.1.0](https://github.com/pabloalvarez99/production-rag/releases/tag/v0.1.0)
is published and the free path is public-ready.** A reviewer clones this repository and reaches a
cited answer without an account, a credential, or a billed call. One item is deliberately
open — a hosted-provider baseline run — and it is stated wherever a number appears rather
than hidden.

This page is the short version. The [README](../README.md) is the long one.

## Run it — the whole demo, no key

Requires Docker and Python 3.12+.

```powershell
git clone https://github.com/pabloalvarez99/production-rag
cd production-rag
.\scripts\demo_setup.ps1          # macOS or Linux: ./scripts/demo_setup.sh
```

Open <http://localhost:8000/> and ask two questions:

- *Why does hybrid search use reciprocal rank fusion?* — a grounded answer with citation
  markers that resolve to the passages behind them, and per-node timings.
- *Who won the Antarctic underwater chess championship?* — an explicit refusal with its
  reason, because the corpus does not support an answer.

`docker compose down` stops the stack and keeps the vector volume. The rest of the
credential-free path — ingest, the query API and CLI, both evaluation tiers, and regenerating
the scorecard — is in [Try it free](../README.md#try-it-free-0-no-api-key).

## What CI proves on every push and pull request

One workflow, one job, and the interesting part is what it withholds: `OPENAI_API_KEY`,
`COHERE_API_KEY` and `QDRANT_API_KEY` are all set to the empty string for the steps that
matter. If the code ever starts quietly requiring a credential, CI goes red instead of
someone discovering it after cloning.

| Step | What it establishes |
| --- | --- |
| `ruff check .` and `mypy --strict` | Lint and full type coverage across the package |
| `pytest -q` with empty provider keys | The suite passes with no credential and no network |
| Qdrant client/server drift check | The pinned client and the running server agree on major.minor |
| Ingest `data/raw` with the deterministic embedder | A collection can be built from a clean checkout |
| Both evaluation tiers, deterministic providers | The retrieval and answer harnesses run end to end |
| `POST /v1/query` against a live server | The HTTP contract returns an answer with citations, credential-free |

The published scorecard region is generated, not typed: `python tools/render_docs.py --check`
fails when the README and the measurement artefact disagree, and the test suite runs that
check as well.

## What is live

Ingest with citable chunk payloads · dense + sparse retrieval fused with RRF · cross-encoder
reranking that degrades ordering rather than availability · grounded answers whose `[n]`
markers resolve to source records · refusal as a first-class outcome · a server-rendered query
UI pinned to the local providers · health, readiness, request ids and structured logs · a
tracing seam that is off by default · two evaluation tiers with paired statistics · scorecard
publication that fails closed.

The [capability table](../README.md#what-runs-today) is the authority, with the file behind
each row.

## What is not shipped

Stated once, plainly, because an honest boundary is worth more than a feature list:

- **No hosted-provider quality numbers.** The published table comes from deterministic local
  providers. It proves the measurement path works; it says nothing about retrieval or answer
  quality. Replacing it needs one billed run with named providers, and that run has not
  happened.
- **No authentication, authorization or rate limiting.** Anyone who can reach the port can
  query the service. Do not expose it to an untrusted network. See [SECURITY.md](../SECURITY.md).
  Metadata filters are live, and they are not a substitute: an allowlist bounds which *fields*
  a query may filter on, never which documents a caller may see
  ([ADR-0011](adr/0011-metadata-filters.md)).
- **No metrics endpoint.** The configuration shape exists and is marked `DECLARED`; the route
  does not.
- **No production retry, timeout or circuit-breaker policy, and no multi-tenancy.**
- **No load or concurrency testing.** Nothing here claims a throughput or latency figure.
- **No judge validated against human labels.** The tier-2 judged columns are indicative, and
  the harness says so in every report.

That hardening is not missing by oversight: it is assigned to the platform project in the
series ([PORTFOLIO.md](PORTFOLIO.md)).

## Where to go next

| You want to | Read |
| --- | --- |
| See all five systems and what each one does not prove | [PORTFOLIO.md](PORTFOLIO.md) |
| Present the whole series live, minute by minute | [DEMO-DAY.md](DEMO-DAY.md) |
| Run the checks, or send a fix | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Report a vulnerability, or handle a leaked key | [SECURITY.md](../SECURITY.md) |
| Read why the system is shaped this way, end to end | [CASESTUDY.md](CASESTUDY.md) |
| Understand a decision rather than the code | [`docs/adr/`](adr/) |
| Read how the measurement is defined | [evaluation.md](evaluation.md) and [ADR-0010](adr/0010-statistical-reporting.md) |
| Operate the stack | [runbook.md](runbook.md) |
