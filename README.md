# production-rag

Production-grade Retrieval-Augmented Generation service: **hybrid retrieval over Qdrant, cross-encoder reranking, answers that carry citations, and an evaluation gate that decides whether a change ships.**

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Vector store](https://img.shields.io/badge/vectors-Qdrant-DC244C.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this is

Most RAG demos are one embedding call, one cosine search and a prompt. They fall over on the first real corpus: acronyms and part numbers that dense embeddings cannot see, top-k results that are similar but not relevant, answers no one can trace back to a source, and no way to tell whether last week's prompt change made things better or worse.

This service is the other thing — the shape a RAG system takes when it has to be operated:

| Concern | Position taken here |
| --- | --- |
| Retrieval | **Hybrid**: dense vectors *and* sparse/BM25 in one Qdrant query, fused with reciprocal rank fusion. Exact identifiers stop being invisible. |
| Precision | A **cross-encoder reranker** (`bge-reranker-base`) reorders the fused candidates before anything reaches the LLM. Retrieval recall and answer precision are separate problems. |
| Trust | Every answer returns **citations** to the chunks that produced it. An answer without a source is not an answer. |
| Change safety | **Ragas** plus a golden set, run as a regression gate. "It feels better" is not a result. |
| Operability | Config from the environment, structured logs with a correlation id per request, liveness and readiness probes, containerised from day one. |

The stack is locked so each milestone is an implementation task rather than an architecture debate: **LlamaIndex** (ingest and node parsing), **LangGraph** (the query graph, from M4), **Qdrant** (dense + sparse vectors, payload filters), **bge-reranker-base** (local reranking, with Cohere as an optional swap), **FastAPI** + **Pydantic v2** (HTTP surface), **Ragas** (evaluation), **structlog** + OpenTelemetry (observability).

## Current status — M0 (scaffold)

M0 is the walking skeleton, and nothing more. What exists today:

- An installable, typed Python package (`src/` layout, `production_rag`).
- Environment-driven configuration with validation and secret masking.
- `GET /health`, `GET /v1/health`, `GET /v1/ready`, plus OpenAPI at `/docs`.
- A correlation id (`X-Request-ID`) bound to every request and every log line.
- A test suite that passes **with no network and no Qdrant running**.

What deliberately does not exist yet: ingestion, embeddings, retrieval, reranking, generation, evaluation. Readiness reports whether a vector store is *configured*; it opens no sockets. See the [roadmap](#roadmap).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, unversioned — for container and orchestrator probes that should never chase an API version. |
| `GET` | `/v1/health` | The same payload inside the versioned namespace, for API clients. |
| `GET` | `/v1/ready` | Readiness: configuration parsed, and whether a Qdrant endpoint is set. |
| `GET` | `/docs`, `/openapi.json` | Interactive documentation and the machine-readable schema. |

```console
$ curl -s localhost:8000/health
{"status":"ok","service":"production-rag","version":"0.1.0","environment":"local"}

$ curl -s localhost:8000/v1/ready
{"status":"ready","qdrant_configured":true,"checks":{"settings":"ok"}}
```

Both probes are separate operations in the schema (`health`, `health_unversioned`) so generated clients stay valid.

## Local quickstart (no Docker)

Requires Python 3.12+. Nothing below contacts the network at runtime, and no credentials are needed to boot or to run the tests.

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install the package plus the dev toolchain, editable
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 3. Run the API
python -m uvicorn production_rag.main:app --reload --port 8000

# 4. Probe it (in another shell)
curl -s localhost:8000/health
curl -s localhost:8000/v1/ready
open http://localhost:8000/docs
```

### Tests, linting, types

```bash
python -m pytest -q            # unit suite: offline, no Qdrant, no API key
python -m ruff check .         # lint
python -m ruff format --check .
python -m mypy                 # strict, config in pyproject.toml
```

### Configuration

Every setting is an environment variable, optionally seeded by a `.env` file in the project root. Unknown keys are ignored, so the same `.env` can be shared with Docker Compose.

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_NAME` | `production-rag` | Service name reported by the probes. |
| `APP_VERSION` | `0.1.0` | Version reported by the probes. |
| `ENVIRONMENT` | `local` | Deployment environment. Anything other than `local` switches logs to JSON. |
| `LOG_LEVEL` | `INFO` | Standard level name, any case. |
| `API_PREFIX` | `/v1` | Versioned route prefix. Normalised; `/` is rejected. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address for uvicorn. |
| `QDRANT_URL` | `http://localhost:6333` | Vector store endpoint. Reported by `/v1/ready`; dialled from M1. |
| `QDRANT_COLLECTION` | `documents` | Collection holding chunks and their vectors. |
| `CONFIG_PATH` | *unset* | Optional YAML file with retrieval/rerank knobs; consumed from M1. |
| `OPENAI_API_KEY` | *unset* | Only needed once generation lands (M4). |

`OPENAI_API_KEY` is never logged and never serialised into a response. `Settings.safe_dump()` is the only sanctioned way to render settings, and a test asserts the masking.

> **Never commit a real `.env`.** It is gitignored; the tracked template is `.env.example`.

The quickstart above is the fastest edit loop and needs neither Docker nor a running Qdrant. For the container path, see the next section.

## Docker & Ops

Two containers: `api` (built from `Dockerfile` — `python:3.12-slim`, non-root, uvicorn on 8000) and `qdrant` (pinned to `qdrant/qdrant:v1.13.2`). Both carry healthchecks; the API waits for Qdrant to report healthy before it starts. Prerequisites are Docker Engine 24+ with the Compose v2 plugin and ports 8000, 6333 and 6334 free.

```bash
cp .env.example .env             # PowerShell: Copy-Item .env.example .env
docker compose up -d --build     # or: make up  /  .\scripts\up.ps1
docker compose ps                # both services should read (healthy)
make health                      # probe the API and Qdrant, non-zero exit on failure
make logs                        # tail both services
make down                        # stop, keep the vector index
```

| Surface | URL |
| --- | --- |
| API docs | <http://localhost:8000/docs> |
| Liveness | <http://localhost:8000/health> |
| Readiness | <http://localhost:8000/v1/ready> |
| Qdrant dashboard | <http://localhost:6333/dashboard> |

Vectors live on the named volume `production-rag-qdrant-storage`, which survives `docker compose down`. Only `down -v` (`make clean`, or `scripts/down.ps1 -Purge`, which prompts first) destroys it, and that costs a full re-ingest. The Qdrant tag is pinned because its storage format is version-sensitive — the [runbook](docs/runbook.md) has the backup-then-upgrade procedure.

`configs/` and `data/` are bind-mounted read-only, so a config edit or a new corpus document needs a restart, not a rebuild.

### Make targets and PowerShell scripts

`make` is optional. The PowerShell scripts are behaviourally equivalent for Windows machines without it.

| Entry point | Purpose |
| --- | --- |
| `make up` / `scripts/up.ps1` | Build and start the stack, then wait for `/health`. On timeout, dumps the last 50 API log lines and exits non-zero. |
| `make down` / `scripts/down.ps1` | Stop the stack. `-Purge` also drops the vector volume, after a confirmation prompt. |
| `make restart` | Rebuild and recreate the API container only — the fast loop for ops changes. |
| `make health` / `scripts/health.ps1` | Probe every health surface plus Qdrant readiness and collection presence. |
| `scripts/smoke_health.py` | Stdlib-only smoke test, so it runs from any interpreter even when the install is what you suspect. `--json` for CI, `--retries` for the cold-boot window. |
| `make logs` / `make ps` | Tail logs / show status and health. |
| `make test` | Run the suite inside the API container (`tests/` is mounted, not baked in — the image ships runtime deps only, so `make test-host` is the fallback). |
| `make shell-api` | Interactive shell in the running API container. |
| `make clean` | `down -v`. Destructive: drops the vector index. |

### Layered configuration

Environment variables (documented in the table above) are the highest-precedence layer and the only place a credential appears. Below them sits `configs/default.yaml`: the declarative knobs for ingest, retrieval, rerank, generation, Qdrant topology, evals and observability. It is deliberately broader than what M0 consumes, so later milestones change values rather than structure — `CONFIG_PATH` selects a profile that overrides it.

Config files name the environment variable holding a secret (`api_key_env: OPENAI_API_KEY`), never its value. `.env` is both gitignored and dockerignored; `.env.example` is the tracked template and carries empty values only.

### Documentation

- [Architecture](docs/architecture.md) — components, query and ingest paths, failure behaviour
- [Data model](docs/data-model.md) — collection schema, payload fields, chunk identity
- [Runbook](docs/runbook.md) — start, verify, debug, recover, upgrade Qdrant
- [Evaluation](docs/evaluation.md) — golden dataset, metrics, thresholds
- ADRs — [0001 hybrid retrieval on Qdrant](docs/adr/0001-hybrid-qdrant.md) · [0002 LangGraph query orchestration](docs/adr/0002-langgraph-query.md) · [0003 evaluation strategy](docs/adr/0003-eval-strategy.md)
- Data layout — [`data/raw/`](data/raw/README.md) · [`data/eval/`](data/eval/README.md) · [`data/processed/`](data/processed/README.md)

## Project structure

```text
production-rag/
├── src/production_rag/
│   ├── __init__.py            # package version and public surface
│   ├── config.py              # Settings (pydantic-settings) + cached get_settings()
│   ├── main.py                # create_app() factory, logging setup, ASGI `app`
│   └── api/
│       ├── deps.py            # SettingsDep — the injection seam the tests use
│       ├── middleware.py      # correlation id, timing, one structured access log
│       ├── schemas.py         # response models (HealthResponse, ReadyResponse)
│       └── routes/
│           ├── health.py      # GET /health and GET /v1/health
│           └── ready.py       # GET /v1/ready
├── tests/
│   ├── conftest.py            # fixtures: isolated settings, app + client factories
│   └── unit/                  # health, ready, config, middleware, version
├── configs/                   # YAML profiles (retrieval and rerank knobs, M1+)
├── data/                      # raw corpora, processed chunks, eval sets
├── docs/                      # architecture notes and ADRs
├── scripts/                   # operator helpers
├── docker-compose.yml         # API + Qdrant
├── Dockerfile
├── Makefile
└── pyproject.toml             # deps, ruff, mypy, pytest — one config file
```

Milestones add sibling packages under `src/production_rag/` (`ingest/`, `retrieval/`, `rerank/`, `generation/`, `evaluation/`) rather than growing the existing modules. Each stage is exercisable on its own — that is what makes an evaluation gate possible.

### Design notes

A few decisions worth stating, because they are the ones that get undone by accident:

- **Liveness carries no dependency state.** A liveness probe that fails when Qdrant is down makes the orchestrator restart a healthy process, which does not fix Qdrant and drops in-flight requests. Dependency state belongs to readiness.
- **`create_app()` is a factory, not module-level assembly.** Tests build an app around explicit settings via `app.dependency_overrides`, so no state leaks between cases and the suite never mutates the environment.
- **The correlation id is bound once, into structlog contextvars.** From M4 a single question fans out into embedding, retrieval, rerank and generation calls; without one id on every line, debugging an incident is guesswork. An inbound id that is absent or malformed is replaced with a fresh UUID4 — it reaches both a response header and the logs, so it is validated, not trusted.
- **Offline tests are a feature, not a limitation.** Readiness is a configuration check by design, which is what lets `pytest` be green with the wifi off.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| **M0** | Scaffold: package, config, health/readiness, tests, container stack | ✅ done |
| **M1** | Ingest with LlamaIndex; chunking; dense embeddings; Qdrant collection and upsert | ⏳ next |
| **M2** | Hybrid retrieval: sparse/BM25 vectors alongside dense, fused with RRF | 📋 planned |
| **M3** | Cross-encoder reranking (`bge-reranker-base`), Cohere as an optional swap | 📋 planned |
| **M4** | Generation with citations; `POST /v1/query` orchestrated as a LangGraph graph | 📋 planned |
| **M5** | Observability: OpenTelemetry traces, structured logs, token and latency metrics, optional Langfuse | 📋 planned |
| **M6** | Evaluation: Ragas metrics, a golden set, and a regression gate in CI | 📋 planned |
| **M7** | Hardening: rate limits, timeouts, retries and backoff, input validation, graceful degradation | 📋 planned |
| **M8** | Portfolio polish: architecture write-up, benchmark numbers, ADRs, demo corpus | 📋 planned |

Retrieval quality claims arrive with M2–M3 and will be reported as measured numbers on a stated corpus, never as adjectives.

## Ingest (M1)

M1 adds the offline ingest path and nothing else: walk a corpus directory, parse
front matter, chunk on structural boundaries, embed the chunks (dense), and
upsert them into Qdrant with a payload that makes every chunk citable. There is
still no query endpoint, no hybrid retrieval, and no generation — those are M2
and M4. No retrieval number in this repository has been measured.

The job is `python -m production_rag.ingest`. Everything below is a wrapper over
it, and it runs inside the `api` container by default so `QDRANT_URL` resolves
to the compose hostname and nothing needs installing on the host.

### The fake path — no key, no network, no spend

```bash
make ingest-dry                  # walk + chunk, report counts, write nothing
make ingest-fake                 # ingest data/raw with the fake embedder
make ingest-fake SOURCE=data/raw/my-corpus
```

`SOURCE` is the corpus **root**, not a document folder. Payload `source_path`
values are relative to it, and its first path segment becomes the filterable
`source` field — which is why the default is `data/raw` and a sample chunk reads
as `sample/08-bm25-vs-dense.md`. That is exactly the string
`data/eval/golden.jsonl` labels; pointing `SOURCE` deeper silently stops those
labels matching.

```powershell
.\scripts\ingest.ps1 -DryRun     # Windows without make
.\scripts\ingest.ps1
```

`--embedder fake` is a deterministic hash embedder: the same text always maps to
the same vector, of the same declared dimensionality as the real one. That makes
the whole path — walk, chunk, embed, upsert, count — runnable in CI and on a
laptop with no credentials, which is the only way this stage stays testable.

Its vectors carry no semantics. Any similarity or recall number measured against
a fake-embedded collection is noise, and none is reported anywhere in this repo.
Use it to prove the plumbing; never to make a quality claim.

Verify afterwards:

```bash
python scripts/smoke_health.py       # liveness + Qdrant readiness + collection presence
curl -s http://localhost:6333/collections/production_rag | jq .result.points_count
```

### The provider path — real embeddings, real money

```bash
cp .env.example .env             # put OPENAI_API_KEY in .env, never on the CLI
make ingest-sample
```

```powershell
.\scripts\ingest.ps1 -Embedder openai
```

The key is read from the environment or from the gitignored `.env` that Compose
loads. Neither the Makefile nor the PowerShell script has a flag to pass it,
because a credential on the command line lands in shell history and in `docker
inspect` output.

Embedding is the only stage in this system that costs money per document, which
is why the content hash and the incremental skip sit *before* the embed call.
Re-running ingest over an unchanged corpus is nearly free; `make ingest-dry`
reports the chunk count, and that count times the average chunk size is the
token bill you are about to pay.

### Corpus and golden set

`data/raw/sample/` holds nine committed Markdown documents on RAG mechanics —
enough for chunking to produce a meaningful number of chunks and for the
exact-token versus paraphrase distinction to be visible.
`data/eval/golden.jsonl` holds a 14-item seed set labelled at *document*
granularity (`expected_source_paths`), because chunk ids do not survive a
chunking change and M1 is exactly when those settings move. It pins the schema
and is explicitly not a merge gate — see [evaluation](docs/evaluation.md).

Details: [runbook](docs/runbook.md#ingest-m1) for operations and failure modes,
[data model](docs/data-model.md) for the exact payload written to Qdrant,
[architecture](docs/architecture.md) for the stage diagram.

## Retrieve (M2)

M2 makes retrieval real. Sparse BM25 vectors are written alongside the dense
ones at ingest, both named vectors are queried for every question, and the two
ranked lists are fused with reciprocal rank fusion. This is the milestone where
the hybrid claim at the top of this README stops being a plan.

**What is live: retrieval. What is not: generation.** There is no
`POST /v1/query`, no LLM call, no answer, no citations. Retrieval is a batch
command that prints ranked passages. Reranking is M3; generation and the HTTP
query surface are M4.

```
  question
     │
     ├─────────────────────┬──────────────────────┐
     ▼                     ▼                      │
  embed(query)        tokenise → BM25             │
     │ dense vector        │ sparse vector        │
     ▼                     ▼                      │
  Qdrant `dense`      Qdrant `sparse`             │  one collection,
  cosine kNN          dot product = BM25          │  one round trip
  40 candidates       40 candidates               │
     │                     │                      │
     └──────────┬──────────┘                      │
                ▼                                 │
     RRF:  score = Σ w / (60 + rank)  ────────────┘
                ▼
     threshold, then top 12
                ▼
     hits[] carrying per-branch ranks
```

Every fused hit carries the rank it held in each branch that returned it, and
that branch's share of the fused score. "Second, because BM25 ranked it 1st and
dense ranked it 14th" is a debuggable statement; a bare fused score is not. A hit
ranked by `sparse` and not `dense` is a document the embedding branch never
returned — hybrid retrieval doing the thing it was adopted for, visible rather
than assumed.

### Migration: an M1 collection must be rebuilt

M1 created the collection with `dense` as its only named vector. It did **not**
pre-declare an empty `sparse` vector, so M2 is a migration rather than a
backfill, and a collection left over from M1 cannot serve hybrid retrieval:

```bash
make reingest-fake       # drop + re-ingest with both vectors
```

```powershell
.\scripts\ingest.ps1 -Recreate
```

Free on the fake embedder. On the `openai` path it re-embeds and re-bills every
chunk, because the collection the content-hash skip compares against is the one
being dropped — `make ingest-dry` prices it first. Sparse encoding itself costs
nothing: it is arithmetic over the corpus, not a provider call.

Querying a pre-M2 collection aborts with an explicit message instead of quietly
degrading to dense-only. A hybrid system that silently stops being hybrid loses
the recall it was built for and reports nothing.

### Asking a question

The retrieval primitives — BM25 encoding, both search paths on the store, RRF
fusion — are in place and unit-tested offline. The commands below are the agreed
operator surface over them (`python -m production_rag.retrieval`, logs on stderr,
a JSON object as the last line of stdout, exit `0`/`1`/`2` exactly as the ingest
job does). Until that entry point lands they fail with a module-not-found error,
which is the correct outcome.

```bash
make retrieve-fake QUERY="how does reciprocal rank fusion work"
make retrieve-fake QUERY="QDRANT__SERVICE__GRPC_PORT" MODE=sparse
make retrieve-fake QUERY="what is a cross-encoder" TOPK=5
```

```powershell
.\scripts\retrieve.ps1 -Query "how does reciprocal rank fusion work"
.\scripts\retrieve.ps1 -Query "QDRANT__SERVICE__GRPC_PORT" -Mode sparse
```

`MODE` / `-Mode` accepts `hybrid` (default), `dense` or `sparse`. The
single-branch modes exist to attribute a regression: a fused score that never
beats its best branch is a fusion problem, not a retriever problem.

### fake vs openai — what each path is good for

The embedder used at query time **must** match the one that built the
collection. Nothing detects a mismatch: both produce 1536 dimensions, the search
succeeds, and the hits are confidently ranked noise.

| | `--embedder fake` | `--embedder openai` |
|---|---|---|
| API key | none | `OPENAI_API_KEY`, from `.env` or the environment |
| Network | none | HTTPS per batch of chunks, and per query |
| Cost | zero | per chunk at ingest, per query at retrieval |
| Dense vectors | deterministic hashes of the text — same input, same vector | `text-embedding-3-small`, 1536-d |
| Dense branch | semantically meaningless; a paraphrase match is luck | real semantic retrieval |
| **Sparse branch** | **genuinely lexical** — BM25 weights are computed from the text in pure Python, no model involved | identical; the embedder does not touch it |
| Runs in CI | yes | no |
| Quality claims | never | with the corpus, chunking and k values stated |

The row that surprises people is the sparse one. The fake path is not a
simulation of half the system: the lexical branch is the real thing, so
exact-token retrieval can be exercised and measured with no credentials at all.
Only the dense side is a stand-in.

### Scoring the golden set

```bash
make eval-hit-fake       # source-level hit@k over data/eval/golden.jsonl
```

```powershell
.\scripts\eval_hit.ps1 -PerBranch
```

`scripts/eval_hit.py` asks one question per golden item: did any retrieved chunk
come from a labelled document? That is `hit@k` — coarser than `recall@k`, and
the only metric the current document-level labels support.

It reports and never gates. No thresholds, no non-zero exit on a low score:
gating a 14-item seed set would be theatre, and the harness with Ragas metrics
and a CI gate is M6.

**How to read the number.** On a `fake`-embedded collection it is a plumbing
assertion — the corpus is indexed, both branches return, fusion orders, and the
payload paths match the labels. Only the sparse branch's contribution reflects
anything about retrieval. On `openai` it is a real measurement over 14 items,
which is to say a smoke test with error bars a whole document wide: one item
moves `hit@5` by seven points. Neither number is a quality claim, and neither
should ever be quoted without the embedder that produced it.

If `hit@k` reads exactly `0.00` at every k, the labels and the stored paths
disagree — almost always because ingest ran with `SOURCE=data/raw/sample`, which
strips the `sample/` prefix every label expects.

Details: [runbook](docs/runbook.md#retrieve-m2) for operations and failure modes,
[architecture](docs/architecture.md#retrieval-flow-m2--live) for the fusion
mechanics, [evaluation](docs/evaluation.md#status-after-m2) for what is and is
not measured, [ADR 0001](docs/adr/0001-hybrid-qdrant.md) — now **Accepted** — for
why hybrid on one collection.

## Rerank (M3)

M3 adds the stage between fusion and everything downstream: a **cross-encoder
that rescores the fused candidates** and keeps the best few. It is opt-in,
fail-open, and ships with three providers — one of which is deliberately fake.

**What is live: reranking. What is still not: generation.** No `POST /v1/query`,
no LLM call, no answer, no citations. Reranking reorders passages; it is the last
stage of retrieval, not the first stage of an answer. That is M4.

### Why a reranker after RRF

RRF is scale-free by construction — it sums `1/(k + rank)` over the branches that
returned a document, so cosine similarity and BM25 scores never have to be
calibrated against each other. That property costs magnitude: a document that is
overwhelmingly the best match contributes exactly what a merely adequate one at
the same rank contributes.

Worse, neither branch ever reads the query and the passage *together*. The dense
branch compares two independently produced vectors; the sparse branch sums
per-term weights. The predictable result is the failure everyone recognises from
a first RAG build: the top 12 are all on-topic and the passage that actually
answers the question is fourth.

A cross-encoder scores the pair in one forward pass, with attention across both
texts. It is accurate precisely because it cannot be precomputed — which is also
why it runs over a shortlist instead of being the retriever.

```
  fused candidates (RRF, recall-oriented)
        │  40
        ▼
  ┌─────────────────────────────────────────┐
  │ CROSS-ENCODER — score(query, passage)   │   fake | local | cohere
  │ one forward pass per pair, ties break   │   on error: fusion order,
  │ on the fusion rank so runs repeat       │   reported, never silent
  └────────────────┬────────────────────────┘
        │  6
        ▼
  hits[] + pre_rerank_rank + rerank_score
```

**Retrieval owns recall; rerank owns precision at the top.** The stage never
queries Qdrant and never introduces a document fusion did not return.

### `input_top_k` (40) > `top_k` (6), on purpose

The reranker is fed more than it keeps. Handing it exactly the number that
survives makes it a no-op sorter of an already-final list; handing it 40 lets it
lift a passage buried at rank 30 into position 2 — the case the stage exists for.

Cost is linear in `input_top_k` (40 candidates = 40 forward passes, or 40
passages on the wire), so it is the price dial. The flip side is a hard ceiling:
with rerank on, a chunk outside that window is unreachable by construction, which
is why every result reports how many candidates the stage actually saw.

### fake vs local vs cohere

| `--rerank` | Model | Needs | What it is for |
|---|---|---|---|
| `off` *(default)* | — | — | plain M2 behaviour |
| `fake` | query-term overlap, pure Python | nothing: no key, no download, no network | **plumbing only.** CI, offline laptops, contract tests |
| `local` | `BAAI/bge-reranker-base` on CPU | `sentence-transformers` + a ~1.1 GB one-time download | real reranking, no per-query spend, no passage leaves the machine |
| `cohere` | `rerank-english-v3.0` | the `cohere` package + `COHERE_API_KEY` | hosted swap when a deployment cannot host a model |
| `auto` | from `rerank.provider` | depends | the YAML is the switch for a deployment |

`fake` carries the same warning as the fake embedder, one stage later. It is
deterministic — so tests can assert an order — and it **models nothing**: it
scores lexical coverage, a cruder version of what BM25 already did. Under it the
flag path, the candidate arithmetic, the fail-open branch, the emitted fields and
the JSON contract are all genuinely exercised. Its *ordering* is not.

Said plainly: **rerank plumbing runs everywhere; rerank quality exists only on
`local` or `cohere`.** No ordering number produced by `fake` appears anywhere in
this repository, and none should.

### `fail_open` — degrade the ordering, never the availability

If the reranker errors or times out, the stage logs it and returns **fusion
order**; the query succeeds. The un-reranked result is correct in kind, only
ordered worse — availability beats a few points of nDCG.

That is the opposite of how a *missing capability* is treated: a collection with
no `sparse` vector aborts, because the system would otherwise be silently
unhybrid. A missing improvement is not a missing capability.

Never silent, either way. Every result carries `rerank: {applied, reranker,
candidates, error}`, present even when nothing reranked, and a reranked hit adds
`pre_rerank_rank` and `rerank_score` — so "the cross-encoder pulled this from
rank 27 to rank 2" is measurable rather than felt. `fail_open: false` exists for
a deployment that would rather fail the request, and means a provider outage is
an outage.

### Running it

```bash
docker compose run --rm api python -m production_rag.retrieve \
    --query "what is a cross-encoder" --rerank fake     # plumbing, no credentials
docker compose run --rm api python -m production_rag.retrieve \
    --query "what is a cross-encoder" --rerank local    # the real thing
```

```powershell
.\scripts\retrieve_rerank.ps1 -Query "what is a cross-encoder"
.\scripts\retrieve_rerank.ps1 -Query "what is a cross-encoder" -Rerank local
.\scripts\retrieve_rerank.ps1 -Query "what is a cross-encoder" -Rerank local -Compare
```

`-Compare` runs the query twice — rerank off, then on — because an ordering on
its own is not evidence and a delta is. `scripts/retrieve.ps1` stays the plain M2
surface and is unchanged.

### What it is worth, measured honestly

No pre/post number is quoted here. The procedure that produces one is in
[evaluation.md](docs/evaluation.md#pre--and-post-rerank-hitk): score the golden
set with the stage off, then on, changing nothing else, and compare `hit@k` at
`k ≤ rerank.top_k` — comparing above that cut shows a fake regression, because
the reranked run was only asked for six hits. `hit@k` also under-reports the
stage: moving the answer from rank 5 to rank 1 changes `hit@1` and nothing else.
`nDCG` is the metric that sees a reordering, and it needs graded chunk-level
labels — M6.

On a 14-item set with a `fake` embedder, that comparison measures plumbing twice.

Details: [ADR 0004](docs/adr/0004-rerank-cross-encoder.md) for the decision and
the alternatives, [runbook](docs/runbook.md#rerank-m3--off-by-default-opt-in-per-run)
for operations, model download and failure modes,
[architecture](docs/architecture.md#why-rerank-runs-after-rrf-and-not-instead-of-it)
for where the stage sits, [data model](docs/data-model.md#the-two-m3-fields-and-why-they-are-optional-keys)
for the two fields a reranked hit gains.

## License

[MIT](LICENSE).
