# Runbook

Operational procedures for the local stack. Everything goes through
`docker compose` (or the `Makefile` / `scripts/*.ps1` wrappers, which are
thin shims over the same commands).

## Start the stack

```bash
make up                 # or: docker compose up -d --build
# Windows without make:
.\scripts\up.ps1
```

Wait for health, then verify:

```bash
make health             # API /health + Qdrant /readyz
python scripts/smoke_health.py
.\scripts\health.ps1    # PowerShell equivalent
```

Expected: both probes return 2xx. The Qdrant dashboard is at
http://localhost:6333/dashboard, API docs at http://localhost:8000/docs.

Liveness is the default probe because it is the only surface that must answer
with no dependency up — a failure there means the process is wrong, not the
stack around it. Readiness reports dependency state and is opt-in:

```bash
make health-ready
python scripts/smoke_health.py --ready
.\scripts\health.ps1 -Ready
```

## Ingest (M1)

Ingest is a batch job, not an endpoint. It walks a corpus directory, chunks,
embeds, and upserts into Qdrant, then prints counts. It is safe to re-run: an
unchanged chunk hashes the same and is skipped before the embedding call.

### The fake embedder is the default, on purpose

`--embedder fake` maps text to a vector by hashing it. No API key, no network
call, no spend, and identical text always yields an identical vector. That makes
the whole path — walk, chunk, embed, upsert, count — runnable on a laptop with
no credentials and in CI.

Its vectors carry no semantics. Any similarity or recall number measured against
a fake-embedded collection is noise. Use it to prove the plumbing works; never
to make a quality claim.

### Windows, first run

The stack must already be up (`.\scripts\up.ps1`). The job runs inside the `api`
container, so `QDRANT_URL` resolves to the compose hostname and nothing needs to
be installed on the host.

```powershell
# 1. Dry run first: walks and chunks, writes nothing. Cheap sanity check that
#    the corpus is visible inside the container and the counts look sane.
.\scripts\ingest.ps1 -DryRun

# 2. Real ingest with the fake embedder — offline, free.
.\scripts\ingest.ps1

# 3. Confirm the collection exists and is populated.
.\scripts\health.ps1
Invoke-RestMethod http://localhost:6333/collections/production_rag |
    Select-Object -ExpandProperty result |
    Select-Object status, points_count, vectors_count
```

Expected after step 2: `points_count` greater than zero, `status` = `green`, and
`health.ps1` reporting `collection 'production_rag' is present.`

`make` equivalents, for machines that have it:

```bash
make ingest-dry                              # walk + chunk, write nothing
make ingest-fake                             # offline, no API key
make ingest-sample                           # real embeddings, needs the key
make ingest-fake SOURCE=data/raw/my-corpus   # any corpus root
```

### `SOURCE` / `-Source` is the corpus root

Payload `source_path` values are relative to whatever root is passed, and its
first path segment becomes the filterable `source` field. Ingesting `data/raw`
gives `sample/08-bm25-vs-dense.md` with `source: "sample"`; ingesting
`data/raw/sample` gives `08-bm25-vs-dense.md` with `source: "root"`.

Both are valid provenance and only one matches `data/eval/golden.jsonl`, which
labels `sample/…`. The default is `data/raw` for that reason. Changing it does
not error — the eval labels simply stop matching, which reads as a retrieval
regression.

`data/raw/README.md` sits at that root and is excluded via
`ingest.exclude_globs`, so the layout documentation does not become a corpus
document.

### Real embeddings

`data/raw/` is bind-mounted read-only into the container, so a new corpus
directory is visible without a rebuild — but it is **not** free to ingest.
Embedding is the only stage in this system that costs money per document.

```powershell
Copy-Item .env.example .env       # then put the key in .env; never on the CLI
.\scripts\ingest.ps1 -Embedder openai
```

The key is read from the environment or from the gitignored `.env` that Compose
loads. Passing it as a command-line argument would put it in shell history and
in `docker inspect` output, so the scripts have no flag for it.

Cost control before a large run: `-DryRun` reports the chunk count, and chunks
times average chunk size is the token bill. Check that number before spending
it, not after.

### M2 migration — an M1 collection must be rebuilt

M1 created the collection with **one** named vector, `dense`. It did not declare
an empty `sparse` vector; earlier revisions of these documents said it did, and
that was wrong about the shipped code. Hybrid retrieval therefore cannot run
against a collection left over from M1, and the collection cannot be upgraded in
place.

Symptom: the retrieve command exits `2` saying the collection has no `sparse`
named vector. It aborts rather than falling back to dense-only, because a hybrid
system that quietly stops being hybrid loses exactly the recall it was built for
and reports nothing.

```bash
make reingest-fake          # drop + re-ingest with both vectors. Free.
```

```powershell
.\scripts\ingest.ps1 -Recreate
```

> **Destructive, and on the `openai` path it costs money.** The rebuild drops
> the collection, so the incremental content-hash skip has nothing to compare
> against and **every chunk is re-embedded and re-billed**. Sparse encoding
> itself is free — it is arithmetic over the corpus, not a provider call. Run
> `make ingest-dry` first and price the chunk count.

Confirm both vectors afterwards:

```powershell
Invoke-RestMethod http://localhost:6333/collections/production_rag |
    Select-Object -ExpandProperty result |
    Select-Object -ExpandProperty config |
    Select-Object -ExpandProperty params
```

Expect `vectors` to hold `dense` and `sparse_vectors` to hold `sparse`. A
collection with only `dense` is still an M1 collection.

### Re-ingest and recreate

Re-running ingest over the same corpus is cheap: unchanged chunks are skipped by
content hash, and changed ones upsert over their own deterministic point ids
rather than duplicating.

> **Destructive:** `-Recreate` drops the collection and every vector in it
> before ingesting. There is no undo, and the cost of rebuilding is a full
> re-embed of the corpus — real money on the `openai` path. The script prompts
> for confirmation.

```powershell
.\scripts\ingest.ps1 -Recreate
```

Reach for it when the collection's vector dimensionality or its named-vector
layout changed, since those cannot be altered in place. For a chunking change,
a plain re-ingest is enough — but note it orphans the old chunk ids, which
invalidates eval labels and previously stored citations.

### Ingest failures

**`corpus directory not found`.**
The path is resolved from the repo root, not the current directory. Use
`data/raw/<name>`, and confirm the directory is under `data/` — anything outside
it is not mounted into the container.

**Zero files walked, exit 0.**
The corpus has no file with an extension in
`configs/default.yaml → ingest.include_extensions` (`.md`, `.markdown`, `.txt`).
Skipped files are logged with a count; read that count before assuming the
corpus is empty.

**`401` or `429` from the embedding provider.**
`401` means the key is absent or wrong — check `.env` is present and Compose
loaded it (`docker compose config` shows the resolved environment, with the
value masked in recent versions; do not paste its output into a ticket
regardless). `429` is rate limiting: the job retries with backoff and then
aborts rather than upserting a partial batch.

**`vector size mismatch` on upsert.**
The collection was created with a different embedding dimensionality. Nothing is
written. Either point at a different collection or `-Recreate`.

**Ingest succeeded, `health.ps1` still says the collection is missing.**
The job wrote to a different collection than the one the probe checks. Compare
`QDRANT_COLLECTION` in the container against the `-Collection` argument on the
probe.

## Retrieve (M2 + M3)

Retrieval is a batch command, like ingest. It takes a question, queries the
dense and sparse branches against the collection, fuses the two ranked lists
with RRF, optionally reranks them with a cross-encoder (M3, off by default), and
prints the hits.

**It returns passages, not an answer.** There is no LLM call anywhere in this
path and no citation rendering — reranking reorders passages, it does not answer.
For an answer with citations, use [Query (M4)](#query-m4); this command remains
the way to inspect what retrieval produced without spending a generation call,
and it is what the eval script drives.

The job is `python -m production_rag.retrieve`: flags map one-to-one onto the
config keys, logs go to stderr, a single JSON object is the last line of stdout,
and exit codes are graded `0` ok / `1` the run failed / `2` the invocation or the
collection is wrong — the same contract as the ingest job.

```bash
make retrieve-fake QUERY="how does reciprocal rank fusion work"
make retrieve-fake QUERY="QDRANT__SERVICE__GRPC_PORT" MODE=sparse
make retrieve-fake QUERY="what is a cross-encoder" TOPK=5
```

```powershell
.\scripts\retrieve.ps1 -Query "how does reciprocal rank fusion work"
.\scripts\retrieve.ps1 -Query "QDRANT__SERVICE__GRPC_PORT" -Mode sparse
.\scripts\retrieve.ps1 -Query "what is a cross-encoder" -TopK 5 -OnHost
```

Prerequisites: the stack is up, and the collection was built by M2 (see
[the migration](#m2-migration--an-m1-collection-must-be-rebuilt)).

### The embedder must match the collection

`--embedder` / `-Embedder` selects how the **query** is embedded. It must be the
same one that embedded the corpus.

Querying a `fake`-embedded collection with `openai` — or the reverse — compares
two unrelated vector spaces. Nothing errors: both produce 1536 dimensions, the
kNN search succeeds, and you get twelve confidently ranked, meaningless hits.
The sparse branch keeps working, which makes the result look *partly* sane and
is the reason this is worth stating: the failure hides.

There is no automatic check. If retrieval quality collapses for no reason,
verify which embedder built the collection before touching fusion weights.

### `MODE` attributes a regression to a branch

| Mode | Queries | Use it when |
|---|---|---|
| `hybrid` (default) | both branches, fused | normal operation |
| `dense` | embedding kNN only | asking whether the dense side alone finds it |
| `sparse` | BM25 only | asking whether the lexical side alone finds it |

Each hit carries the rank it held in every branch that returned it. A hit ranked
by `sparse` and not by `dense` is a document the embedding branch never returned
— hybrid retrieval doing the job it was adopted for, visible per hit rather than
inferred.

### Rerank (M3) — off by default, opt in per run

```bash
make retrieve-fake QUERY="how does reciprocal rank fusion work"                  # M2 behaviour
docker compose run --rm api python -m production_rag.retrieve \
    --query "how does reciprocal rank fusion work" --rerank fake                 # plumbing
docker compose run --rm api python -m production_rag.retrieve \
    --query "how does reciprocal rank fusion work" --rerank local                # real
```

```powershell
.\scripts\retrieve_rerank.ps1 -Query "how does reciprocal rank fusion work"
.\scripts\retrieve_rerank.ps1 -Query "how does reciprocal rank fusion work" -Rerank local
.\scripts\retrieve_rerank.ps1 -Query "…" -Rerank local -Compare   # side by side with rerank off
```

| `--rerank` | What it does | Needs |
|---|---|---|
| `off` (default) | nothing; fusion order, exactly M2 | — |
| `fake` | deterministic query-term overlap | nothing — no key, no download, no network |
| `local` | `BAAI/bge-reranker-base` cross-encoder on CPU | `sentence-transformers`, plus a one-time ~1.1 GB model download |
| `cohere` | hosted `rerank-english-v3.0` | the `cohere` package and `COHERE_API_KEY` |
| `auto` | whatever `rerank.enabled` / `rerank.provider` say in the YAML | depends on the provider it resolves to |

**`fake` is plumbing, not quality.** It re-scores by the share of query terms a
passage contains, which is a cruder version of what BM25 already did — so it
mostly reproduces the fusion order and will never do what a cross-encoder does.
Use it to prove the stage is wired: the flag path, the candidate counts, the
fail-open branch, the emitted fields. Never quote an ordering it produced. The
providers that mean something are `local` and `cohere`.

The two questions worth asking of any reranked run come straight out of its JSON:

```powershell
# did the stage run, over how many candidates, and did it fail open?
.\scripts\retrieve_rerank.ps1 -Query "…" -Rerank local -Json |
    ConvertFrom-Json | Select-Object -ExpandProperty rerank

# what did it actually move?
.\scripts\retrieve_rerank.ps1 -Query "…" -Rerank local -Json |
    ConvertFrom-Json | Select-Object -ExpandProperty hits |
    Select-Object rank, pre_rerank_rank, rerank_score, source_path
```

A hit whose `pre_rerank_rank` is far below its `rank` is the stage earning its
latency. If every `pre_rerank_rank` equals its `rank`, reranking ran and changed
nothing — which is information, not a bug.

`rank` and `score` disagreeing after a reranked run is expected: `score` stays
the fused RRF score and is deliberately not overwritten by `rerank_score`, so the
top hit need not carry the highest `score`.

### `input_top_k` is the ceiling, and the price

`rerank.input_top_k` (default 40) is how many fused candidates the reranker sees;
`rerank.top_k` (default 6) is how many survive. Feeding it more than it keeps is
the point — that is what lets it lift a passage from rank 30 to rank 2.

Two operational consequences:

- **Cost and latency are linear in `input_top_k`.** 40 candidates is 40 forward
  passes on CPU, or 40 passages on the wire. This is the dial to turn when a
  reranked query is too slow.
- **Anything outside the window is unreachable.** With rerank on, a relevant
  chunk that fusion ranked 45th cannot come back. So on a bad reranked result,
  check `rerank.candidates` in the output *before* blaming the ordering, and
  compare against the same query with `--rerank off`.

Raising `input_top_k` above `retrieval.dense_top_k + retrieval.sparse_top_k`
buys nothing: fusion can only hand over what the branches returned.

### The local reranker downloads a model

`--rerank local` pulls `BAAI/bge-reranker-base` (~1.1 GB) from Hugging Face on
first use and caches it under the HF cache directory. Inside a container that
cache is **not** persisted by default, so every fresh container re-downloads it.

- First run is slow and needs network. Later runs on the same container are not.
- For repeated use, mount the cache (`~/.cache/huggingface` on the host) or bake
  the weights into the image. A production container that downloads a
  cross-encoder on its first request is a latency incident, not a cold start.
- No key is involved, and no passage leaves the machine — that is the point of
  the local provider.
- The model is loaded lazily, on the first query rather than at construction, so
  an argument error never pays for a download.

### Rerank failures

**`--rerank local` fails with "needs sentence-transformers", exit 2.**
The optional dependency is not installed in the interpreter running the command.
Install it (it ships in the `rag` extra) or use `--rerank fake` / `--rerank off`.
This is a usage error, not a runtime fault: retrying cannot fix it, and silently
falling back to `fake` would fabricate a quality claim.

**`--rerank cohere` fails with "needs an API key", exit 2.**
`COHERE_API_KEY` is unset in the environment the command runs in — inside the
container that means it is not in `.env`. It is checked before any query runs, so
the failure is a usage error rather than a mid-request outage. Never pass the key
as a flag: it would land in shell history and in `docker inspect`.

**The run succeeds but `rerank.applied` is `false` and `rerank.error` is set.**
This is `fail_open: true` doing its job: the reranker errored or timed out, the
fusion order was returned, and the query succeeded. A `rerank_failed_open`
warning is in the logs with the provider and the error. Serving un-reranked hits
is a degradation, not an outage — but a *persistent* one means ordering quality
has quietly dropped, so treat a steady stream of these as an incident.

To make the same failure loud instead, set `rerank.fail_open: false` and accept
that a provider outage becomes a failed request.

**Reranking is much slower than retrieval.**
Expected. A cross-encoder over 40 candidates is the slowest stage in the path —
hundreds of milliseconds on CPU against tens for the Qdrant round trip. Lower
`input_top_k`, or leave rerank off for latency-sensitive work.

**Ordering looks wrong with `--rerank fake`.**
Not a defect. `fake` models nothing. Compare with `--rerank off` and, if you need
a real answer, `--rerank local`.

### Score the golden set

```bash
make eval-hit-fake            # source-level hit@k, fake embedder
make eval-hit-sample          # real embeddings, costs money
```

```powershell
.\scripts\eval_hit.ps1
.\scripts\eval_hit.ps1 -PerBranch     # also scores dense-only and sparse-only
```

Reports and never gates. Read the number with the embedder attached: on `fake`
the dense branch is hash noise so the score is a plumbing assertion — though the
sparse branch is genuinely lexical even there, because BM25 weights come from
the text. Full caveats in [evaluation.md](evaluation.md#reading-a-hitk-number-honestly).

To ask what reranking is worth, score the same golden set twice — once with
rerank off, once with the provider under test — and compare `hit@k` at small `k`.
The comparison is only meaningful on `--embedder openai` with `--rerank local` or
`--rerank cohere`; a `fake` embedder plus a `fake` reranker measures plumbing
twice. See [evaluation.md](evaluation.md#pre--and-post-rerank-hitk).

### Retrieval failures

**`collection has no named vector 'sparse'`, exit 2.**
The collection predates M2. Rebuild it:
[M2 migration](#m2-migration--an-m1-collection-must-be-rebuilt). Retrying cannot
fix this, which is why it exits 2 rather than 1.

**Collection not found, exit 2.**
Nothing has been ingested into the name being queried. Compare
`QDRANT_COLLECTION` in the container against `--collection`, then run
`make ingest-fake`.

**Zero hits on every query.**
Check `points_count` first — an empty collection retrieves nothing:

```powershell
Invoke-RestMethod http://localhost:6333/collections/production_rag |
    Select-Object -ExpandProperty result |
    Select-Object status, points_count
```

If it is populated, the next suspect is `retrieval.score_threshold`. It is
applied to the **fused RRF score**, whose scale comes from rank positions, not
similarity: with two branches and `k = 60` the theoretical maximum is
`2/61 ≈ 0.0328`. A threshold set at anything resembling a cosine similarity —
`0.7`, say — filters out everything, always. Default is `0.0`.

**Hits look plausible but ranking is nonsense.**
Embedder mismatch, above. Verify which embedder built the collection.

**The sparse branch returns nothing for a specific query.**
Legitimate when every query term is a stopword or absent from the corpus
vocabulary — there is no lexical signal to match. Dense results are still
returned and the empty branch is reported. Compare against
`make retrieve-fake QUERY="..." MODE=sparse` to confirm it is the query and not
the index.

**Exact-token queries regress after adding documents (IDF drift).**
BM25 term weights are computed at ingest time from corpus-wide statistics, so
they reflect the corpus as of the last **full** ingest. Adding documents shifts
the true IDF while the stored weights keep the old values, and incremental
ingest makes it worse rather than better: it skips unchanged chunks, which are
precisely the ones whose weights are stale.

Immaterial at this corpus size. After a large corpus change, re-ingest fully
before quoting any sparse-branch number:

```bash
make reingest-fake
```

## Query (M4)

M4 is the first surface that returns an **answer** rather than passages. Two ways
in, one pipeline behind both:

- `POST /v1/query` — the HTTP endpoint, for clients.
- `python -m production_rag.query` — the batch CLI, for operators. Logs to
  stderr, one JSON object as the last line of stdout, exit codes graded `0` ok /
  `1` the run failed / `2` the invocation was wrong — the same contract as the
  ingest and retrieve jobs.

Prerequisites are the same as retrieval: the stack is up, and the collection was
built by M2 or later with the embedder you intend to query with.

### The endpoint

```bash
curl -s localhost:8000/v1/query \
  -H 'content-type: application/json' \
  -H 'X-Request-ID: local-debug-1' \
  -d '{"question": "how does reciprocal rank fusion work"}' | jq
```

```powershell
$body = @{ question = "how does reciprocal rank fusion work"; llm = "fake" } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/v1/query `
    -ContentType 'application/json' -Body $body |
    Select-Object answer, refused, refusal_reason
```

Request fields: `question` (required, 1–8000 characters, whitespace-stripped),
`mode` (`dense` / `sparse` / `hybrid`), `rerank` (`off` / `auto` / `fake` /
`local` / `cohere`), `llm` (`fake` / `openai`, **default `fake`**), `debug`.
Anything omitted falls back to `configs/default.yaml`. Unknown fields are
rejected with 422 rather than ignored.

The response is four fields — `answer`, `citations`, `refused`, `refusal_reason`
— and nothing else. Field-by-field spec in the
[data model](data-model.md#queryresponse--the-shape-post-v1query-returns).
Timings, hit counts, the collection name and the model are on the **library**
result and in the logs, not on the endpoint.

`llm` defaults to `fake`. A request that does not ask for `openai` gets the
deterministic offline path — no key, no spend, and no answer worth reading. That
default is on purpose: an unrequested bill is worse than an obviously fake
answer. It also means a demo that forgets the field is demonstrating the schema.

There is no `filters` field yet. `retrieval.filters.allowed_fields` in the config
is still declared-only.

### `X-Request-ID` is the debugging handle

Send one and it comes back on the `X-Request-ID` response header, and it is on
every log line the request emitted. Send nothing (or something malformed) and a
fresh UUID4 is generated and used the same way. It is deliberately not in the
response body.

One question is now an embedding call, two Qdrant searches, possibly a rerank
provider call and an LLM call. That id is what stitches them back together:

```powershell
docker compose logs api | Select-String "local-debug-1"
```

Quote it in any bug report. Without it, correlating a slow answer with the stage
that caused it is guesswork.

### The CLI

```bash
make query-fake QUERY="how does reciprocal rank fusion work"        # offline, free

docker compose run --rm api python -m production_rag.query \
    --question "how does reciprocal rank fusion work" --llm fake     # same thing, explicit

docker compose run --rm api python -m production_rag.query \
    --question "how does reciprocal rank fusion work" \
    --llm openai --rerank local                                      # billed
```

Flags: `--question` (required), `--mode`, `--rerank`, `--llm` (default `fake`),
`--debug`, `--log-level`. The last line of stdout is one JSON object —
`{"ok": true, …}` with the response fields, or
`{"ok": false, "error": …, "error_type": …}` on failure.

`make query-fake` requires `QUERY="…"` and refuses without it (exit 2), the same
guard `retrieve-fake` carries: a query command with a default question silently
scores a question nobody asked. There is no `make` target for the billed path —
run the CLI explicitly with `--llm openai`, so spending is never a habit.

The CLI exists for the same reason the retrieve command does: it exercises the
whole path with no HTTP client, no port and no container networking in the way,
and it returns the same object the endpoint does.

### `fake` vs `openai` — what each generator is for

| | `--llm fake` *(default)* | `--llm openai` |
|---|---|---|
| API key | none | `OPENAI_API_KEY`, from `.env` or the environment |
| Network | none | HTTPS per query |
| Cost | zero | per prompt + completion token |
| Answer | deterministic extractive stitching of the top passages | `generation.model` (`gpt-4o-mini`), temperature 0.1 |
| `[n]` markers | emitted and resolvable — the contract is real | emitted and resolvable |
| Refusal checks | both exercised | both exercised |
| Reasoning, synthesis across passages | **none** | real |
| Runs in CI | yes | no |
| Quality claims | never | with corpus, chunking, k values and model stated |

**`fake` is the contract, not the answer.** It genuinely exercises budgeting,
prompt rendering, marker resolution, invalid-marker stripping, both refusal
checks, the response schema, the per-node timings and the endpoint — offline,
deterministic, free. What it cannot do is reason: it will not synthesise across
two passages and its prose is not meant to be read. Same warning as the fake
embedder and the fake reranker, one stage later. Never quote an answer it
produced.

The embedder still has to match the collection, independently of the generator. A
`fake`-embedded collection answered by `--llm openai` produces a fluent answer
grounded in hash noise — the most expensive way to be wrong this system offers.

### Reading a refusal

A refusal is a **200** with `refused: true`, `citations: []`, and a
`refusal_reason` from a closed set:

| `refusal_reason` | What happened | LLM called |
|---|---|---|
| `no_evidence` | retrieval returned nothing to ground an answer in | no |
| `model_abstained` | the model emitted the `INSUFFICIENT_CONTEXT` sentinel | yes |
| `no_citations` | the answer resolved to no citation at all, with `require_citation: true` | yes |
| `empty_answer` | the model returned only whitespace | yes |

```powershell
docker compose run --rm api python -m production_rag.query `
    --question "what is the airspeed velocity of a swallow" --llm fake |
    Select-Object -Last 1 | ConvertFrom-Json |
    Select-Object refused, refusal_reason, answer
```

Branch on `refused` and `refusal_reason`, never on the message text — the message
is `generation.citations.refusal_message` and it is meant to be changed per
deployment.

`no_evidence` is the only reason that costs nothing: the model was never called.
The other three mean the model ran and its output was not servable.

**Refusals spiking is a retrieval investigation, not a prompt one.** The answer
path can only be as good as what was retrieved. Order of checks: run the same
question through `make retrieve-fake QUERY="…"` and see whether the passage comes
back at all; confirm the embedder that built the collection matches the one
querying it; then check `retrieval.score_threshold` — it is applied to the fused
RRF score, whose maximum with two branches is `≈ 0.0328`, so anything set at a
cosine-like `0.7` refuses everything, always.

### Reading citations

```powershell
docker compose run --rm api python -m production_rag.query --question "…" --llm fake |
    Select-Object -Last 1 | ConvertFrom-Json |
    Select-Object -ExpandProperty citations |
    Select-Object marker, rank, chunk_id, source_path
```

The `[n]` in the answer is an ordinal into **this response's** prompt blocks —
`[2]` is the second passage this request rendered, and it identifies nothing
tomorrow. What is durable is `chunk_id`. Clients store citations; they never
store markers.

Markers are not renumbered: an answer that cites only `[3]` keeps `[3]`, so the
answer still lines up with the prompt that produced it.

Two diagnostics live on the **library** result (`run_query(...).to_dict()`, and
the CLI JSON when A3 widens it) rather than on the HTTP response:

- `invalid_markers` — out-of-range `[n]` the model emitted. They are stripped
  from the answer rather than left as footnotes that go nowhere. One occasionally
  is model noise; a steady stream is a model or prompt problem.
- `uncited_claims` — sentences of 24+ characters carrying no marker. Reported,
  never fatal: refusing on one uncited transition sentence would make the
  guardrail useless. It is citation coverage measured on every request.

### Query failures

**422, question rejected.**
Empty or whitespace-only after stripping, or longer than 8000 characters. Nothing
was retrieved and no token was spent — validation runs before the pipeline.

**422 on an unknown field.**
The request body carries a field the schema does not define (a typo, or a control
that does not exist yet, such as `filters` or `top_k`). Rejected rather than
ignored: a silently dropped control answers a different question than the one
asked.

**503 from `POST /v1/query`.**
The query pipeline module is not present in this checkout. That is a
split-milestone state, not a runtime fault: the route fails honestly instead of
growing a second implementation inside itself.

**200 with `refused: true` on a question the corpus clearly covers.**
Retrieval, not generation. See the refusal section above.

**The run fails with a provider error.**
Rate limit, timeout, or an upstream error. The OpenAI SDK does the bounded
retrying (`generation.max_retries`, `generation.timeout_seconds` are handed to
it), and then the request fails. Unlike the reranker this stage is deliberately
**not** fail-open: there is no degraded answer to serve, only an ungrounded one.
The CLI prints the exception *type*, never the provider message, because an SDK
error can carry the request that caused it — which here is the whole prompt.

**`OPENAI_API_KEY` missing with `--llm openai`.**
Fails before any query runs, as a configuration error. Put the key in the
gitignored `.env` that Compose loads; never pass it as a flag, which would put it
in shell history and in `docker inspect` output. Falling back to `fake` would
fabricate an answer, so it does not.

**Answers are slow.**
Read `latency_ms` on the library result before anything else. `generate`
dominating is normal and is a model / `max_output_tokens` question. `rerank`
dominating means `input_top_k` is the dial (40 candidates is 40 forward passes).
`retrieve` dominating points at Qdrant, not at the LLM.

**Answer text is fine but the citations look thin.**
Compare `hits_used` against `hits_retrieved`. A gap means the context budget
(`generation.prompt.max_chunks_in_prompt`, `generation.max_context_tokens`)
dropped the tail — and retrieval order is truncation order, so the reranker
decided what the model was allowed to cite. Truncation is also logged as
`context_truncated` with kept and dropped counts.

> **Never turn on `observability.logging.log_prompts` or `log_retrieved_text` in
> a deployment.** The prompt contains corpus text verbatim, so a log aggregator
> with those enabled becomes a copy of the corpus with none of its access
> controls. They exist for a local debugging session and default to `false`.

## Day-to-day commands

| Task | Command |
|------|---------|
| Status + health | `make ps` |
| Tail logs | `make logs` |
| Rebuild after dependency change | `make build` |
| Recreate API only | `make restart` |
| Stop, keep index | `make down` |
| Run tests in container | `make test` |
| Ingest the sample corpus, offline | `make ingest-fake` |
| Chunk-count a corpus without writing | `make ingest-dry` |
| Rebuild an M1 collection for M2 | `make reingest-fake` |
| Retrieve passages, offline | `make retrieve-fake QUERY="…"` |
| Answer a question, offline | `make query-fake QUERY="…"` |
| Score the golden set | `make eval-hit-fake` |

## Common failures

**`api` unhealthy, Qdrant healthy.**
Check `make logs` for the API container. Most common cause in M0: missing
`OPENAI_API_KEY` when a query path that needs a provider is exercised.
Copy `.env.example` to `.env` and fill it in, then `make restart`.

**Qdrant container restarting in a loop.**
Almost always a corrupted storage volume or a port conflict on 6333/6334.
Check `docker compose logs qdrant`. If the volume is corrupt and the corpus
is re-ingestible, rebuild it: `make clean && make up`, then re-run ingest.

**Port already in use (`8000`, `6333`, `6334`).**
Another compose project or a local dev server is holding the port.
`docker ps` to find it; stop the offender or remap ports in
`docker-compose.yml`.

**Collection missing after `up`.**
The volume survived but ingest has not run (or ran against a different
`QDRANT_COLLECTION`). `scripts/health.ps1` reports this as a warning, not a
failure — an empty stack before ingest is a valid state.

## Destructive operations

```bash
make clean              # docker compose down -v --remove-orphans
.\scripts\down.ps1 -Purge
```

Both delete the `qdrant_storage` volume. The vector index is gone and must be
rebuilt by re-running ingest. This is safe only because `data/raw/` is the
source of truth.

## Upgrading Qdrant

The image tag is pinned (`qdrant/qdrant:v1.13.2`) on purpose. To upgrade:
bump the tag, `make clean` (storage format migrations are one-way), `make
up`, re-ingest, and re-run the eval suite before trusting the new version
(see [evaluation.md](evaluation.md)).
