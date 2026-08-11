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

## Retrieve (M2)

Retrieval is a batch command, like ingest. It takes a question, queries the
dense and sparse branches against the collection, fuses the two ranked lists
with RRF, and prints the hits.

**It returns passages, not an answer.** There is no LLM call anywhere in this
path, no reranking (M3) and no citation rendering (M4). If you are looking for
`POST /v1/query`, it does not exist yet.

> The retrieval **primitives** are in place — BM25 encoding, both search paths on
> the store, RRF fusion, all unit-tested offline. The command below is the agreed
> operator surface over them: `python -m production_rag.retrieval`, flags mapping
> one-to-one onto the config keys, logs on stderr and a JSON object as the last
> line of stdout, exit `0` ok / `1` the run failed / `2` the invocation or the
> collection is wrong — the same contract as the ingest job. Until that entry
> point lands, these commands fail with a module-not-found error, which is the
> correct outcome and not a stack problem.

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
| Ask one question, offline | `make retrieve-fake QUERY="…"` |
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
