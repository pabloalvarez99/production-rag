# Evaluation

A RAG system without offline evaluation is a system where every change is a
guess. This document defines what is measured, how, and what number blocks a
merge. The strategy rationale is in [ADR-0003](adr/0003-eval-strategy.md).

## Status after M6

M2 landed a retriever, so the first measurement in this repository became
possible: **source-level `hit@k`**. M3 landed a rerank stage, which makes the
first *comparison* possible — the same metric with the stage off and on. M4
landed answers, which changes what is *measurable* and changes nothing about what
is measured. M5 changed nothing about it either: observability made the request
path report on itself — timings, `invalid_markers`, `hits_used` — and none of
those is a quality number; see
[ops signals are not eval metrics](#ops-signals-are-not-eval-metrics).

M6 is where the strategy in [ADR-0003](adr/0003-eval-strategy.md) stops being a
plan. The ADR is **Accepted** as of this milestone, and the two-tier split is
now the way this repository measures itself. What M6 does *not* do is arm a merge
gate, and the rest of this section is the honest accounting of why.

| Piece | State |
|---|---|
| `data/eval/golden.jsonl` | **committed** — 17-item seed set, document-level labels, explicit `answerable`, 4-item `unanswerable` slice |
| Unified runner, both tiers, one report | **live** — `python -m production_rag.evals.run --tier all`; `make eval-tier1`, `make eval-tier2-fake`, `make eval-all-fake` |
| **Tier 1** — `source_hit@k`, `source_recall@k`, `mrr`, `ndcg@k` | **live** — `production_rag.evals.tier1_retrieval`, over `expected_source_paths` |
| **Tier 1** — per-branch ablation (dense / sparse / fused / +rerank) | **live** — `production_rag.evals.ablation`, `make eval-ablation-fake` |
| Pre- vs post-rerank comparison on the same set | **live** — `--rerank off` then `--rerank local`; see [below](#pre--and-post-rerank-hitk) |
| **Tier 2** — `citation_precision`, `invalid_marker_rate`, `refusal_accuracy` | **live and judge-free** — `production_rag.evals.tier2_answer`, scored over the real `run_query` path |
| **Tier 2** — `faithfulness`, `relevance` | **live, judge-dependent** — default judge is `fake` (lexical overlap, offline). A hosted judge needs `--judge openai`, `RUN_LLM_EVALS=1` **and** a credential |
| Opt-in retrieval gate | **live** — `--fail-under-hit`, default `0.0`, i.e. reporting only |
| Chunk-level metrics (`recall@k` etc. over `relevant_chunk_ids`) | **not implemented** — the labels do not exist; every metric above is source-level and named `source_*` where it matters |
| Thresholds in `configs/default.yaml` | **read by nothing** — see [thresholds](#thresholds) |
| Merge gate wired into CI | **not wired** — the mechanism exists (`--fail-under-hit`), the number to set it to does not |

Three rows are the ones people misread, so they are stated plainly:

- **Every metric is source-level, not chunk-level.** ADR-0003 specifies tier 1
  over `relevant_chunk_ids`; what runs is the same arithmetic over
  `expected_source_paths`. `tier1_retrieval` names its outputs `source_hit_at_k`
  and `source_recall_at_k` for exactly that reason — reporting a document-level
  number as `recall@k` would overstate it. `citation_precision` inherits the same
  limit: it checks that a citation points at the right *document*.
- **The default run is a plumbing check, not a quality claim.** Fake embedder,
  fake generator, fake judge: free, deterministic, CI-runnable, and semantically
  meaningless in its dense branch and its two judged columns. The report says so
  in `offline_defaults` rather than leaving it to be reconstructed from the
  invocation.
- **`evals.thresholds` in the config gates nothing.** The runner reads no
  threshold from config; the only gate is the `--fail-under-hit` flag, and it
  defaults to off.

**No pre/post number is reported in this repository.** The procedure below is
what produces one; running it needs `--embedder openai` and a real reranker, and
the result is a 17-item smoke test either way.

`scripts/eval_hit.py` predates the harness and stays: stdlib only, one metric, no
thresholds, no gate. It is the zero-dependency way to check a collection end to
end. The M6 runner is what to use for anything that produces a report —
`production_rag.evals.run`, below.

### What M4 changed, and what it did not

**Changed.** The answer path now emits exactly the structured artifacts an answer
metric consumes, which is why the schema was designed before the harness:

| Artifact | The metric it feeds later |
|---|---|
| `answer` prose with resolvable `[n]` markers | `faithfulness` — every claim traceable to a cited passage |
| `citations[]` with `chunk_id` and `source_path` per marker | `citation_precision` — as implemented, whether the citation points at an expected *document*; whether the passage supports the sentence is the judge's question |
| `invalid_markers` | model-invented markers, countable **without a judge** |
| `uncited_claims` | citation coverage per request, also judge-free |
| `refused` | `refusal_accuracy` — judge-free, compared against `answerable` on the golden item |
| `hits_retrieved` / `hits_used` | separates a truncation failure from a retrieval failure |
| `refusal_reason` (closed set) | groups refusals by cause instead of counting them as one bucket |

Because the citations are structured data rather than prose to be parsed, none of
these needs the answer to be re-analysed. That was the design intent of
[ADR 0005](adr/0005-grounded-generation.md).

**Not changed.** M6 scores answers; it does not produce a defensible *quality*
number, and the difference is the whole of this section. Specifically:

- **The faithfulness column is not a faithfulness rate.** `tier2_answer` reports
  `faithfulness` and `relevance` from whichever `AnswerJudge` was passed, and the
  default is `FakeJudge` — lexical overlap between the answer and the cited text,
  offline, deterministic, and not a semantic judgement of anything. The report
  always carries the judge's name so the column can be read for what it is. A
  hosted judge is `--judge openai`, and it is still uncalibrated: no 20-item
  hand-labelled subset has been scored against it.
- **`citation_precision` is document-level.** It is the share of emitted
  citations whose `source_path` is in `expected_source_paths`. A citation can be
  document-correct and point at a passage that says something else; catching that
  is the judge's job, which is why the judge-free and judged metrics are reported
  side by side and never averaged together.
- **`--llm fake` measures the contract, not the answer.** The fake generator
  stitches top passages together and emits markers that resolve by construction,
  so its `citation_precision` and `invalid_marker_rate` are close to free wins.
  What it *does* prove is real: the refusal decision, marker resolution, and the
  wiring of the whole path. `offline_defaults: true` in the report is the flag
  that keeps those two readings apart.
- **A four-item `unanswerable` slice can show a pattern, not a rate.** M6 grew
  the slice from one item to four (`q-0012`, `q-0015`, `q-0016`, `q-0017`), which
  is what makes `refusal_accuracy` meaningful at all — with one item, a single
  flip was the whole slice. Four items is enough to see the refusal path fire or
  fail to fire; it is not enough to quote the resulting percentage as a property
  of the system, and no such figure is quoted anywhere in this repository.

### Ops signals are not eval metrics

M5 made the request path report on itself: per-node `timings_ms`, `invalid_markers`,
`hits_used` vs `hits_retrieved`, the `rerank` summary, `refusal_reason`
([ADR 0006](adr/0006-observability.md)). Every one of those is produced on every
request, for free, with no judge and no labels. That is exactly what makes them
tempting to quote as quality, and exactly why they are not.

The distinction is one line: **an ops signal says what the system did; an eval
metric says whether it was right.** Nothing in the first column below needs a
golden set, and nothing in it can replace one.

| Ops signal (live, every request) | What it detects | The eval metric it is *not* |
|---|---|---|
| `timings_ms` per node | which stage got slow | nothing — latency is not quality, and a fast wrong answer scores perfectly here |
| `invalid_markers` | the model emitting markers that resolve to nothing | `citation_precision`. Zero invalid markers says nothing about whether the markers that *did* resolve support their sentences |
| `hits_used` vs `hits_retrieved` | the context budget truncated the tail | `recall@k`. It separates a truncation failure from a retrieval failure; it does not measure either |
| `rerank.applied` / `rerank.error` | the reranker failed open and ordering quality dropped silently | `nDCG`. It says the stage ran, not that it ordered well |
| `refusal_reason` | which of the four refusal paths fired | `refusal_accuracy`, which needs an `unanswerable` slice to compare against |

Read the columns in the right order and they are complements rather than rivals.
An ops signal moving is the *prompt* to run an eval; an eval number moving is the
prompt to read the ops signals for a mechanism. What neither of them does is
substitute for the other:

- **A dashboard of ops signals cannot gate a merge.** All five are green for a
  system that answers every question fluently and wrongly, provided it cites
  something.
- **An eval run cannot diagnose.** `hit@5` dropping tells you nothing about
  whether the reranker has been failing open since Tuesday. That is in the logs.

Two of these are worth counting over the golden set today, because they
cost a `--llm fake` run and no judge: a non-empty `invalid_markers` across many
items is a model or prompt problem, and a systematic `hits_used` far below
`hits_retrieved` means the context budget — not retrieval — is deciding what gets
cited. Both are findings about the *system*, reportable without a quality claim
attached.

### Reading `refusal_accuracy`, honestly

Tier 2 computes it, judge-free: a case scores 1 when `refused` equals
"this item is unanswerable", 0 otherwise, averaged. `is_unanswerable` accepts
either signal — the `unanswerable` category *or* an empty
`expected_source_paths` — because the golden set is hand-written and an item that
carries only one of the two must not silently score as a system failure.

The number is a mean over a two-by-two, and at this set size the table is the
finding and the mean is a summary of it:

| | refused | answered |
|---|---|---|
| unanswerable (4 items) | correct | **hallucination under pressure** — the defect the slice exists to catch, listed in `refusal_failures` |
| answerable (13 items) | over-refusal; check `refusal_reason` | expected |

So `refusal_accuracy` is scored but not quotable: one flip in the top row is 25
points of that row. Read `refusal_failures` — it names the cases — before reading
the percentage.

Two degenerate outcomes stay diagnostic on sight, regardless of slice size:

| Observation | What it means |
|---|---|
| Nearly everything refuses | retrieval or `score_threshold`, not the prompt. Check `hits_retrieved`, and remember the threshold applies to a fused RRF score whose maximum is `≈ 0.0328` |
| Nothing ever refuses, including on the unanswerable items | the evidence bar is doing nothing. Confirm a corpus-impossible question comes back `refused: true` with `refusal_reason: no_evidence` |

Both are visible with `--llm fake`, offline, because the pre-call refusal is
system logic rather than model behaviour — the model is never called on that
path. That is one of the few things the fake generator genuinely proves, and it
is why `make eval-tier2-fake` is worth running on a change that touches
retrieval, thresholds or the evidence bar.

### Why no answer number is quoted as quality

An answer metric that can be quoted needs three things. M6 built the machinery
for all three and satisfies one and a half. Naming which is which is the point —
"tier 2 runs" and "faithfulness is measured" are not the same claim:

| Precondition | State after M6 |
|---|---|
| **A judge, calibrated.** `gpt-4o` scoring `faithfulness` is a proxy, and an uncalibrated proxy is a number rather than a measurement — hence the 20-item hand-labelled calibration subset in the target design below. | **machinery yes, calibration no.** `--judge openai` exists behind `RUN_LLM_EVALS=1` and a credential; nothing has been calibrated against hand labels, and the default judge is lexical. |
| **A set that can carry a threshold.** One item moves `source_hit@k` by roughly eight points on the 13 scorable cases. An answer metric on a set this size produces a decimal with no right to two significant figures. | **missing.** 17 items against a stated minimum of 50. |
| **The composition slices.** `refusal_accuracy` is meaningless without unanswerable items; `faithfulness` measured only on questions retrieval already got right overstates the system exactly where it fails. | **the slice landed.** 4 explicit `unanswerable` items, and `is_unanswerable` accepts either signal — the `unanswerable` category *or* an empty `expected_source_paths` — so a hand-written item that carries only one of them still scores correctly. |

The alternative — quoting a faithfulness number from a handful of hand-read
answers, or from a lexical-overlap judge — is precisely the "seems better" this
document exists to refuse. Tier 2 running makes the refusal path *scored*; it
does not make the answers *graded*.

### Reading a `hit@k` number honestly

`hit@k` asks one question: *did any chunk from a labelled document appear in the
top k?* It is coarse by construction — a document can be retrieved for the wrong
reason and still count.

What the number means depends entirely on which embedder built the collection,
and the two cases are not comparable:

| Collection built with | What `hit@k` measures | What it does **not** measure |
|---|---|---|
| `--embedder fake` | that the plumbing works end to end: the corpus is indexed, both branches return, fusion orders, payload paths match the labels. The sparse branch is genuinely lexical, so exact-token items can legitimately hit. | anything about dense retrieval quality. The dense branch contributes hash noise, and a conceptual/paraphrase item hitting is luck. |
| `--embedder openai` | retrieval quality on this corpus, at this chunking, with this fusion config — a real number for a 17-item set, which is to say a smoke test with error bars measured in whole documents. | statistical significance. One item moves `hit@5` by roughly eight points, the aggregate being over the 13 scorable items. |

So: a `fake`-path `hit@k` is a **plumbing assertion**, and the only honest way to
report it is alongside the embedder that produced it. Neither number belongs in a
README as a quality claim. See [ADR-0003](adr/0003-eval-strategy.md).

The per-branch breakdown is worth more than the headline at this size. `hit@k`
on the sparse branch alone, over the `exact_token` slice, is a direct test of the
thing M2 was built for — and it is meaningful even on a `fake` collection.

## Pre- and post-rerank `hit@k`

Reranking is the first change in this project whose value is a *delta*, not a
level. "Post-rerank `hit@3` is 0.71" says nothing on its own; "it was 0.57 before
the stage and 0.71 after, same corpus, same embedder, same run" is a finding.

Run the same golden set twice, changing exactly one thing:

```bash
# baseline: fusion order (M2 behaviour)
docker compose run --rm api python scripts/eval_hit.py --embedder openai --rerank off

# with the cross-encoder
docker compose run --rm api python scripts/eval_hit.py --embedder openai --rerank local
```

```powershell
.\scripts\eval_hit.ps1 -Embedder openai                    # baseline
.\scripts\eval_hit.ps1 -Embedder openai -Rerank local      # reranked
```

> `--rerank` takes the same values as the retrieve command — `off` (the default,
> so a score stays comparable with the M2 baseline), `fake`, `local`, `cohere`,
> `auto` — and passes straight through to the retriever. Toggling
> `rerank.enabled` / `rerank.provider` in a config profile and passing `--config`
> changes the same thing. Either way the script reports and never gates.

### Where the delta should show up, and where it should not

| Metric | Expected effect of reranking | Why |
|---|---|---|
| `hit@1`, `hit@3` | should improve if the stage is worth its latency | rerank moves the right passage up |
| `hit@10`, `hit@12` | should be **flat** | rerank cannot add a document fusion never returned; at `k` equal to the candidate window it is a permutation of the same set |
| `hit@k` for `k` > `rerank.top_k` | **not comparable** | the reranked run returns only `rerank.top_k` hits (6 by default), so every `k` above that scores against a shorter list. Compare at `k ≤ 6`, or raise `rerank.top_k` for the run |

That last row is the trap. A naive comparison at `hit@10` shows reranking making
things *worse*, because the reranked run was asked for six hits. It is an
artifact of the cut, not a regression.

`hit@k` also under-reports what reranking does. The stage's job is ordering, and
`hit@k` is a set membership question: moving the answer from rank 5 to rank 1
changes `hit@1` and nothing else. `nDCG` is the metric that sees the whole
reordering. Tier 1 reports `ndcg_at_k`, but with **binary** gain — no golden item
carries the graded `relevance` field — so it sees the reordering and not the
grades. Graded nDCG still waits for chunk-level labels.

### What makes the comparison meaningless

- **`--embedder fake`.** The dense branch is hash noise, so the candidate list
  the reranker is handed is partly arbitrary. A cross-encoder reordering noise
  produces a number about nothing.
- **`--rerank fake`.** It scores by query-term overlap, a cruder version of what
  BM25 already contributed to fusion. Expect a near-zero delta and read nothing
  into it either way. Both fakes together measure plumbing twice.
- **Changing two things.** Embedder, `input_top_k`, `top_k`, chunking — one per
  run, or the delta has no owner.
- **Seventeen items, thirteen of them scored.** One item moves `hit@3` by
  roughly eight points. A delta smaller than that is noise, and no threshold
  should be built on this set.

The honest report is therefore: *`hit@k` at `k ≤ 6`, on an `openai`-embedded
collection, with the reranker named, the candidate window stated
(`input_top_k`), and the item count attached.* Anything less is an adjective.

### The candidate ceiling is measurable too

With rerank on, a chunk outside `input_top_k` is unreachable. Before concluding
that the reranker ordered badly, check `rerank.candidates` in the output and
compare against the `--rerank off` run at the same `k`: if the baseline did not
retrieve it either, the defect is in retrieval and no reranker can fix it. That
separation — recall failure versus ordering failure — is the same principle the
[two-tier strategy](#principle-separate-the-two-failure-modes) applies one level
up.

## What the seed set is for

Seventeen items is far too few to gate a merge, and it is not trying to. It
exists to do three things now, so that none of them has to be invented later
under deadline:

1. **Pin the schema.** Downstream code can be written against a real file.
2. **Prove the corpus is reachable.** Every `expected_source_paths` entry
   resolves to a committed file under `data/raw/`; a path that does not exist
   scores as a permanent miss and reads exactly like a retrieval regression.
3. **Write the questions before the retriever exists.** Reading a chunk and then
   writing a question about it produces questions phrased in the chunk's own
   vocabulary, which flatters sparse retrieval and measures nothing. Authoring
   ahead of the retriever removes that temptation structurally.

### Document labels first, chunk labels later

The seed set names source paths, not chunk ids. Chunk ids exist after ingest,
but a chunk id is `<doc_id>:<index>` — so changing `chunk_size` or
`chunk_overlap` does not invalidate a label, it quietly repoints it at a
different passage. A label that breaks loudly is recoverable; one that keeps
scoring against the wrong text is not. M1 is exactly the milestone in which
those chunking values are most likely to move.

Document-level labels support `hit@k` — did any chunk from the right document
reach the top k — which is a coarser metric than `recall@k` and a real one. When
chunking settles, `relevant_chunk_ids` is added *alongside*
`expected_source_paths`, not instead of it: the coarse labels stay useful as the
check that survives the next re-chunk. M2 did not settle it — chunk size and
overlap are unchanged, but nothing has yet been tuned against a measurement, so
the chunk-level labels still wait.

### Labels match on `source_path`, exactly

The comparison `scripts/eval_hit.py` performs is a string match between
`expected_source_paths` and the `source_path` field of each returned hit. Both
sides must be relative to the same corpus root.

That makes the ingest `SOURCE` argument part of the eval contract, not an
operator preference. Ingesting `data/raw` stores `sample/08-bm25-vs-dense.md`,
which is what the labels say. Ingesting `data/raw/sample` stores
`08-bm25-vs-dense.md`, every label misses, and `hit@k` reads `0.00` — identical
in shape to a total retrieval failure and caused by nothing but a path prefix.

`scripts/eval_hit.py` reports unmatched-label paths separately from misses for
exactly this reason: a label whose file does not exist under `data/raw/` is a
dataset bug, not a retrieval result.

## Running the two tiers

One command runs both, because ADR-0003's stated cost of splitting the strategy
in two was "two commands and two sets of results to reconcile" — and that
reconciliation is what nobody does. `production_rag.evals.run` scores retrieval
and answers in the same process, over the same sample, against the same
collection, into one versioned JSON report.

```bash
make reingest-fake        # the eval scores what is in Qdrant; rebuild it first
make eval-tier1           # retrieval only: free, deterministic
make eval-tier2-fake      # answers with the fake generator and fake judge
make eval-all-fake        # both tiers, one report
```

The targets are thin wrappers. The runner takes the knobs:

```bash
python -m production_rag.evals.run --tier all --embedder fake --llm fake     --k 5 --rerank off --report data/eval/reports/all-fake.json
```

| Flag | Default | Why it matters |
|---|---|---|
| `--tier` | `all` | `1`, `2` or `all` |
| `--embedder` | `fake` | `openai` is the only setting under which a retrieval number is about retrieval |
| `--llm` | `fake` | the fake generator is extractive, so tier 2's citation and refusal columns are real while the prose is not |
| `--judge` | `fake` | lexical overlap, offline. `openai` also needs `RUN_LLM_EVALS=1` **and** a credential |
| `--sample` / `--seed` | all cases / `42` | a sampled run is reproducible, and the report carries both |
| `--fail-under-hit` | `0.0` | the only gate; `0.0` means report-only |
| `--no-answers` | off | omit generated answers from the report — they quote corpus text, and report files get pasted around |

Anything that costs money is opt-in twice: a flag, and for the judge an
environment variable plus a credential. A judge that runs because someone typed
the wrong flag is a bill and a surprise.

### Reading the report

The last line of stdout is the report, `report_version: 1`. Four fields decide
what the numbers are allowed to mean, and they are worth reading before the
scores:

| Field | Why it comes first |
|---|---|
| `offline_defaults` | `true` when the embedder and generator are both fake. The scores are then a plumbing check. Stated in the file, because the reader of a committed report was not at the terminal |
| `embedder`, `llm`, `rerank` | a `hit@k` without its embedder, or a citation number without its generator, is not a reproducible figure |
| `golden_cases` / `scored_cases` / `sample` / `seed` | 17 items; a sampled run says which and how it chose |
| `gate` | present only when `--fail-under-hit` was raised above zero; carries `metric`, `threshold`, `value`, `passed` |

**Tier 1** reports `source_hit_at_k`, `source_recall_at_k`, `mrr` and
`ndcg_at_k`, plus `by_category`. Two of those are separate for a reason: a
multi-source item such as `q-0008`, whose answer spans two documents, scores
`1.0` on hit and `0.5` on recall when retrieval finds one of them. Only recall
notices, and the difference is a real partial failure.

The aggregate denominator is **13, not 17**. The four `answerable: false` items
carry no expected path, and the scorer excludes them and reports them as
`unscored_cases`. Retrieval cannot hit a document that does not exist; scoring
them as misses would depress a number retrieval cannot improve, and scoring them
as hits would be a lie. Their correct outcome is a refusal, which tier 2 owns.

**Tier 2** reports `citation_precision`, `invalid_marker_rate` and
`refusal_accuracy` with no judge at all, and `faithfulness` and `relevance` from
whichever judge ran. `refusal_failures` is the actionable list — a missed refusal
is a hallucination with citations on it, and it is the one failure in this report
that is worth stopping for.

### Interpreting the gate

There is exactly one gate, it is a flag, and it is off by default:

```bash
make eval-tier1 EVAL_ARGS="--fail-under-hit 0.8"    # exit 1 if source hit@k < 0.80
```

Three properties of that design are deliberate:

- **It gates on `source_hit_at_k` only.** The one metric that is deterministic,
  free, and independent of any model or judge. Gating on a judged column would
  make a build outcome depend on a non-deterministic proxy.
- **It reads no threshold from `configs/default.yaml`.** The value is typed at
  the call site, so a gate is always something someone chose for this run rather
  than a number inherited from a file nobody re-read. See
  [thresholds](#thresholds).
- **`0.0` means report, and that is the default.** A gate that fires on a
  17-item set fires on noise, and a gate that fires on noise gets switched off
  within a week — which is worse than no gate, because the pipeline still claims
  one.

When the gate fails the run exits `1` and `ok` is `false` in the report; the
report is still written, because a failing gate with no numbers attached is an
alert, not a finding.

### What a run proves offline, and what it does not

| Runs offline, means something | Runs offline, means nothing |
|---|---|
| `refusal_accuracy` — the pre-call refusal is system logic, not model behaviour: the model is never called on that path | `faithfulness` / `relevance` under the fake judge — lexical overlap between an answer and the text it was stitched from |
| `invalid_marker_rate` — marker resolution is code | `citation_precision` under `--llm fake` — the generator emits markers that resolve by construction |
| `source_hit_at_k` on the `exact_token` slice — BM25 weights come from the text, so the sparse branch is genuinely lexical | `source_hit_at_k` on conceptual items with `--embedder fake` — the dense branch is hash noise |

## Design principles and target state

### Principle: separate the two failure modes

A wrong answer has exactly two causes, and they need different fixes:

1. **Retrieval failed** — the supporting chunk was never handed to the model.
   No prompt change fixes this.
2. **Generation failed** — the chunk was present and the model ignored,
   misread, or embellished it.

Measuring only end-to-end answer quality collapses these into one number and
sends you tuning prompts when the real defect is chunk size. So retrieval is
scored first and independently — that is tier 1 — and the answer columns are
read next to it rather than on their own.

**The "judge only where retrieval succeeded" filter is specified and not
implemented.** `evaluate_tier2` answers every case in the sample. That is
tolerable at 17 items, where the per-case rows are read individually and
`refusal_failures` names the cases that mattered, and it stops being tolerable as
soon as an aggregate is quoted: a `faithfulness` mean that includes items whose
supporting document was never retrieved is grading the retriever. Until the
filter lands, read tier 2's aggregates against tier 1's `misses` list from the
same run — which is one reason both tiers report into one file.

M4 made this principle load-bearing rather than tidy. Because the system refuses
when nothing clears the evidence bar
([ADR 0005](adr/0005-grounded-generation.md)), a retrieval miss now surfaces as a
**refusal**, which looks to a user like "the system does not know" and to a naive
end-to-end metric like an answer-quality failure. It is neither: it is a recall
failure wearing a polite message. `refusal_reason` distinguishes the causes —
`no_evidence` is a retrieval miss, the other three are the model — and the
library result carries `hits_retrieved` and `hits_used`, so the two can be told
apart without re-running anything.

### Golden dataset

`data/eval/golden.jsonl` — one JSON object per line. Field spec in
[`data/eval/README.md`](../data/eval/README.md). Minimum viable set is 50
queries; below that, a single item moves recall@5 by two points and the metric
stops being a signal. The committed seed set is 17 items and is explicitly not a
gate — see [Status after M6](#status-after-m6).

Composition targets:

| Slice | Share | Why it exists |
|---|---|---|
| Paraphrase / conceptual | 40% | the case dense retrieval is supposed to win |
| Exact token (IDs, names, codes) | 25% | the case sparse retrieval is supposed to win |
| Multi-hop (needs 2+ chunks) | 20% | exposes context-budget truncation |
| Unanswerable | 15% | the refusal path; without these, a model that never refuses scores well |

The unanswerable slice is the one most often skipped and the most diagnostic:
it is the only measurement of whether the system hallucinates under pressure.

### Retrieval metrics

No LLM involved, so runs are free, fast, and deterministic. Two columns, because
what is *specified* and what is *computed* differ by one label granularity and
the names in the report say so:

| Specified in ADR-0003 | Computed today | Question it answers |
|---|---|---|
| `recall@k` over `relevant_chunk_ids` | `source_recall_at_k` over `expected_source_paths` | what share of a case's expected sources appeared in the top k? |
| `hit@k` | `source_hit_at_k` | did *any* expected source appear at all? |
| `mrr` | `mrr` | how far down the list was the first expected source? |
| `ndcg@k` | `ndcg_at_k` (binary gain) | is the ordering good? Graded gain needs the `relevance` field, which no item carries |

`source_hit_at_k` is the headline while labels stay document-level, and it is the
metric `--fail-under-hit` gates on. Hit and recall are both reported because they
disagree exactly where it matters: `q-0008` expects two documents, so finding one
of them scores `1.0` hit and `0.5` recall, and only recall calls that a partial
failure.

Report retrieval metrics per branch as well as fused — `dense`, `sparse`,
`fused`. A fused score that never beats its best branch means the fusion
constants need work, not the retriever. That is what
`production_rag.evals.ablation` is for.

Chunk-level labels remain the upgrade. The metric functions do not change when
they arrive; the names lose their `source_` prefix and the numbers stop being
coarse.

### Answer metrics

Scored over the real `run_query` path — the same entry point the API calls,
because an eval that reimplements the pipeline measures the reimplementation.
Three of the five need no judge:

| Metric | Definition | Needs a judge |
|---|---|---|
| `citation_precision` | share of emitted citations whose `source_path` is one of the case's expected sources | no |
| `invalid_marker_rate` | share of emitted markers that resolved to nothing | no |
| `refusal_accuracy` | refused exactly on the items whose correct outcome is a refusal | no |
| `faithfulness` | every claim in the answer is supported by a cited passage | **yes** |
| `relevance` | the answer addresses the question actually asked | **yes** |

The judge-free three are deterministic and reproducible, so they are the ones
worth watching per commit. The judged two carry the judge's name in the report,
and under the default `fake` judge they are lexical overlap rather than a
semantic judgement — present in CI, and meaning very little there.

Faithfulness is still the metric that matters most: an unfaithful answer with
confident citations is worse than no answer, because it survives review. It is
also the one this repository cannot yet quote, because no judge here has been
calibrated against hand labels.

### `invalid_marker_rate` is not `citation_precision`

These two are the pair most likely to be confused, because both are free, both
are about markers, and both come back from the same tier-2 run. They differ on
**which markers they look at**:

| | `invalid_marker_rate` | `citation_precision` |
|---|---|---|
| Looks at | markers the model emitted that resolve to **nothing** | citations that **did** resolve to a chunk |
| Question | did the model invent a reference? | is the cited chunk from a document the answer was supposed to come from? |
| Denominator | every marker emitted, valid or not | citations emitted, on cases that expected a source |
| Undefined when | never — zero emitted markers scores `0.0` | a case cited nothing, or expected nothing; those cases are left out rather than scored `0` |

They are disjoint by construction: a marker cannot be both unresolvable and
wrongly-sourced. So neither substitutes for the other —

<!-- provenance-allow: explanatory counterexample; zero invalid markers does not prove citation support -->
- `invalid_marker_rate: 0.0` on every item is compatible with every citation
  pointing at a real chunk from the wrong document. Every marker resolved; none
  of them was right.
- A burst of invalid markers is a *model or prompt* defect — the markers are
  stripped from the answer and never rendered — and says nothing about the
  quality of the citations that survived.

And there is a third level neither of them reaches. `citation_precision` as
implemented compares `source_path` against `expected_source_paths`: it says the
citation points at the right **document**, not that the cited **passage**
supports the sentence citing it. That last question needs a judge, which is why
the judged columns sit beside these two in the report instead of being folded
into them.

The practical rule: watch `invalid_marker_rate` continuously because it is free,
and never write it in a sentence containing the words "citation quality".

## Thresholds

There are two unrelated things called a threshold in this repository, and
conflating them is how a project ends up believing it has a gate:

| | `evals.thresholds` in `configs/default.yaml` | `--fail-under-hit` |
|---|---|---|
| Read by | **nothing** | `production_rag.evals.run` |
| Names | `recall_at_5`, `mrr`, `faithfulness` | `source_hit_at_k` |
| Effect | none | exit `1`, `ok: false`, `gate` block in the report |
| Default | 0.80 / 0.65 / 0.90 | `0.0` — reporting only |

**The config block gates nothing.** `recall_at_5` names a chunk-level metric no
code computes — tier 1 computes `source_recall_at_k` over document paths, which
is a different measurement under a similar name — and `faithfulness` is a judged
column whose default judge is lexical. The runner never reads the block.

**Neither set of numbers came from a baseline run.** They were written with the
ADR, before anything ran, which makes them aspiration. ADR-0003 requires
thresholds to be set from a first full baseline; that baseline has not been
performed, so the committed values satisfy the letter of nothing. They get
*replaced* by measured numbers, not adjusted toward them. The config comments say
this in the file rather than leaving the values to look authoritative.

Arming a real gate needs three things, in this order:

1. **A recorded baseline.** One full run on an `openai`-embedded collection, with
   the corpus, config, embedder and item count stated, whose numbers become the
   thresholds. A threshold above the system's own baseline blocks every merge; one
   far below it blocks nothing and reads as diligence.
2. **A set that can carry it.** 17 items, 13 of them scored. One item is roughly
   eight points. A gate at this size fires on noise.
3. **A decision about which metric.** `--fail-under-hit` gates the deterministic
   one on purpose. Gating a judged column makes the build depend on a
   non-deterministic proxy, and ADR-0003 requires that proxy to be calibrated
   first.

Once armed: raise thresholds as the system improves; lowering one to make a build
pass is a decision that belongs in an ADR, not in a commit.

## Where results land

Results land in `data/processed/eval-runs/<timestamp>/` (gitignored) as
`summary.json` plus a per-query `details.jsonl`. `seed: 42` in config keeps two
runs over the same dataset comparable.

## Interpreting a regression

| Symptom | Likely cause | Where to look |
|---|---|---|
| `hit@k` is exactly `0.00` at every k | labels and payload paths disagree | the ingest `SOURCE` root — see [labels match on `source_path`](#labels-match-on-source_path-exactly) |
| `recall@5` drops, `mrr` stable | fewer relevant chunks retrieved overall | chunking size/overlap, `dense_top_k` |
| `recall@5` stable, `mrr` drops | ordering degraded | fusion weights, `rrf.k` |
| Sparse branch collapses | tokenisation or stopwords changed | `ingest.sparse` block |
| Sparse branch quietly returns nothing | collection predates M2, no `sparse` vector | rebuild: `make reingest-fake` |
| Exact-token items regress after adding documents | IDF drift since the last full ingest | full re-ingest, then re-measure |
| `hit@k` drops at high `k` right after enabling rerank | the reranked run returns only `rerank.top_k` hits | compare at `k ≤ rerank.top_k`, or raise it — see [pre/post](#pre--and-post-rerank-hitk) |
| Reranking changes nothing at any `k` | the reranker never ran, or ran on `fake` | check `rerank.applied` and `rerank.error` in the output; a fail-open degradation reports itself |
| Post-rerank ordering is worse than the baseline | the right chunk was outside `input_top_k`, or the provider is `fake` | check `rerank.candidates` against the `--rerank off` run at the same `k` |
| `faithfulness` drops, retrieval flat | prompt or model change | `generation.prompt`, model version |
| Refusals spike | `score_threshold` raised too far, or retrieval regressed | `retrieval.score_threshold` first, then `refusal_reason` and `hits_retrieved` — `no_evidence` is a recall failure until proven otherwise |
| `invalid_markers` keeps coming back non-empty | the model is inventing markers | `generation.model` or `configs/prompts/system.md`; the markers are stripped from the answer, never rendered |
| Answers get vaguer, citations point at plausible-but-tangential chunks | the context budget is truncating | `hits_used` vs `hits_retrieved`; `generation.prompt.max_chunks_in_prompt`, `generation.max_context_tokens` — retrieval order is truncation order |
| Refusals that are all `model_abstained` and never `no_evidence` | the pre-call evidence check is not firing, so every refusal costs a call | `generation.citations.refuse_without_evidence`; see [ADR 0005](adr/0005-grounded-generation.md) |

Change one variable per run. Two changes and a moved metric produce a story,
not a finding.

## What is deliberately not measured yet

- **Latency under concurrency.** Single-request latency is recorded per stage on
  every request (library result and logs always; the HTTP response only under
  `debug`), but per-request timings say nothing about behaviour under load. Load
  testing waits until the deployment target is real.
- **Cost per query.** The provider reports prompt and completion tokens on the
  generation call, but nothing aggregates them yet and the query result does not
  carry them; a per-query cost figure is a later milestone.
- **Human preference.** The LLM judge is a proxy. It is calibrated against a
  20-item hand-labelled subset once, and re-calibrated when the judge model
  changes — an uncalibrated judge is a number, not a measurement.
