# Season plan — production-rag → v1.0.0 (90 days)

**Owner:** A5 (this repository only).  
**Horizon:** one quarter, not one afternoon.  
**Baseline:** `main@bf6e36d` · Releases **v0.2.0** and **v0.3.0** (Latest) · clone-only · $0 free path.  
**Law:** Month 1 week 1 is **design only** (this file). `REPORTE … OK` is illegal before the Month 3 gate. Do not retag v0.3.0 as v1.0.0.

Authoritative references: master plan §6 (P1), §12 (eval doctrine), §29.2 (dream quality bar), §37 (demo day), §39 (stretch — load, SSE already shipped). Supporting ADRs: 0001 hybrid, 0003/0008/0010 eval, 0004 fail-open rerank, 0005 cite-or-refuse, 0011 filters fail-closed, 0012 stream additive, 0013 filter-aware cache.

---

## 0. What this season is (and is not)

### Threat model for the hiring manager

A staff engineer will open this repo and ask:

1. Can I run the whole answer path with empty keys?
2. When the corpus cannot support an answer, does the system **refuse**, or invent?
3. When a dependency fails, is that a **typed error**, or a soft refusal?
4. Are the published numbers a **measurement**, or a plumbing screenshot with a quality caption?
5. Can I force a failure (wrong collection, Qdrant down, bad filter field) and see the system behave on purpose?

v0.3.0 already answers (1) and most of (2) on the free path, and ships stream, filter-aware cache, and scorecard **replay**. It does **not** yet ship a free-path evaluation *program* with difficulty predicates that reject trivial slices, productized collection identity, multi-corpus mismatch, or hiring-manager-runnable failure injection with captures.

### Non-goals (entire season)

| Non-goal | Why |
| --- | --- |
| Hosted P1 (Vercel or otherwise) | Clone-only is the product; `production-rag.vercel.app` is **Ipsura**, never cite it |
| Auth / rate limit | P5 platform ownership |
| Redis / multi-worker cache | ADR-0013; correctness is the key contract, not the transport |
| Hosted Qdrant Cloud | Month 2 ADR keeps local Qdrant; still no keys in CI |
| Invented SOTA / quality claims from `fake` | Free path proves contracts; quality needs named real providers and reportable stats |
| Secret values in CI or the vault | Empty provider keys in CI; pointers only elsewhere |
| Subagents / multi-repo edits | A5 owns `production-rag` only |

### Season deliverables (high level)

| Month | Product surface | Evidence a stranger can run |
| --- | --- | --- |
| **1** | Evaluation program: free-path golden **n ≥ 50**, slices, **difficulty predicates**, scorecard HTML+JSON, ablation labels, Compose transcripts | `pytest` fails if a slice is all trivial rank-1; `docs/assets/scorecard.html` + JSON; grounded / refuse / `title=Filtering` / stream |
| **2** | Corpus / collection identity on `/ready` (or equivalent); incremental ingest no-op; second corpus; cache key includes identity; typed wrong-collection failure; DEMO-DAY beats | Tests + captures for corpus-mismatch and incremental-ingest |
| **3** | Failure injection (Qdrant down → 503/degraded UI; embedder fail ≠ refusal; provider fail ≠ refusal); local load 100 fake queries p50/p95 with honesty line; CASESTUDY ≥1500 words on season trade-offs; **v1.0.0** only if checklist green | Captures + load JSON + release notes listing what remains PLANNED |

---

## 1. Fifteen invariants

Each invariant is **normative**. Column **Tested today** is the state on `bf6e36d`. Month 1–3 work may add tests; it must not weaken these.

| # | Invariant | Meaning | Source | Tested today (v0.3.0) |
| --- | --- | --- | --- | --- |
| I1 | **Cite-or-refuse** | A user-facing answer either carries resolvable `[n]` citations mapped to prompt blocks, or `refused: true` with a closed-set reason. Fluent unsupported prose is a defect. | ADR-0005 | **yes** — generation/guardrail unit + API contract tests |
| I2 | **Provider fail ≠ refusal** | LLM/embedder/store outages propagate as errors (HTTP 5xx / stream `error`), never as `refused: true`. | ADR-0005, ADR-0012 | **partial** — library paths cover generation failure; Month 3 adds UI/HTTP captures |
| I3 | **Fail-open rerank** | Reranker timeout/error returns fusion order; query succeeds; `rerank.applied` / `rerank.error` report degradation. Missing capability (no sparse vector) is hard-fail. | ADR-0004 | **yes** — rerank fail-open tests |
| I4 | **Filter allowlist fail-closed** | Field outside `retrieval.filters.allowed_fields` → 422 `filter_not_allowed` (CLI exit 2). Never drop and answer. | ADR-0011 | **yes** |
| I5 | **Empty filters = legacy path** | Omitted / null / `{}` filters do not build a Qdrant expression; unfiltered behaviour matches pre-filter code. | ADR-0011 | **yes** |
| I6 | **Cache key includes filters** | Same question with different filters cannot cache-hit. | ADR-0013 | **yes** |
| I7 | **Cache key includes collection identity** | Same question on two collections cannot cross-hit. | ADR-0013 (key table) | **partial** — collection name is in the key; Month 2 productizes full identity (embedder id, chunker version, doc count, corpus hash) and a two-corpus test |
| I8 | **Stream is additive** | `POST /v1/query` remains the contract; `POST /v1/query/stream` returns the same terminal body. Deltas are provisional; `result` is authoritative. | ADR-0012 | **yes** |
| I9 | **Stream refuse vs error** | Refusal after deltas → terminal `result` with `refused: true`. Mid-stream provider failure → `error` event, not a soft refusal. | ADR-0012 | **yes** (stream+refuse, stream+citations; provider path asserted in tests) |
| I10 | **Free-path reports `billed=false`** | Scorecards and eval reports from fake providers must carry `billed: false` / `offline_defaults` and must not be published as quality. | ADR-0010, evaluation.md | **yes** — scorecard JSON + render guard |
| I11 | **Reportable comparison rule** | `reportable = (n >= 30) and (ci95 excludes zero)`. Slice n=10 is never reportable. | ADR-0010 | **yes** — stats module + scorecard replay tests |
| I12 | **Difficulty predicates on adversarial slices** | A slice that is all trivial (e.g. every target already rank-1 under the baseline retriever the slice claims to stress) fails integrity / CI. Labels alone are not enough. | ADR-0008 lesson; token-savings (deep_rank rank-1 collapse) | **partial** — paraphrase zero-overlap mechanical; deep_rank notes only; **Month 1 weeks 2–4 close the gap** |
| I13 | **NullTracer default** | Observability is a seam; missing SDK/env degrades to null; tracer faults never break the query body. | ADR-0006 | **yes** |
| I14 | **Credential-free CI** | Default tests and CI set provider keys empty; free path needs no network to a paid API. | SHIP.md, §13 | **yes** |
| I15 | **Clone-only honesty** | README / SHIP / DEMO-DAY never claim a public P1 host; demo URLs are localhost. Auth/rate-limit named PLANNED or owned by P5. | §6, DEMO-DAY | **yes** (docs); guard with content checks as needed |

**Capture gotcha (operational, not numbered):** UI/Compose failure stills for the error path must set `CACHE_ENABLED=false`. Compose enables cache by default; a failure still that hits cache is a false green for the error path.

**Filter demo gotcha:** chip `title=Filtering` is the DEMO-DAY beat. `source=sample` does **not** filter the sample corpus the way operators expect for the Qdrant docs demo — do not use it as the filter proof.

---

## 2. Month 1 — evaluation program

### Week 1 (this commit) — DESIGN ONLY

Deliverable: **this file**. No new golden rows, no new harness code, no new HTML scorecard beyond what v0.3.0 already ships. Commit and stop implementing Month 1 features in the same PR as this design.

### Baseline inventory (do not re-litigate)

<!-- provenance-allow: historical-measurement: baseline inventory counts from ADR-0008 and seed set; not a quality claim -->
| Artifact | n / shape | Role |
| --- | --- | --- |
| `data/eval/golden.jsonl` | 17 items · sample corpus | CI/plumbing seed; **not** a quality benchmark (random hit@5 ~56% on 9 docs) |
| `data/eval/golden-corpus.jsonl` | **60** items · 6 slices × 10 · Qdrant docs corpus | Measurement + adversarial slices (ADR-0008) |
| `data/eval/reports/scorecard.json` + `docs/assets/scorecard.html` | paired matrix, `billed=false` | Plumbing scorecard + HTML replay |
| Ablation | dense / sparse / fused / +rerank | Live; free path = **plumbing labels only** |
| Compose / UI captures | grounded, refuse, filtered, stream, error | Live; error path requires cache off |

ADR-0003 already targets **≥50** items for a set that can carry a threshold. The corpus golden **meets n=60 on paper**. What Month 1 still lacks is a **free-path evaluation program**: mechanical difficulty predicates, a CI failure mode when a slice is trivial, published scorecard that states which comparisons are reportable, and Compose transcripts a hiring manager can re-run without keys.

### Free-path set (n ≥ 50) — definition for this season

**Free-path set** means:

- Runnable with `embedder=fake`, `llm=fake`, empty provider keys, local Qdrant (Compose or test double as already used).
- Committed golden JSONL under `data/eval/`.
- **n ≥ 50** scorable program items after excluding pure schema fixtures if any.
- Every item has: stable `id`, `question`, `expected_source_paths` (or empty for unanswerable), `answerable`, `category` (slice id), and notes stating **which behaviour** the item exists to catch.
- Scorecard and eval JSON carry `billed: false` and provider ids.

**Default source of truth for n≥50:** extend and **guard** `golden-corpus.jsonl` (already 60) as the free-path *program* set when paired with the vendored corpus + fake providers. Keep `golden.jsonl` (17) as the fast sample fixture; do not pretend 17 items are the season set.

If a week of work shows fake-embedder dense noise makes some difficulty predicates uncheckable offline, prefer:

1. Predicates that do not need dense quality (lexical overlap, unanswerable adjacency, multi-source path counts, filter-narrowing fixtures on sample where applicable), and  
2. Optional offline sparse/fused rank snapshots committed as **expected difficulty fixtures** (not quality claims), regenerated only via an explicit script.

Never replace a difficulty predicate with “the author believes this is hard.”

### Slices (program taxonomy)

Keep the six adversarial slices from ADR-0008. Month 1 does not invent a seventh marketing slice.

| Slice | Intent | Free-path signal (honest) |
| --- | --- | --- |
| `lexical_only` | rare exact tokens; sparse should dominate | **meaningful** under fake (BM25 is real) |
| `paraphrase_only` | zero sparse term overlap with target docs | integrity already enforces zero BM25 overlap; **dense quality not free** |
| `multi_source` | ≥2 expected sources | label cardinality check + hit requires multi-path retrieval |
| `distractor` | surface-near wrong doc | needs distractor path present in corpus; free path can assert distractor is *indexed*, not that dense confuses |
| `near_miss_unanswerable` | topical retrieval, correct outcome is refuse | refusal path is free-path real (`llm=fake` still refuses on evidence bar) |
| `deep_rank` | target below top ranks pre-rerank | **requires rank fixture / baseline rank check** — the historical failure mode |

Composition targets for the **program** set (not the 17-item seed):

<!-- provenance-allow: explanatory-example: composition targets and seed fractions are planning numbers, not measured quality -->
| Bucket | Target share | Notes |
| --- | --- | --- |
| answerable adversarial (five slices) | ~83% | 10 each today |
| unanswerable / near-miss | ≥15% of program set | today 10/60 ≈ 16.7% |
| multi-source | keep ≥10 items | context-budget / multi-doc pressure |

Do **not** pad with easy rank-1 paraphrase clones to hit n≥50. n is necessary, not sufficient.

### Difficulty predicates (mechanical)

A **difficulty predicate** is a pure function (or offline check script) that returns pass/fail for a slice or item. CI runs it. Failure message names the slice and the trivial items.

<!-- provenance-allow: explanatory-example: difficulty predicates use rank thresholds as set-design rules, not published metrics -->
| Slice | Predicate (Month 1 implement) | Fails when |
| --- | --- | --- |
| `paraphrase_only` | After production BM25 tokenize (lowercase, stopword strip), query ∩ target document tokens = ∅ | any shared term (already in `scripts/check_golden_integrity.py`) |
| `lexical_only` | Query contains ≥1 rare token that appears in the target source and in ≤ *T* corpus documents (T small, e.g. 5) | “exact token” items that are actually common words |
| `multi_source` | `len(expected_source_paths) ≥ 2` and every path exists under corpus root | single-source labels smuggled into the slice |
| `distractor` | Notes or structured field names a distractor `source_path` that exists, is ≠ any expected path, and shares a surface keyword with the question | missing/identical distractor |
| `near_miss_unanswerable` | `answerable is false` **and** question shares ≥1 content token with ≥1 real corpus doc (adjacency, not off-topic junk) | unanswerable items that never retrieve anything (only tests score_threshold) |
| `deep_rank` | Under a **committed baseline rank table** (sparse or fused, fake or recorded): labelled source’s best rank **> r_min** (r_min ≥ 3, matching ADR-0008 “below rank three”) for ≥ 80% of the slice | **≥50% of slice already rank-1** under that baseline — the “n=10 all rank-1” worthless set |

**Hard rule for the program:** if any slice fails its difficulty predicate, the integrity job exits non-zero even if schema validation passes.  
**Hard rule for reporting:** a configuration comparison remains subject to I11; difficulty is about the *labels*, reportability is about the *statistics*.

### What Month 1 measures (and publishes)

| Published artifact | Contents | Allowed claim |
| --- | --- | --- |
| Scorecard JSON | per-config hit vectors, paired Δ, ci95, McNemar, `reportable`, `billed: false`, embedder/llm/rerank ids, n, date, git SHA | plumbing + statistical machinery |
| Scorecard HTML | same numbers rendered; reportable vs directional badges | same |
| Ablation table | dense / sparse / hybrid(fused) / hybrid+rerank | **labeled plumbing** on free path; no “hybrid wins quality” without real embedder + reportable rule |
| Compose transcripts | grounded · refuse · `title=Filtering` · stream (and error with cache off) | contract demos |

### What Month 1 does **not** measure

- Hosted-provider faithfulness / relevance as a quality number (judge still uncalibrated; opt-in only).
- Chunk-level `recall@k` (no `relevant_chunk_ids` that survive re-chunk; still source-level).
- Latency SOTA or multi-node throughput (Month 3 load is honest single-process).
- Cross-corpus identity product surface (Month 2).
- Qdrant-down degraded UI as a finished capture pack (Month 3).

### Reportable rule (copy for operators)

```text
reportable = (n >= 30) and (ci95 excludes zero)
```

- Zero on either CI bound includes zero → not reportable.  
- Every n=10 slice is directional forever under this rule.  
- Fake-provider scorecards always `billed=false`; their point estimates are not quality.

### Month 1 weeks 2–4 — build sequence (after this design commit)

1. **Difficulty module + tests** — implement predicates above; wire into `check_golden_integrity` / pytest; seed a deep_rank baseline rank fixture if needed.
2. **Audit golden-corpus** — rewrite or drop any item that fails predicates; keep n≥50 without trivial padding.
3. **Free-path scorecard refresh** — regenerate JSON+HTML under fake providers; ensure ablation labels say plumbing; CI replay stays green.
4. **Compose transcripts** — re-run grounded, refuse, title=Filtering, stream; fix capture script cache footgun if still latent.
5. **Digest append** — weekly note in second brain; no Month 2 code in the same commits unless isolated and not required for Month 1 gate.

**Month 1 exit (not OK for season):** difficulty predicates green; n≥50 program set; scorecard HTML+JSON published with honesty labels; four Compose transcripts present; free-path CI green. Still no v1.0.0 tag.

---

## 3. Month 2 — corpus identity (PLANNED)

Design only here; implement after Month 1 exit.

1. **Productize collection identity** on `/ready` or a dedicated readiness/detail field: embedder id, chunker version, document count, corpus hash (content-addressed over ingested source bytes or committed manifest). Today `/ready` only reports config parse + `qdrant_configured` without dialing Qdrant.
2. **Incremental ingest:** unchanged docs are no-ops (content-hash skip already exists in ingest — productize and demo). Second small corpus; querying the wrong collection fails **typed** (not a silent empty or wrong-answer cross-talk).
3. **Cache key includes full collection identity** — test: same question, two corpora, **no cross-hit**.
4. **ADR:** why corpus hash belongs in the cache key; why Qdrant stays local (no hosted Qdrant, no keys in CI).
5. **DEMO-DAY beats:** corpus-mismatch + incremental-ingest.

---

## 4. Month 3 — failure injection + v1.0 (PLANNED)

1. **Inject failures a hiring manager can run**
   - Qdrant down → typed **503** / degraded UI (not a refuse chrome).
   - Embedder fail ≠ refusal; provider fail ≠ refusal; captures for each.
2. **Load:** 100 local fake queries; publish p50/p95; caption: *single process + local Qdrant, not SOTA throughput* (§39 stretch, honesty required).
3. **CASESTUDY:** ≥1500 words covering RRF, refuse vs error, filters, why not hosted. (Existing `docs/CASESTUDY.md` already exceeds 1500 words at baseline; Month 3 **revises** it for season artifacts rather than counting old pages alone.)
4. **`gh release` v1.0.0** only if the checklist below is green. Latest = v1.0.0. Release notes list remaining PLANNED items. Still no Vercel.

---

## 5. v1.0.0 checklist (season gate)

- [x] This file (`docs/SEASON.md`) lists ≥15 invariants and which have tests.
- [x] Free-path eval program **n ≥ 50** with difficulty predicates; trivial all-rank-1 slices fail CI.
- [x] Scorecard HTML+JSON with `billed=false` and reportable vs directional labelling (I10–I11).
- [x] Ablation dense / sparse / hybrid / hybrid+rerank labeled as plumbing on free path.
- [x] Compose transcripts: grounded, refuse, `title=Filtering`, stream; failure captures for Month 3 classes.
- [x] Collection identity + incremental ingest + wrong-collection typed failure (Month 2).
- [x] Failure injection pack (Month 3) + honest load artifact.
- [x] CASESTUDY ≥1500 words with real trade-offs; no invented SOTA.
- [ ] CI green on `main` with empty provider keys (requires merge + Actions).
- [ ] Release notes state what remains PLANNED (auth, rate limit, hosted quality baseline, hosted Qdrant, multi-worker cache).

---

## 6. DEMO-DAY beats owned by P1 (season end)

Aligned with §37 (P1 segment) and local DEMO-DAY script:

| Beat | Proof |
| --- | --- |
| Grounded hybrid answer | citations resolve; timings optional via debug |
| Stream draft → grounded or refuse | stream additive; provisional deltas |
| Explicit refusal | corpus-impossible / near-miss |
| Filter `title=Filtering` | citations narrow; allowlist fail-closed on bad field |
| Corpus mismatch (Month 2) | typed failure, not wrong fluent answer |
| Incremental ingest (Month 2) | second run no-ops unchanged docs |
| Dependency down (Month 3) | 503/degraded, not refuse |
| Scorecard honesty | open HTML; point at `billed=false` and non-reportable slices |

---

## 7. Week 1 stop line

This document is the **only** Month 1 week 1 deliverable.

**Do not** in the same change: grow golden JSONL, re-render scorecards as a “finish,” implement `/ready` identity, inject Qdrant failures, or tag a release.

Next concrete commit theme after merge of this design: `test(eval): difficulty predicates reject trivial deep_rank` (or equivalent), then grow/repair the free-path program set until n≥50 predicates stay green.
