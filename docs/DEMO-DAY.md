# Demo day — forty-five minutes across five systems

A minute-boxed script for showing the whole series live: P1 production-rag, P2
agentic-rag-research, P3 multi-agent-orchestration, P4 RepoMind, P5 ai-platform.

Everything here runs on the **free path**: deterministic local providers, no OpenAI key,
no hosted provider, no billed call, no signup. If a step in this script needs a
credential, the step is wrong — not the reviewer's setup.

The script is written to be read aloud. Each segment states the exact command, the exact
question, **what the reviewer must see**, and **what not to claim** — because the fastest
way to lose a technical interviewer is to oversell a deterministic fixture as a quality
result.

> This document is a hosting-free plan. Nothing in the series is deployed on the public
> internet; every URL below is localhost, started by the presenter.

## Minute map

| Min | System | The one thing it proves |
| --- | --- | --- |
| 0–8 | P1 production-rag | Grounded answer with resolvable citations, and refusal as a designed outcome |
| 8–18 | P2 agentic-rag-research | A bounded loop that stops for a stated reason, with a trace and a step budget |
| 18–28 | P3 multi-agent-orchestration | Writer-only final output, and specialist failure surfacing as a typed degraded result |
| 28–38 | P4 RepoMind | `path:line` citations a reviewer can open, and honesty about the dogfood snapshot |
| 38–45 | P5 ai-platform | An edge that rejects, throttles, and reports its upstreams as unconfigured |

The master plan's §37 sketch allocated the first five minutes to profile narrative and
gave P1 ten. This script folds the narrative into the P1 segment: a reviewer who has just
watched a system refuse a question is a better audience for the series thesis than one
watching a slide. The systems and their order are unchanged.

## Before the call — preparation, not part of the forty-five minutes

Five dependency installs and one Docker image build do not fit inside the demo. Do this
beforehand, on the machine you will present from, and leave the terminals open.

```bash
git clone https://github.com/pabloalvarez99/production-rag
git clone https://github.com/pabloalvarez99/agentic-rag-research
git clone https://github.com/pabloalvarez99/multi-agent-orchestration
git clone https://github.com/pabloalvarez99/repomind
git clone https://github.com/pabloalvarez99/ai-platform
```

Each repository below uses its own virtual environment. Commands use the Windows venv
layout (`.venv/Scripts`); on macOS or Linux, use `.venv/bin`.

| System | Port | Prepared beforehand |
| --- | --- | --- |
| P1 production-rag | 8000 | `docker compose build api`, then `.\scripts\demo_setup.ps1` |
| P2 agentic-rag-research | 8010 | `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` |
| P4 repomind | 8020 | `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` |
| P3 multi-agent-orchestration | 8030 | `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` |
| P5 ai-platform | 8080 | `py -3.12 -m venv .venv && .\.venv\Scripts\python -m pip install -e ".[dev]"` |

P3's README serves on port 8000, which P1 already holds. This script moves P3 to 8030 with
`--port 8030`, matching the port P5 documents for `MAO_URL`. Confirm all five ports are
free before starting; a port collision mid-demo costs more time than the segment it breaks.

**Rehearse the failure segments.** Two moments in this script deliberately break something
(P1's stopped Qdrant is optional; P5's throttle is not). Know how long each takes to
recover on your machine.

---

## 0–8 · P1 production-rag — grounded, then refusing

**Open with the thesis, in one breath:** a retrieval system is only as good as the
evidence it can show for its own behaviour. Five systems, each one runnable for free,
each one able to return a result its author would not have chosen.

### Start it

```powershell
cd production-rag
.\scripts\demo_setup.ps1          # macOS or Linux: ./scripts/demo_setup.sh
```

One command starts Qdrant, rebuilds the `prag_demo` collection from `data/corpus` with the
deterministic embedder, and serves the UI. Open <http://localhost:8000/>.

### Ask, in this order

1. **"Why does hybrid search use reciprocal rank fusion?"**
2. **"Who won the Antarctic underwater chess championship?"**
3. **"How does filtering work in Qdrant?"** — twice. Once unfiltered, then again with the
   **metadata filter** set to `title` = `Filtering`.

The order carries the argument. A refusal shown first reads as a system that cannot
answer; a refusal shown second reads as a system that chose not to. The third pair is one
question asked twice, so the reviewer sees the narrowing change which documents the
citations come from rather than taking the filter on trust.

### What the reviewer must see

- A **Demo mode (free / deterministic)** label in the header and repeated in every result
  footer — the UI route pins the fake embedder and fake LLM, so no key is read.
- On the first question: an answer, citation markers, and the **source passages behind
  them**. Click a marker. It resolves to the exact chunk that was in the prompt context.
- Expand **Pipeline timings** and name the nodes aloud: retrieve, rerank, generate,
  validate citations, guardrails.
- On the second question: an explicit **refusal that states its reason**, with no
  invented supporting text.
- On the filtered run: every citation from `qdrant/search/filtering.md`, and a
  **Filtered: title = Filtering** chip in the result footer — an answer that was narrowed
  never reads as an answer over the whole corpus. The field list in the control is the
  deployment's allowlist, read from configuration; there is nothing outside it to pick,
  and a field posted by hand is answered 422 with `filter_not_allowed`, never dropped.

### Say this

"Marker validation happens against the exact prompt context, not against the corpus. If
the model cites something that was not retrieved, that is a failed validation, not a
lucky answer. When nothing survives, the guardrail refuses."

### Do not claim

- Do **not** present the deterministic run as retrieval or answer quality. The embedder,
  the LLM, and the reranker are fakes; this segment demonstrates plumbing and contracts.
- Do **not** cite the published scorecard as a quality result. It is an explicitly
  labelled local-provider fixture, and no hosted baseline has been run.
- Do **not** promise auth or rate limiting here. That is P5's repository, on purpose.
- Do **not** present the filter as access control. The allowlist bounds which *fields* a
  query may filter on, never which documents a caller may see. `title` is also the one
  allowlisted field with no payload index, so that filter is a scan — logged as
  `filter_field_unindexed` rather than hidden as unexplained latency.

### If a reviewer asks for the failure mode

`docs/assets/ui-service-failure.png` shows the dependency-failure rendering, captured from
a genuinely stopped Qdrant. Show the committed capture rather than stopping the container
live — recovering it costs more minutes than the segment has.

---

## 8–18 · P2 agentic-rag-research — a loop that stops for a reason

### Start it

```bash
cd agentic-rag-research
.venv/Scripts/uvicorn agentic_rag.main:app --port 8010
```

Open <http://127.0.0.1:8010/>. The UI pins `retriever=fake`; there is no control that
reaches a hosted model.

### Ask

1. **"What does hybrid retrieval buy over dense retrieval alone?"** — the loop completes.
2. **"What were the quarterly revenues in Patagonia?"** — the loop refuses.

Then show the same run as JSON, because the trace is the point:

```bash
.venv/Scripts/python -m agentic_rag.research \
  --question "Why use RRF in hybrid search?" --retriever fake
```

JSON goes to stdout, a one-line summary to stderr, so `| jq` works without a flag.

### What the reviewer must see

- **`stop_reason` as a field, not a mood.** `evidence_sufficient` on the completed run,
  `no_evidence` on the refusal.
- **The step budget.** Steps taken against the configured maximum. The loop is bounded by
  construction; it cannot decide to keep going.
- **The trace**, expandable in the UI and present in the JSON: `plan_created`,
  `tool_call`, `tool_result`, `critique`, then `synthesize` and `stop`.
- On the refusal: **empty citations and populated gaps** naming what was looked for and
  not found.

### Say this

"The interesting output of an agent is not the answer. It is the record of why it stopped.
A budget that only exists in a prompt is a suggestion; this one is enforced in the loop
and reported in the trace."

### Do not claim

- Do **not** claim the agent beats a single retrieval pass. The scorecard measures
  deterministic contract conformance, not uplift over a one-pass baseline.
- Do **not** imply the P1 HTTP retriever is running. It is opt-in and off; the default
  path contacts nothing.

---

## 18–28 · P3 multi-agent-orchestration — Writer-only, and degraded on purpose

### Start it

```bash
cd multi-agent-orchestration
.venv/Scripts/uvicorn mao.main:app --port 8030
```

Open <http://127.0.0.1:8030/> for the task console.

### Send one task

```bash
curl -s -X POST http://127.0.0.1:8030/v1/tasks \
  -H "content-type: application/json" \
  -d '{"task":"Audit retrieval risk","budget":{"max_handoffs":8}}'
```

The same path as a library call, if the console is easier to narrate:

```bash
.venv/Scripts/python -m mao.task --task "Compare hybrid vs dense retrieval"
```

### What the reviewer must see

- **The ordered trace**: Research, Critic, and Writer, with handoffs numbered and a
  bounded Critic-to-Research retry visible in the timeline.
- **Writer-only final output.** No other specialist speaks in the result. Say the rule
  aloud — one role owns the final text, so there is never a question of which voice the
  reviewer is reading.
- **Handoff and retry accounting** against the budget, plus the request ID.
- **The degraded path.** Ask for the remote Research specialist without configuring it:

```bash
curl -s -X POST http://127.0.0.1:8030/v1/tasks \
  -H "content-type: application/json" \
  -d '{"task":"Compare hybrid and dense retrieval","research":"http"}'
```

  Missing configuration returns a typed `409 capability_missing`. Timeout, transport, HTTP,
  and response-contract failures terminate as `degraded` — a named outcome, not a crash and
  not a silent fallback to a guess.

### Say this

"A multi-agent system without budgets is a system that can bill you forever. Every handoff
is counted, every retry is capped, and a specialist that fails makes the result degraded
and says so. The orchestrator has no path that hides a failure."

### Do not claim

- Do **not** claim answer quality or multi-model uplift. The specialists are deterministic
  fakes; what is proven is routing, ownership, budgets, and the trace contract.
- Do **not** claim P2 integration is live in this run unless `AGENTIC_RAG_URL` is set and
  P2 is actually running on 8010. If it is, say that you configured it a minute ago.

---

## 28–38 · P4 RepoMind — `path:line`, and the snapshot caveat

### Start it

```bash
cd repomind
.venv/Scripts/python -m uvicorn repomind.main:app --port 8020
```

Open <http://127.0.0.1:8020/> for the ask console. It uses no CDN.

### Ask, on the fixture first

**"Where is create_app defined?"** against the `mini` repository.

```bash
curl -s http://127.0.0.1:8020/v1/code/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Where is create_app defined?","repo_id":"mini"}'
```

Then the dogfood snapshot — the segment's real moment:

```bash
.venv/Scripts/python -m repomind ask \
  "Where is reciprocal_rank_fusion defined?" --repo production_rag
```

Then an unknown symbol, to show the refusal.

### What the reviewer must see

- Citations as **`path:start-end`**, not prose references. Open one. It is a real range in
  a real file.
- On the dogfood question: a citation into `production_rag/retrieval/rrf.py` — **the same
  RRF implementation demonstrated in the first segment**. Name that out loud. The series
  reading its own flagship is the strongest single moment in the forty-five minutes.
- On the unknown symbol: the fixed refusal with an **empty citation list**.

### Say this — the honesty beat

"That `production_rag` repository is a **frozen snapshot committed under `fixtures/`**, not
a live checkout of the sibling project. I vendored it so the demo is deterministic and the
eval runs offline. It proves navigation and citation correctness on committed bytes. It is
not a general code-RAG benchmark, and it does not prove the tool works on your repository."

Say it before a reviewer asks. Volunteering the boundary is the difference between a
portfolio and a pitch.

### Do not claim

- Do **not** describe the dogfood gate as evidence of general code comprehension.
- Do **not** imply the snapshot tracks the sibling repository. It does not update itself.
- Do **not** quote eval counts as a score. They are gates over committed fixtures, and the
  repository's own evaluation contract says exactly that.

---

## 38–45 · P5 ai-platform — the edge, and nothing behind it

Seven minutes. Three things: rejected, throttled, unconfigured.

### Start it

```powershell
cd ai-platform
.\.venv\Scripts\python -m uvicorn gateway.main:app --port 8080
```

### 1. Rejected

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/platform/status
```

`/health` is open. The `/v1/*` call without `X-API-Key` returns a typed **401**. Show the
error object: it names the failure type and carries a request ID, and it leaks nothing
about what is behind the edge.

### 2. Throttled

```powershell
$h = @{"X-API-Key"="dev-local"}
1..30 | ForEach-Object {
  try { Invoke-WebRequest http://127.0.0.1:8080/v1/platform/status -Headers $h | Out-Null }
  catch { $_.Exception.Response.StatusCode.value__ }
}
```

The per-key fixed window trips and returns a typed **429** carrying `Retry-After`. Rehearse
this: know your configured window, and know how long the reviewer waits before the next
call succeeds.

`dev-local` is a **public local fixture, not a credential**. Say that plainly. Then say
where a real deployment would keep keys: a secret store, injected as `GATEWAY_API_KEYS`,
never a file in the repository.

### 3. Unconfigured

Open <http://127.0.0.1:8080/> for the status console. Four upstreams, all reported
`unconfigured`. This is the expected free-path state, and it is the honest close of the
whole demo:

### Say this — the closing line

"This is a gateway, and there is nothing behind it right now. The four upstream URLs are
empty, so the console says `unconfigured` rather than pretending. Proxying is implemented
and tested against the prefix contract — unset upstreams fail immediately with a typed 503,
configured but unreachable ones return a typed 502 — but I do not operate a deployment
where a request through this edge reaches a running system. What I am showing you is that
the edge behaves. What I am not showing you is a platform in production."

If a reviewer wants the wired path, it exists as an opt-in local exercise: set the four
URLs, start P1 through P4, and run `.\scripts\e2e_local.ps1`. It is deselected in CI by
default. Offer it as a follow-up, not as a live segment — four services and a gateway is
not a seven-minute demonstration.

### Do not claim

- Do **not** say P5 hosts, runs, or serves P1 through P4.
- Do **not** call `dev-local` a key, a secret, or a credential.
- Do **not** claim the series is deployed. Nothing here is on the public internet.

---

## The three sentences to close on

1. Every system runs for free, deterministically, without a credential — which is why CI
   can run the whole path and a reviewer can reproduce it.
2. Every system can return a result its author would not have chosen: a refusal, a
   degraded outcome, a `no_evidence` stop, an `unconfigured` upstream.
3. Every number published anywhere in the series carries what produced it, and where a
   number would flatter without support, the repository says so instead.

## Recovery notes

| If this breaks | Do this |
| --- | --- |
| P1's Docker stack is slow to start | Show the committed captures in the README and keep talking; the stack catches up. |
| A port is already bound | Every service takes `--port`; P1 is the only one pinned by its compose file. |
| P3's remote research segment errors unexpectedly | That is the segment. Read the typed error aloud and move on — it is evidence, not a failure. |
| P5's rate limit will not trip | Say the window aloud, show the code path, and move to the status console. Do not improvise a louder loop. |
| Anything asks for an API key | Stop. Something is misconfigured; no segment in this script needs one. |
