# Record the demo

The recording target is `docs/assets/production-rag-demo.gif`. Do not add an
image at that path until a real recording exists.

## Prepare

From the repository root, run exactly one setup command:

```powershell
.\scripts\demo_setup.ps1
```

On macOS or Linux, run `./scripts/demo_setup.sh`. Both scripts start Qdrant,
rebuild only the `prag_demo` collection from `data/corpus` with the deterministic
fake embedder, and start the web server. They make no billed provider call.

## Record in this order

1. Open <http://localhost:8000/> and keep the browser wide enough to show the
   answer and its evidence together.
2. Ask **“Why does hybrid search use reciprocal rank fusion?”** Show the grounded
   answer, citation links, source passages, and pipeline timings.
3. Ask **“Who won the Antarctic underwater chess championship?”** Show the explicit refusal and
   its reason.
4. Stop recording after the refusal. The refusal reads as a deliberate product
   choice only after the grounded path has established that the system will
   answer confidently when it has evidence.

This is a plumbing demonstration, not a quality evaluation: the embedder, LLM,
and reranker are deterministic fakes. Stop the stack afterward with
`docker compose down`; the named Qdrant volume is retained.

## What the page shows

The header carries a **Demo mode (free / deterministic)** label, and every result
repeats it in its diagnostics footer. The label is literal: the UI route pins the
request to the fake embedder and the fake LLM, so no API key is read and no billed
provider is called. A grounded result separates the **Answer** from the
**Citations** that support it, each marker linking to the passage it came from; a
refusal states its reason instead of guessing; a dependency failure says so without
leaking the underlying error. Per-node durations sit behind the collapsed
**Pipeline timings** control.

## Refresh the still captures

`docs/assets/ui-grounded.png`, `ui-refusal.png` and `ui-service-failure.png` come
from `python scripts/capture_ui.py`, after `pip install -e ".[docs]"` and
`playwright install chromium`. That script drives the same credential-free stack
these setup scripts start — fake providers, local corpus, no key — so refreshing the
images costs only the time it takes to run. The committed images predate the current
header label, the Answer/Citations split, and the service-link footer; re-run the
capture to bring them back in step.
