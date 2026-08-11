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
3. Ask **“What is the capital city of the Moon?”** Show the explicit refusal and
   its reason.
4. Stop recording after the refusal. The refusal reads as a deliberate product
   choice only after the grounded path has established that the system will
   answer confidently when it has evidence.

This is a plumbing demonstration, not a quality evaluation: the embedder, LLM,
and reranker are deterministic fakes. Stop the stack afterward with
`docker compose down`; the named Qdrant volume is retained.
