---
title: Citations and grounding — making an answer checkable
tags: [rag, citations, grounding, faithfulness]
---

# Citations and grounding — making an answer checkable

An answer without a source is a claim. An answer with a source is a claim plus
the means to check it. In a retrieval system the sources are already in hand —
the chunks that were retrieved — so attaching them costs almost nothing and
changes what the system is for.

## Grounding is a property of the answer, not the prompt

"Grounded" means every factual claim in the answer traces to a passage that was
actually in the context window. Asking the model politely to only use the
provided context raises the rate but does not make it a property of the output.
The claim has to be checked after generation, not merely requested before it.

The two checks that matter are different:

- **Faithfulness** — does the cited passage support the sentence citing it?
- **Citation precision** — of the citations emitted, how many are load-bearing
  rather than decorative?

A model that appends `[1][2][3]` to every sentence scores well on the first and
badly on the second, which is why both are reported.

## Citation markers and their mapping

The simplest scheme that survives contact with a real UI is a bracketed ordinal:
the model writes `[2]`, and `2` indexes the retrieved chunk list as it was
presented in the prompt. Generation never sees a `chunk_id`, so it cannot
hallucinate one; the mapping from ordinal back to `chunk_id` and `source_path`
happens in application code after the answer is produced.

The failure mode this avoids is subtle. If the model is asked to emit chunk ids
directly, it will occasionally emit a plausible-looking id that was never in the
context, and that citation points at nothing — or worse, at a real chunk that
says something else.

Ordinals outside the presented range are dropped and logged rather than
rendered. A citation to `[9]` when 6 chunks were supplied is a defect, and
silently rendering it as a dead link hides the defect from everyone.

## Chunk identity has to be stable

A citation is only useful if it resolves to the same text tomorrow. `chunk_id`
is therefore derived from the document path, the chunk index, and a hash of the
chunk text — not from a random UUID minted at ingest. Re-ingesting an unchanged
document produces the same ids, and a stored citation still resolves.

The corollary is that changing chunk size or overlap changes every id in the
corpus. That is honest rather than annoying: the chunk that was cited genuinely
no longer exists.

## Quoting versus summarising

Citations are more checkable when the answer stays close to the source
wording for the load-bearing claim. A short verbatim span plus a marker lets a
reader confirm the claim without opening the document; a heavy paraphrase of
three chunks into one fluent sentence is where unsupported detail creeps in.

This is a prompt-level choice, and it trades readability against verifiability.
Systems whose answers get audited should lean toward quoting.

## What citations do not fix

Citations make an answer auditable. They do not make it correct, and they do not
make retrieval better. A confidently wrong answer with a real citation attached
to a passage that does not support it is the single most expensive failure mode
in the system, because the citation is what convinces the reviewer to stop
reading. Faithfulness scoring exists precisely to catch it.
