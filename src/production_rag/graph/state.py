"""The typed state threaded through the query graph.

ADR-0002 constraint 2: **the state object is a project-owned Pydantic model, not
a framework dict.** A dict state is how a graph ends up with a key that one node
writes and another silently never reads, and no type checker to notice.

The model is deliberately wide — it carries the intermediate artefacts
(``retrieval``, ``draft``, ``cited``) as well as the final answer. That is the
observability argument from the ADR: when a request is slow or wrong, the first
question is *which stage*, and the answer has to be readable off the state
rather than reconstructed from logs.

Fields default to a "nothing ran yet" value so a state built from a query alone
is valid. Nodes return partial updates; LangGraph merges them, so no node
mutates what it was given.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from production_rag.generation.citations import Citation, CitationResult
from production_rag.generation.generator import Draft, GenerationResult
from production_rag.retrieval.hybrid import RetrievalHit, RetrievalResult

RETRIEVE_NODE = "retrieve"
RERANK_NODE = "rerank"
GUARD_NODE = "guard"
GENERATE_NODE = "generate"
CITE_NODE = "cite"
FINALISE_NODE = "finalise"
NODE_NAMES = (
    RETRIEVE_NODE,
    RERANK_NODE,
    GUARD_NODE,
    GENERATE_NODE,
    CITE_NODE,
    FINALISE_NODE,
)
"""Node names. Constants because they are also the keys in ``timings_ms``, and a
timing dashboard that breaks on a rename is a dashboard nobody trusts."""


class QueryState(BaseModel):
    """Everything the query path reads or produces, in one typed object.

    ``arbitrary_types_allowed`` is on because the retrieval and generation
    artefacts are frozen stdlib dataclasses owned by their own modules. Wrapping
    them in Pydantic models to satisfy the graph would put the framework's
    requirements into the library, which is exactly the inversion ADR-0002
    forbids.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- inputs -----------------------------------------------------------
    query: str
    mode: str | None = None
    top_k: int | None = None

    # --- retrieval --------------------------------------------------------
    hits: tuple[RetrievalHit, ...] = ()
    # Annotated `Any`, and only this field. Pydantic introspects nested stdlib
    # dataclasses, and RetrievalResult carries a RerankOutcome whose `hits`
    # annotation names RetrievalHit — a name rerank.py imports only under
    # TYPE_CHECKING, because hybrid.py imports rerank.py and a runtime import
    # would close a cycle. So the annotation is unresolvable at schema-build
    # time through no fault of either module. The alternatives were worse:
    # rewriting retrieval's import graph to satisfy a validator it does not use,
    # or flattening the result into fields that would drift from it. Read it
    # through `retrieval_result`, which restores the type.
    retrieval: Any = None

    # --- generation -------------------------------------------------------
    draft: Draft | None = None
    cited: CitationResult | None = None
    generation: GenerationResult | None = None

    # --- outcome ----------------------------------------------------------
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    refusal_reason: str | None = None
    model: str | None = None

    # --- observability ----------------------------------------------------
    # Per-node wall time. The `latency_ms` breakdown the API returns and the
    # runbook tells an operator to read first comes straight from here.
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def retrieval_result(self) -> RetrievalResult | None:
        """The retrieval result, typed. See the note on :attr:`retrieval`."""
        return cast(RetrievalResult | None, self.retrieval)

    def timed(self, node: str, elapsed_ms: float) -> dict[str, float]:
        """Return ``timings_ms`` with *node* recorded, without mutating self.

        Each node merges rather than appends because the alternative — a
        LangGraph reducer annotation on the field — would put framework syntax
        into the state model, and ADR-0002 keeps that boundary sharp.
        """
        return {**self.timings_ms, node: round(elapsed_ms, 3)}

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe view for logging a completed run."""
        retrieval = self.retrieval_result
        return {
            "query": self.query,
            "mode": retrieval.mode if retrieval else self.mode,
            "hits": len(self.hits),
            "citations": len(self.citations),
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "model": self.model,
            "timings_ms": dict(self.timings_ms),
        }
