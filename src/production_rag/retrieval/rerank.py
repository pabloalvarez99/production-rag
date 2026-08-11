"""Reranking: a second, more expensive look at the candidates fusion produced.

    query + top-40 fused hits -> reranker -> top-6

Retrieval and reranking answer different questions. A bi-encoder embeds the query
and the chunk separately, so the two never see each other; that is what makes an
index possible, and it is also what caps its precision. A cross-encoder reads the
pair together and can say "this passage mentions the flag but does not explain
it". The cost is that it cannot be indexed: every candidate is a forward pass, so
it only ever runs on a shortlist.

Hence the shape of this stage, and of ``rerank.input_top_k`` > ``rerank.top_k``:
fusion is recall-oriented and hands over more than the caller wants, the reranker
is precision-oriented and cuts it down. A reranker can only recover what it was
shown, so under-feeding it is the one tuning mistake that cannot be undone later.

Three implementations, honestly labelled:

* :class:`FakeReranker` — query-term overlap, pure Python, deterministic, offline.
  It is a **plumbing double, not a quality claim**: it re-ranks by lexical
  coverage, which is roughly what BM25 already did. It exists so the wiring, the
  fail-open path and the CLI can be tested with no model download and no network.
* :class:`LocalCrossEncoderReranker` — ``BAAI/bge-reranker-base`` through
  sentence-transformers. The real thing; the import is lazy so the dependency
  stays optional and the base install stays small.
* :class:`CohereReranker` — hosted, used only when a key is present. Never
  required: a portfolio that cannot be run without someone else's credential is a
  portfolio nobody runs.

**Failure is a policy decision, not an accident.** ``rerank.fail_open`` decides
whether a reranker that errors or times out degrades to the fusion order (the
default: availability beats a few points of nDCG) or fails the request. Either
way it is logged and reported in the result, never silent — a rerank that quietly
stopped happening looks exactly like a rerank that is not helping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from production_rag.config_loader import RerankConfig
from production_rag.retrieval.sparse import Bm25Tokenizer

if TYPE_CHECKING:  # pragma: no cover - import cycle only exists for the type checker
    from production_rag.retrieval.hybrid import RetrievalHit

_log = structlog.get_logger(__name__)

RERANK_OFF = "off"
RERANK_FAKE = "fake"
RERANK_LOCAL = "local"
RERANK_COHERE = "cohere"
RERANK_AUTO = "auto"
RERANK_KINDS = (RERANK_OFF, RERANK_FAKE, RERANK_LOCAL, RERANK_COHERE, RERANK_AUTO)
"""What ``--rerank`` accepts. ``auto`` reads the provider from the config file;
``off`` is the default, so M2 behaviour is what you get unless you ask for more."""

LOCAL_PROVIDER = "local-cross-encoder"
COHERE_PROVIDER = "cohere"
NONE_PROVIDER = "none"


class RerankError(RuntimeError):
    """The reranker cannot be built or cannot score.

    Whether this ends the request depends on ``rerank.fail_open`` — see
    :func:`apply_rerank`.
    """


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved hits by a second, pair-aware scoring pass."""

    @property
    def name(self) -> str:
        """Identifier recorded in the result, e.g. ``"fake"`` or the model name."""

    def rerank(self, query: str, hits: Sequence[RetrievalHit], *, top_n: int) -> list[RetrievalHit]:
        """Return at most *top_n* hits, best first.

        Implementations must set ``rerank_score`` and ``pre_rerank_rank`` on what
        they return, and renumber ``rank``, so a hit can always explain both where
        fusion put it and where reranking moved it.
        """


class FakeReranker:
    """Deterministic, offline reranker scoring lexical coverage of the query.

    **Not a quality claim.** The score is the fraction of distinct query terms
    present in the hit's text, which is a cruder version of what BM25 already
    computed; on a real corpus it will not beat the fusion order by much, and it
    will never do what a cross-encoder does. Its job is to make the rerank stage
    exercisable — wiring, ordering, truncation, fail-open, CLI flags — without a
    model download, a GPU or a network call, so the unit suite stays offline.
    """

    def __init__(self, *, tokenizer: Bm25Tokenizer | None = None) -> None:
        """Use the corpus tokenizer, so query and text are split the same way."""
        self._tokenizer = tokenizer or Bm25Tokenizer()

    @property
    def name(self) -> str:
        """Identifier recorded in the result."""
        return RERANK_FAKE

    def rerank(self, query: str, hits: Sequence[RetrievalHit], *, top_n: int) -> list[RetrievalHit]:
        """Score each hit by the share of query terms its text contains."""
        _require_positive(top_n)
        terms = set(self._tokenizer.tokenize(query))
        scored: list[tuple[RetrievalHit, float]] = []
        for hit in hits:
            if not terms:
                # A stopword-only query carries no signal to rerank by. Scoring
                # everything 0 preserves the fusion order rather than shuffling it.
                scored.append((hit, 0.0))
                continue
            present = terms & set(self._tokenizer.tokenize(hit.text))
            scored.append((hit, len(present) / len(terms)))
        return _ordered(scored, top_n=top_n)


class LocalCrossEncoderReranker:
    """Cross-encoder reranker running locally through sentence-transformers.

    The import is lazy and the dependency optional: installing torch to run a
    ``--rerank off`` retrieval would be a poor trade, so the cost is paid only by
    the caller who asks for this reranker. A missing dependency raises
    :class:`RerankError` naming the extra to install, rather than an ImportError
    from three frames down.

    The model is loaded once, on first use. Loading in the constructor would put
    a multi-second download behind object construction, where a CLI would pay it
    even on the argument-parsing error path.
    """

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        client: Any | None = None,
    ) -> None:
        """Configure the reranker. No model is loaded until :meth:`rerank` runs.

        Args:
            model: Hugging Face model id for the cross-encoder.
            batch_size: Pairs scored per forward pass.
            client: Pre-built ``CrossEncoder``. The seam integration tests and
                offline tests use instead of downloading weights.
        """
        self._model_name = model
        self._batch_size = batch_size
        self._client = client

    @property
    def name(self) -> str:
        """The model id, so an eval number records which reranker produced it."""
        return self._model_name

    def _load(self) -> Any:
        """Return the cross-encoder, importing and loading it on first use.

        Raises:
            RerankError: sentence-transformers is not installed, or the weights
                could not be loaded.
        """
        if self._client is not None:
            return self._client
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankError(
                "the local cross-encoder needs sentence-transformers; install it with "
                "`pip install -e .[rerank]`, or run with --rerank fake / --rerank off"
            ) from exc
        try:
            self._client = CrossEncoder(self._model_name)
        except Exception as exc:  # noqa: BLE001 - loader raises a wide family
            raise RerankError(
                f"could not load cross-encoder {self._model_name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("cross_encoder_loaded", model=self._model_name)
        return self._client

    def rerank(self, query: str, hits: Sequence[RetrievalHit], *, top_n: int) -> list[RetrievalHit]:
        """Score every (query, chunk) pair with the cross-encoder.

        Raises:
            RerankError: The model is unavailable or scoring failed.
        """
        _require_positive(top_n)
        if not hits:
            return []
        model = self._load()
        pairs = [(query, hit.text) for hit in hits]
        try:
            raw = model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise RerankError(
                f"cross-encoder {self._model_name!r} failed to score "
                f"{len(pairs)} pairs: {type(exc).__name__}: {exc}"
            ) from exc
        scores = [float(value) for value in raw]
        if len(scores) != len(hits):
            # A scorer that returns a different number of scores would silently
            # attach one passage's score to another passage.
            raise RerankError(f"cross-encoder returned {len(scores)} scores for {len(hits)} hits")
        return _ordered(list(zip(hits, scores, strict=True)), top_n=top_n)


class CohereReranker:
    """Hosted reranker. Optional: used only when a key is present.

    Same lazy-import discipline as the local one, for the same reason — nobody
    should need a Cohere account to run ``--rerank fake``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-english-v3.0",
        timeout_seconds: float = 15.0,
        client: Any | None = None,
    ) -> None:
        """Configure the client. No connection is opened until :meth:`rerank`.

        Raises:
            RerankError: No API key. Refusing here, before any work, means the
                failure is a usage error rather than a mid-request outage.
        """
        if not api_key and client is None:
            raise RerankError(
                "the cohere reranker needs an API key; set COHERE_API_KEY, or run "
                "with --rerank fake / --rerank local / --rerank off"
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._client = client

    @property
    def name(self) -> str:
        """The model id, recorded in the result."""
        return self._model

    def _load(self) -> Any:
        """Return the Cohere client, importing it on first use."""
        if self._client is not None:
            return self._client
        try:
            import cohere
        except ImportError as exc:
            raise RerankError(
                "the cohere reranker needs the cohere package; install it with "
                "`pip install -e .[rerank-cohere]`, or run with --rerank fake"
            ) from exc
        self._client = cohere.Client(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def rerank(self, query: str, hits: Sequence[RetrievalHit], *, top_n: int) -> list[RetrievalHit]:
        """Send the shortlist to Cohere and reorder by the returned relevance.

        Raises:
            RerankError: The call failed or returned an index out of range.
        """
        _require_positive(top_n)
        if not hits:
            return []
        client = self._load()
        try:
            response = client.rerank(
                model=self._model,
                query=query,
                documents=[hit.text for hit in hits],
                top_n=min(top_n, len(hits)),
            )
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise RerankError(
                f"cohere rerank with {self._model!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        scored: list[tuple[RetrievalHit, float]] = []
        for item in response.results:
            index = int(item.index)
            if not 0 <= index < len(hits):
                raise RerankError(f"cohere returned index {index} for a shortlist of {len(hits)}")
            scored.append((hits[index], float(item.relevance_score)))
        return _ordered(scored, top_n=top_n)


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """What the rerank stage did, whether or not it succeeded.

    Returned next to the hits instead of only logged: "was this ranking reranked?"
    must be answerable from the response a caller already has, otherwise an eval
    number cannot be attributed to a pipeline.
    """

    hits: tuple[RetrievalHit, ...]
    applied: bool
    reranker: str | None = None
    candidates: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Summary form for a CLI response."""
        return {
            "applied": self.applied,
            "reranker": self.reranker,
            "candidates": self.candidates,
            "error": self.error,
        }


def apply_rerank(
    reranker: Reranker | None,
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    top_n: int,
    fail_open: bool = True,
) -> RerankOutcome:
    """Rerank *hits*, applying the configured failure policy.

    Args:
        reranker: The reranker, or ``None`` to pass the hits through untouched.
        query: The user's query, verbatim.
        hits: Fusion output, already ordered.
        top_n: How many hits to keep.
        fail_open: On failure, degrade to the fusion order truncated to *top_n*
            instead of raising. The default, because a slightly worse ranking is
            worth more to a caller than an error.

    Returns:
        The outcome, carrying the hits and whether reranking actually happened.

    Raises:
        RerankError: Reranking failed and *fail_open* is ``False``.
    """
    _require_positive(top_n)
    if reranker is None:
        return RerankOutcome(hits=tuple(hits[:top_n]), applied=False, candidates=len(hits))

    try:
        reranked = reranker.rerank(query, hits, top_n=top_n)
    except RerankError as exc:
        if not fail_open:
            raise
        # Logged at warning, never silent: a rerank that quietly stopped happening
        # looks exactly like a rerank that is not helping.
        _log.warning(
            "rerank_failed_open",
            reranker=reranker.name,
            candidates=len(hits),
            error=str(exc),
        )
        return RerankOutcome(
            hits=tuple(hits[:top_n]),
            applied=False,
            reranker=reranker.name,
            candidates=len(hits),
            error=str(exc),
        )

    _log.info(
        "rerank_applied",
        reranker=reranker.name,
        candidates=len(hits),
        returned=len(reranked),
    )
    return RerankOutcome(
        hits=tuple(reranked),
        applied=True,
        reranker=reranker.name,
        candidates=len(hits),
    )


def build_reranker(
    kind: str = RERANK_OFF,
    *,
    config: RerankConfig | None = None,
    api_key: str | None = None,
) -> Reranker | None:
    """Construct the reranker named by *kind*.

    Args:
        kind: One of :data:`RERANK_KINDS`. ``auto`` reads ``rerank.provider`` from
            the config file and returns ``None`` when ``rerank.enabled`` is false,
            which is what makes the YAML the single switch for a deployment.
        config: The ``rerank`` block; defaults apply when omitted.
        api_key: Credential for a hosted provider. Read from the environment by
            the caller — never a CLI flag, since a key on a command line ends up
            in shell history.

    Returns:
        A reranker, or ``None`` when reranking is off.

    Raises:
        RerankError: Unknown kind, unknown provider, or a hosted provider with no
            key.
    """
    settings = config or RerankConfig()
    resolved = kind.strip().lower()
    if resolved not in RERANK_KINDS:
        raise RerankError(f"unknown reranker {kind!r}; expected one of {', '.join(RERANK_KINDS)}")

    if resolved == RERANK_AUTO:
        if not settings.enabled:
            return None
        provider = settings.provider.strip().lower()
        if provider == NONE_PROVIDER:
            return None
        if provider == LOCAL_PROVIDER:
            resolved = RERANK_LOCAL
        elif provider == COHERE_PROVIDER:
            resolved = RERANK_COHERE
        elif provider == RERANK_FAKE:
            resolved = RERANK_FAKE
        else:
            raise RerankError(
                f"unknown rerank.provider {settings.provider!r}; expected "
                f"{COHERE_PROVIDER!r}, {LOCAL_PROVIDER!r}, {RERANK_FAKE!r} or {NONE_PROVIDER!r}"
            )

    if resolved == RERANK_OFF:
        return None
    if resolved == RERANK_FAKE:
        return FakeReranker()
    if resolved == RERANK_LOCAL:
        return LocalCrossEncoderReranker(model=settings.local_model)
    return CohereReranker(
        api_key=api_key or "",
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
    )


def _ordered(scored: Sequence[tuple[RetrievalHit, float]], *, top_n: int) -> list[RetrievalHit]:
    """Sort by rerank score, truncate, and stamp the provenance onto each hit.

    Ties break on the pre-rerank rank, so a reranker that cannot separate two
    passages leaves them in the order retrieval chose rather than shuffling them.
    That also makes the output deterministic, which an eval number depends on.
    """
    ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0].rank))
    return [
        replace(
            hit,
            rank=position,
            rerank_score=score,
            # The fusion position is kept, not overwritten: "the cross-encoder
            # pulled this from rank 27 to rank 2" is the sentence that justifies
            # paying for this stage.
            pre_rerank_rank=hit.pre_rerank_rank or hit.rank,
        )
        for position, (hit, score) in enumerate(ranked[:top_n], start=1)
    ]


def _require_positive(top_n: int) -> None:
    """Reject a non-positive cut, which would silently return nothing."""
    if top_n <= 0:
        raise RerankError(f"top_n must be positive, got {top_n}")
