"""Unit tests for the LangGraph query path and the public ``run_query`` entry.

Offline throughout: an in-memory store, the fake embedder and the fake LLM. No
Qdrant, no OpenAI, no model download — the whole answer path runs on a laptop
with the wifi off, which is the property that makes this repository runnable
from a fresh clone.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from production_rag.config_loader import (
    CitationsConfig,
    GenerationConfig,
    RerankConfig,
    RetrievalConfig,
    YamlConfig,
)
from production_rag.generation import guardrails
from production_rag.generation.llm import FakeLLM, LLMResponse
from production_rag.generation.prompts import ChatMessage
from production_rag.graph.build import build_query_graph, run_graph
from production_rag.graph.nodes import QueryDeps, rerank_node, should_generate
from production_rag.graph.state import (
    CITE_NODE,
    FINALISE_NODE,
    GENERATE_NODE,
    GUARD_NODE,
    NODE_NAMES,
    RERANK_NODE,
    RETRIEVE_NODE,
    QueryState,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.query_pipeline import build_query_pipeline, run_query
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import FakeReranker
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

RUN_ID = "run-graph-test"

CORPUS = {
    "sample/00-intro.md": "Production RAG systems combine retrieval and generation.",
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-fusion.md": "Reciprocal rank fusion merges two ranked lists by position.",
    "sample/03-chunking.md": "The chunker splits markdown documents on heading boundaries.",
    "sample/04-flags.md": "Pass --recreate-collection to rebuild the index from scratch.",
}


def _retriever(*, corpus: dict[str, str] | None = None, **overrides: object) -> Retriever:
    documents = CORPUS if corpus is None else corpus
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    chunks = [
        Chunk.build(
            document=Document(source_path=path, text=text, title=path, source="sample"),
            chunk_index=index,
            text=text,
            embed_text=text,
        )
        for index, (path, text) in enumerate(documents.items())
    ]
    texts = [chunk.embed_text for chunk in chunks]
    sparse_vectors = None
    if texts:
        encoder = Bm25Encoder()
        encoder.fit(texts)
        sparse_vectors = encoder.encode_documents(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id=RUN_ID,
        embedded_model=embedder.model,
        sparse_vectors=sparse_vectors,
    )
    return Retriever(
        store=store,
        embedder=embedder,
        config=RetrievalConfig(**overrides),  # type: ignore[arg-type]
    )


def _empty_retriever() -> Retriever:
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    return Retriever(store=store, embedder=embedder)


class ExplodingLLM:
    """Fails the test if generation runs. Proves the refusal edge was taken."""

    @property
    def model(self) -> str:
        return "exploding"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise AssertionError("the generate node must not run on the refusal path")


class TestHappyPath:
    def test_a_query_gets_a_cited_answer(self) -> None:
        result = run_query(
            "How does reciprocal rank fusion work?",
            retriever=_retriever(),
            llm=FakeLLM(),
        )
        assert result.refused is False
        assert result.refusal_reason is None
        assert result.citations
        assert all(citation.source_path for citation in result.citations)

    def test_the_answer_only_cites_chunks_that_were_retrieved(self) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM())
        assert all(citation.source_path in CORPUS for citation in result.citations)
        assert result.invalid_markers == ()

    def test_every_node_reports_its_own_timing(self) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM())
        assert set(result.timings_ms or {}) == set(NODE_NAMES)
        assert result.total_ms >= 0.0

    def test_the_result_records_what_produced_it(self) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM())
        assert result.mode == "hybrid"
        assert result.collection == "production_rag"
        assert result.embedded_model == FakeEmbeddingProvider().model
        assert result.model == FakeLLM().model

    def test_the_result_serialises_to_a_response_shape(self) -> None:
        payload = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM()).to_dict()
        assert payload["refused"] is False
        assert payload["citations"]
        assert set(payload["latency_ms"]) == set(NODE_NAMES)
        assert payload["rerank"]["applied"] is False


class TestRefusalEdge:
    def test_an_empty_index_refuses_without_reaching_the_model(self) -> None:
        result = run_query("anything at all", retriever=_empty_retriever(), llm=ExplodingLLM())
        assert result.refused is True
        assert result.refusal_reason == guardrails.REASON_NO_EVIDENCE
        assert result.citations == ()

    def test_the_refusal_path_stops_at_the_guard(self) -> None:
        result = run_query("anything at all", retriever=_empty_retriever(), llm=ExplodingLLM())
        assert set(result.timings_ms or {}) == {RETRIEVE_NODE, RERANK_NODE, GUARD_NODE}
        assert GENERATE_NODE not in (result.timings_ms or {})

    def test_the_refusal_message_is_configurable(self) -> None:
        config = YamlConfig(
            generation=GenerationConfig(
                citations=CitationsConfig(refusal_message="nothing indexed says so")
            )
        )
        result = run_query(
            "anything at all",
            retriever=_empty_retriever(),
            llm=ExplodingLLM(),
            config=config,
        )
        assert result.answer == "nothing indexed says so"

    def test_the_conditional_edge_reads_the_recorded_decision(self) -> None:
        assert should_generate(QueryState(query="q")) == "generate"
        assert should_generate(QueryState(query="q", refused=True)) == "refused"


class TestRetrievalOptions:
    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
    def test_every_mode_reaches_the_retriever(self, mode: str) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM(), mode=mode)
        assert result.mode == mode

    def test_top_k_bounds_what_is_retrieved(self) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM(), top_k=2)
        assert result.hits_retrieved == 2

    def test_an_unknown_mode_is_an_error_not_a_refusal(self) -> None:
        from production_rag.retrieval.hybrid import RetrievalError

        with pytest.raises(RetrievalError):
            run_query("q", retriever=_retriever(), llm=FakeLLM(), mode="magic")


class TestRerankNode:
    def test_rerank_is_off_by_default(self) -> None:
        result = run_query("qdrant vectors", retriever=_retriever(), llm=FakeLLM())
        assert result.rerank is None
        assert (result.timings_ms or {})[RERANK_NODE] == 0.0

    def test_a_reranker_passed_to_the_pipeline_is_applied(self) -> None:
        result = run_query(
            "qdrant vectors",
            retriever=_retriever(),
            llm=FakeLLM(),
            reranker=FakeReranker(),
        )
        assert result.rerank is not None
        assert result.rerank["applied"] is True
        assert result.rerank["reranker"] == "fake"

    def test_the_rerank_top_k_bounds_the_hits(self) -> None:
        config = YamlConfig(rerank=RerankConfig(top_k=2))
        result = run_query(
            "qdrant vectors",
            retriever=_retriever(),
            llm=FakeLLM(),
            reranker=FakeReranker(),
            config=config,
        )
        assert result.hits_retrieved == 2

    def test_a_retriever_that_already_reranked_is_not_reranked_twice(self) -> None:
        retriever = _retriever()
        retriever._reranker = FakeReranker()  # noqa: SLF001 - exercising the guard
        deps = QueryDeps(
            retriever=retriever,
            llm=FakeLLM(),
            generation=GenerationConfig(),
            reranker=FakeReranker(),
        )
        state = QueryState(query="qdrant vectors")
        after_retrieve = run_graph(build_query_graph(deps), state)
        assert after_retrieve.timings_ms[RERANK_NODE] == 0.0

    def test_the_node_stands_down_when_no_reranker_is_configured(self) -> None:
        deps = QueryDeps(retriever=_retriever(), llm=FakeLLM(), generation=GenerationConfig())
        state = QueryState(query="qdrant vectors")
        updates = rerank_node(state, deps)
        assert updates == {"timings_ms": {RERANK_NODE: 0.0}}


class TestPipelineReuse:
    def test_one_compiled_graph_answers_many_queries(self) -> None:
        pipeline = build_query_pipeline(retriever=_retriever(), llm=FakeLLM())
        first = pipeline.run("qdrant vectors")
        second = pipeline.run("how does fusion merge lists?")
        assert first.refused is False
        assert second.refused is False
        assert first.answer != second.answer

    def test_the_pipeline_exposes_what_it_was_built_against(self) -> None:
        pipeline = build_query_pipeline(retriever=_retriever(), llm=FakeLLM())
        assert isinstance(pipeline.deps, QueryDeps)
        assert pipeline.deps.rerank.enabled is False


class TestState:
    def test_timings_accumulate_without_mutating_the_state(self) -> None:
        state = QueryState(query="q", timings_ms={RETRIEVE_NODE: 1.0})
        merged = state.timed(GENERATE_NODE, 2.0)
        assert merged == {RETRIEVE_NODE: 1.0, GENERATE_NODE: 2.0}
        assert state.timings_ms == {RETRIEVE_NODE: 1.0}

    def test_the_summary_is_json_safe(self) -> None:
        import json

        pipeline = build_query_pipeline(retriever=_retriever(), llm=FakeLLM())
        pipeline.run("qdrant vectors")
        json.dumps(QueryState(query="q").summary())

    def test_the_node_names_are_the_timing_keys(self) -> None:
        assert NODE_NAMES == (
            RETRIEVE_NODE,
            RERANK_NODE,
            GUARD_NODE,
            GENERATE_NODE,
            CITE_NODE,
            FINALISE_NODE,
        )
