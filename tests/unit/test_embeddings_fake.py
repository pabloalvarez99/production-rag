"""The deterministic embedder and the provider factory.

No network is touched here. The OpenAI provider is only exercised where it fails
before making a call — a real request belongs to an integration test, not to a
suite that must be green with the wifi off.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from production_rag.config_loader import EmbeddingConfig
from production_rag.retrieval.embeddings import (
    FAKE_DIMENSIONS,
    FAKE_MODEL_NAME,
    EmbeddingError,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedder,
)

CONFIG = EmbeddingConfig()


def test_fake_provider_reports_its_identity() -> None:
    provider = FakeEmbeddingProvider()
    assert provider.model == FAKE_MODEL_NAME
    assert provider.dimensions == FAKE_DIMENSIONS
    # A fake-embedded collection must be recognisable as such after the fact.
    assert "fake" in provider.model


def test_fake_provider_satisfies_the_protocol() -> None:
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)


def test_vectors_have_the_configured_length() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)
    vectors = provider.embed_documents(["a", "b", "c"])
    assert [len(vector) for vector in vectors] == [8, 8, 8]


def test_vectors_are_unit_length() -> None:
    # Cosine distance on a non-normalised vector still works, but a unit vector
    # makes the dot product the similarity, which is what the store assumes.
    for vector in FakeEmbeddingProvider().embed_documents(["alpha", "beta"]):
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-9)


def test_same_text_gives_the_same_vector() -> None:
    provider = FakeEmbeddingProvider()
    assert provider.embed_documents(["hello"]) == provider.embed_documents(["hello"])


def test_different_text_gives_a_different_vector() -> None:
    first, second = FakeEmbeddingProvider().embed_documents(["hello", "goodbye"])
    assert first != second


def test_query_and_document_paths_agree() -> None:
    provider = FakeEmbeddingProvider()
    assert provider.embed_query("hello") == provider.embed_documents(["hello"])[0]


def test_empty_input_yields_no_vectors() -> None:
    assert FakeEmbeddingProvider().embed_documents([]) == []


def test_vectors_are_stable_across_processes() -> None:
    # Built on hashlib, not hash(), which is salted per interpreter. If this ever
    # regresses, ingest and query in different processes would embed the same
    # text differently and retrieval would return noise.
    code = (
        "from production_rag.retrieval.embeddings import FakeEmbeddingProvider;"
        "print(FakeEmbeddingProvider(dimensions=4).embed_query('stability'))"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    in_process = FakeEmbeddingProvider(dimensions=4).embed_query("stability")
    assert result.stdout.strip() == str(in_process)


def test_factory_builds_the_fake_provider_without_a_credential() -> None:
    provider = build_embedder("fake", config=CONFIG)
    assert provider.model == FAKE_MODEL_NAME


def test_factory_honours_a_custom_fake_dimension() -> None:
    assert build_embedder("fake", config=CONFIG, fake_dimensions=64).dimensions == 64


def test_factory_builds_the_openai_provider_from_a_key() -> None:
    provider = build_embedder("openai", config=CONFIG, api_key="test-key-not-real")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == CONFIG.model
    assert provider.dimensions == CONFIG.dimensions


def test_openai_provider_refuses_to_start_without_a_credential() -> None:
    with pytest.raises(EmbeddingError, match="--embedder fake"):
        build_embedder("openai", config=CONFIG, api_key=None)


def test_credential_never_appears_in_the_repr() -> None:
    provider = OpenAIEmbeddingProvider(api_key="super-secret-value")
    assert "super-secret-value" not in repr(provider)


def test_unknown_embedder_is_rejected_by_name() -> None:
    with pytest.raises(EmbeddingError, match="unknown embedder"):
        build_embedder("word2vec", config=CONFIG)
