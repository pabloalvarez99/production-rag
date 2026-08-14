"""Typed loader for the declarative YAML configuration.

Two layers of configuration, with different jobs:

* :mod:`production_rag.config` — the environment. Endpoints, credentials, log
  level: everything that differs between a laptop, a container and production.
* this module — ``configs/default.yaml``. The knobs that describe *behaviour*:
  chunk size, which extensions to walk, vector topology. They belong in a file
  a reviewer can read as a whole and diff between experiments, not in a wall of
  environment variables.

Everything here has a default, so a missing or partial YAML file degrades to
"the documented defaults" instead of a crash — the ingest CLI has to work on a
fresh clone. Unknown keys are ignored on purpose: ``configs/default.yaml`` is
deliberately broader than what the current milestone consumes, and a key that
M4 will read must not make M1 fail.

Secrets never appear here. The YAML names the *environment variable* holding a
credential (``api_key_env``); the value is read from the environment.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

_log = structlog.get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
"""Repo-relative location of the tracked default profile."""


class ConfigFileError(RuntimeError):
    """A YAML profile exists but cannot be used."""


class _Section(BaseModel):
    """Base for every config block: immutable, and tolerant of unknown keys."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class ChunkingConfig(_Section):
    """How a document is split before embedding.

    Sizes are in **characters**, not tokens: character counts are free to
    compute, stable across tokenisers, and close enough at this granularity. The
    tokeniser-exact budget matters at generation time (M4), not here.
    ``chunk_size`` is a hard ceiling including overlap; text without a useful
    separator is cut to enforce it. ``min_chunk_chars`` applies only after a
    longer document has been split, never to a short whole document.
    """

    strategy: str = "recursive"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    min_chunk_chars: int = Field(default=120, ge=0)
    separators: tuple[str, ...] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")
    prepend_heading_context: bool = True

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_fits_in_a_chunk(cls, value: int, info: ValidationInfo) -> int:
        """Reject an overlap that cannot fit, which would loop or duplicate."""
        chunk_size = info.data.get("chunk_size")
        if isinstance(chunk_size, int) and value >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({value}) must be smaller than chunk_size ({chunk_size})"
            )
        return value


class EmbeddingConfig(_Section):
    """Which model turns a chunk into a vector, and how it is called."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=128, gt=0)
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = Field(default=5, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)


class SparseConfig(_Section):
    """BM25 side of the index (M2).

    ``enabled`` states the target, so a corpus ingested with sparse turned off is
    a deliberate, visible choice rather than a forgotten flag.
    """

    enabled: bool = True
    method: str = "bm25"
    k1: float = Field(default=1.2, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    lowercase: bool = True
    # A named set ("english") or "none". A wrong stopword list degrades recall
    # quietly, which is why the value is explicit in the file.
    stopwords: str = "english"


class IngestConfig(_Section):
    """Corpus discovery plus the chunking, embedding and sparse blocks."""

    source_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    include_extensions: tuple[str, ...] = (".md", ".markdown", ".txt")
    exclude_globs: tuple[str, ...] = ("**/.*", "**/node_modules/**")
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    sparse: SparseConfig = SparseConfig()
    # Ingest is the only stage that spends real money, so skipping unchanged
    # content is the default rather than an optimisation flag.
    incremental: bool = True
    hash_algorithm: str = "sha256"


class DenseVectorConfig(_Section):
    """The single named vector M1 writes."""

    name: str = "dense"
    size: int = Field(default=1536, gt=0)
    distance: str = "Cosine"


class SparseVectorConfig(_Section):
    """The named sparse vector M2 writes.

    ``modifier`` is read but **not** applied: ingest stores full BM25 weights
    (IDF included), so asking Qdrant for its own IDF on top would square the
    term. The field stays visible here because the config file declares it and
    silently ignoring a declared knob is worse than explaining it — see
    :mod:`production_rag.retrieval.sparse`.
    """

    name: str = "sparse"
    modifier: str = "none"


class VectorsConfig(_Section):
    """Vector topology: dense from M1, sparse added in M2."""

    dense: DenseVectorConfig = DenseVectorConfig()
    sparse: SparseVectorConfig = SparseVectorConfig()


class HnswConfig(_Section):
    """Index build parameters, passed through to Qdrant unchanged."""

    m: int = Field(default=16, ge=0)
    ef_construct: int = Field(default=128, gt=0)
    full_scan_threshold: int = Field(default=10_000, ge=0)


class SearchConfig(_Section):
    """Query-time index parameters.

    ``ef`` is a build-independent knob: it trades latency for recall on every
    search without touching the stored index, which is why it lives here and not
    in :class:`HnswConfig`.
    """

    hnsw_ef: int = Field(default=128, gt=0)
    exact: bool = False


class PayloadIndexConfig(_Section):
    """One keyword index so metadata filters stay cheap."""

    field: str
    schema_: str = Field(default="keyword", alias="schema")

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class QdrantConfig(_Section):
    """Vector store topology. The URL and credential come from the environment."""

    collection: str = "production_rag"
    api_key_env: str = "QDRANT_API_KEY"
    prefer_grpc: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0)
    vectors: VectorsConfig = VectorsConfig()
    hnsw: HnswConfig = HnswConfig()
    search: SearchConfig = SearchConfig()
    payload_indexes: tuple[PayloadIndexConfig, ...] = (
        PayloadIndexConfig(field="doc_id"),
        PayloadIndexConfig(field="source"),
        PayloadIndexConfig(field="tags"),
    )
    on_disk_payload: bool = True
    # "wait": a smoke test run straight after ingest must not be able to read a
    # half-built index.
    write_consistency: str = "wait"

    @property
    def wait_for_writes(self) -> bool:
        """Whether upserts block until the operation is applied."""
        return self.write_consistency == "wait"


class FusionConfig(_Section):
    """Reciprocal rank fusion parameters.

    Weights stay equal until an eval says otherwise; ``rrf.py`` keeps a
    zero-weighted branch visible so a single-branch ablation is a config change.
    """

    method: str = "rrf"
    k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.0, ge=0.0)


class FiltersConfig(_Section):
    """Metadata fields a query is allowed to filter on, as an allowlist.

    Live from M7: ``POST /v1/query`` and both CLIs accept a ``filters`` object and
    a field outside this list is rejected, never dropped. The enforcement itself
    is in :mod:`production_rag.retrieval.filters`; this block is only the list.

    ``created_at`` was in this default until M7 and is gone. Nothing in the ingest
    payload ever wrote it, so it could only ever have matched nothing — an
    allowlisted field that always returns an empty result is worse than an absent
    one, because the empty result looks like a corpus gap.
    """

    allowed_fields: tuple[str, ...] = ("source", "title", "tags")


class RetrievalConfig(_Section):
    """Query-side knobs. Live from M2 for retrieval; generation is still M4.

    ``dense_top_k`` and ``sparse_top_k`` are deliberately larger than ``top_k``:
    fusion can only reorder what it was given, so under-retrieving per branch is
    a recall loss no amount of fusion tuning can recover.
    """

    mode: str = "hybrid"
    dense_top_k: int = Field(default=40, gt=0)
    sparse_top_k: int = Field(default=40, gt=0)
    top_k: int = Field(default=12, gt=0)
    fusion: FusionConfig = FusionConfig()
    # 0.0 disables the floor. Raise only with eval evidence: a threshold set too
    # high turns a recall miss into a silent empty result.
    score_threshold: float = Field(default=0.0, ge=0.0)
    return_payload_fields: tuple[str, ...] = (
        "doc_id",
        "chunk_id",
        "title",
        "source_path",
        "heading_path",
        "text",
    )
    filters: FiltersConfig = FiltersConfig()


class RerankConfig(_Section):
    """Second-stage scoring applied after fusion (M3).

    ``input_top_k`` is larger than ``top_k`` on purpose: reranking is the stage
    that can recover a relevant chunk fusion buried at rank 30, and it can only
    recover what it is shown.

    ``fail_open`` decides what a reranker failure means. ``True`` degrades to the
    fusion order, because a slightly worse ranking beats no answer; a deployment
    that would rather fail loudly sets it to ``False``.
    """

    enabled: bool = False
    provider: str = "cohere"
    model: str = "rerank-english-v3.0"
    api_key_env: str = "COHERE_API_KEY"
    input_top_k: int = Field(default=40, gt=0)
    top_k: int = Field(default=6, gt=0)
    timeout_seconds: float = Field(default=15.0, gt=0)
    fail_open: bool = True
    # Local cross-encoder weights, used when the provider is the local one. Named
    # here rather than hard-coded so swapping the model is a config diff an eval
    # run can be attributed to.
    local_model: str = "BAAI/bge-reranker-base"


class CitationsConfig(_Section):
    """How an answer is required to point at its evidence (M4).

    ``require_citation`` and ``refuse_without_evidence`` are the two settings
    that decide whether this is a grounded RAG system or a chatbot with a
    retrieval step in front of it. Both default to on: an uncited answer and a
    guessed answer are the two failure modes a user cannot detect by reading.
    """

    enabled: bool = True
    style: str = "bracketed_index"
    require_citation: bool = True
    refuse_without_evidence: bool = True
    refusal_message: str = "I could not find support for that in the indexed documents."


class PromptConfig(_Section):
    """Prompt assembly knobs (M4).

    ``max_chunks_in_prompt`` is a separate ceiling from ``retrieval.top_k``:
    retrieval decides what is worth looking at, the prompt decides what fits.
    Truncation is from the tail, so retrieval order is also truncation order.
    """

    system_path: str = "configs/prompts/system.md"
    include_heading_path: bool = True
    max_chunks_in_prompt: int = Field(default=8, gt=0)


class GenerationConfig(_Section):
    """Answer synthesis (M4).

    ``temperature`` is near zero on purpose: this stage extracts and attributes
    what retrieval found, it does not author. A creative setting here shows up
    as citations that point at passages which do not say what the answer claims.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    # Streaming is live (M7): ``POST /v1/query/stream`` exists unconditionally,
    # because a route that appears and disappears with a profile is a contract
    # no client can code against. What this flag decides is the *default* the
    # demo form starts with, and it stays off so a first visit shows the plain
    # request/response path the API contract describes.
    stream: bool = False
    citations: CitationsConfig = CitationsConfig()
    prompt: PromptConfig = PromptConfig()


class LoggingConfig(_Section):
    """What the structured log is allowed to contain (M5).

    ``log_prompts`` and ``log_retrieved_text`` default to off and are the only
    two settings here that can leak the corpus. A prompt carries retrieved
    document text verbatim, so an aggregator with these on holds a copy of the
    corpus with none of its access controls — and it fails silently, because
    nothing errors when a log line is too informative. They exist for a local
    session on a corpus you own, never for a deployment.
    """

    level: str = "INFO"
    format: str = "json"
    log_prompts: bool = False
    log_retrieved_text: bool = False
    include_request_id: bool = True


class TracingConfig(_Section):
    """Opt-in trace export (M5).

    Off by default and never required: the library, the API, the CLI and the
    whole test suite run with nothing here configured. Only env var *names* are
    stored; the credentials themselves stay in the environment.

    ``sample_rate`` bounds volume, not sensitivity. Enabling this sends prompts
    and answers to whoever runs ``host_env``, which is a decision about where
    corpus text may travel rather than a verbosity knob.
    """

    enabled: bool = False
    provider: str = "langfuse"
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"  # noqa: S105 - a variable name, not a secret
    host_env: str = "LANGFUSE_HOST"
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class MetricsConfig(_Section):
    """Fleet-level metrics (declared, not served).

    Modelled so the block round-trips and so the bucket edges live somewhere
    typed; nothing exposes ``path`` yet. Per-request latency is already on the
    result and in the logs, and is a different question from this one.
    """

    enabled: bool = True
    path: str = "/metrics"
    latency_buckets_seconds: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


class ObservabilityConfig(_Section):
    """Logging, tracing and metrics (M5)."""

    logging: LoggingConfig = LoggingConfig()
    tracing: TracingConfig = TracingConfig()
    metrics: MetricsConfig = MetricsConfig()


class CacheConfig(_Section):
    """Optional in-process result cache (M7).

    Off by default so a production-shaped profile never serves a shared answer
    across workers it did not design for. The local demo turns this on with
    ``CACHE_ENABLED=true`` (compose) or ``enabled: true`` here. Filters are
    part of the key — see ``docs/adr/0013-filter-aware-query-cache.md``.
    """

    enabled: bool = False
    max_entries: int = Field(default=256, ge=1)


class YamlConfig(_Section):
    """The whole file, with only the blocks the code consumes typed out.

    Blocks belonging to later milestones (``evals``) are intentionally absent
    and ignored rather than modelled, so this class describes what the runtime
    actually reads.
    """

    ingest: IngestConfig = IngestConfig()
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    generation: GenerationConfig = GenerationConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    cache: CacheConfig = CacheConfig()


def load_yaml_config(path: str | Path | None = None) -> YamlConfig:
    """Load a YAML profile, falling back to the documented defaults.

    Args:
        path: Explicit profile path. When ``None``, ``configs/default.yaml``
            relative to the current directory is tried.

    Returns:
        A fully populated :class:`YamlConfig`. A missing file yields defaults;
        a file that exists but cannot be parsed is an error, because silently
        ignoring a typo in a config file is how an experiment ends up measuring
        the wrong settings.

    Raises:
        ConfigFileError: The file exists but is not valid YAML, or does not
            contain a mapping at the top level.
    """
    candidate = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not candidate.is_file():
        _log.info("config_file_absent", path=str(candidate), using="defaults")
        return YamlConfig()

    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"{candidate} is not valid YAML: {exc}") from exc

    if raw is None:
        return YamlConfig()
    if not isinstance(raw, dict):
        raise ConfigFileError(f"{candidate} must contain a mapping at the top level")

    _log.info("config_file_loaded", path=str(candidate))
    return YamlConfig.model_validate(raw)
