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


class IngestConfig(_Section):
    """Corpus discovery plus the chunking and embedding blocks."""

    source_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    include_extensions: tuple[str, ...] = (".md", ".markdown", ".txt")
    exclude_globs: tuple[str, ...] = ("**/.*", "**/node_modules/**")
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    # Ingest is the only stage that spends real money, so skipping unchanged
    # content is the default rather than an optimisation flag.
    incremental: bool = True
    hash_algorithm: str = "sha256"


class DenseVectorConfig(_Section):
    """The single named vector M1 writes."""

    name: str = "dense"
    size: int = Field(default=1536, gt=0)
    distance: str = "Cosine"


class VectorsConfig(_Section):
    """Vector topology. Only the dense side exists in M1 (sparse is M2)."""

    dense: DenseVectorConfig = DenseVectorConfig()


class HnswConfig(_Section):
    """Index build parameters, passed through to Qdrant unchanged."""

    m: int = Field(default=16, ge=0)
    ef_construct: int = Field(default=128, gt=0)
    full_scan_threshold: int = Field(default=10_000, ge=0)


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


class YamlConfig(_Section):
    """The whole file, with only the blocks M1 consumes typed out.

    Blocks belonging to later milestones (``retrieval``, ``rerank``,
    ``generation``, ``evals``, ``observability``) are intentionally absent and
    ignored rather than modelled, so this class describes what the code actually
    reads.
    """

    ingest: IngestConfig = IngestConfig()
    qdrant: QdrantConfig = QdrantConfig()


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
