"""Runtime configuration for production-rag.

One source of truth for settings: the process environment, optionally seeded
by a local ``.env`` file. There is no module-level mutable state and no
``os.environ`` lookup anywhere else in the codebase — every consumer receives
a :class:`Settings` instance through dependency injection, which is what makes
the API testable without a live environment.

Secrets are never rendered. :meth:`Settings.safe_dump` is the only sanctioned
way to serialise settings for logs or debug endpoints.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


class Settings(BaseSettings):
    """Application settings, read from the environment.

    Precedence (highest first): constructor arguments, environment variables,
    ``.env`` file, field defaults. Unknown environment keys are ignored rather
    than rejected: the same ``.env`` is shared with Docker Compose and the
    Qdrant container, which define variables this process has no interest in.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    #: Fields whose value must never reach a log line, a response body or the
    #: second brain. Enforced by :meth:`safe_dump`.
    SECRET_FIELDS: ClassVar[frozenset[str]] = frozenset({"openai_api_key"})

    # --- application identity -------------------------------------------------
    app_name: str = "production-rag"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/v1"

    # --- HTTP server ----------------------------------------------------------
    # 0.0.0.0 is the correct default for a containerised service: binding to
    # localhost would make the port unreachable from outside the container.
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    # --- retrieval backend ----------------------------------------------------
    # Declared now, dialled from M1 onwards. M0 only reports whether a target
    # is configured (see the /v1/ready route) — it opens no sockets.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # --- optional file-based configuration ------------------------------------
    # Compose passes CONFIG_PATH=/app/configs/default.yaml. The YAML layer
    # (chunking, retrieval and rerank knobs) is consumed from M1; accepting the
    # value here keeps the container contract explicit instead of silently
    # dropping it via ``extra="ignore"``.
    config_path: str | None = None

    # --- credentials ----------------------------------------------------------
    # Never logged, never echoed in a response. Absent by default so the whole
    # M0 test suite runs with no credentials present.
    openai_api_key: str | None = None

    @field_validator("api_prefix")
    @classmethod
    def _normalise_api_prefix(cls, value: str) -> str:
        """Force a single leading slash and no trailing slash, e.g. ``/v1``.

        A prefix of ``/`` is rejected: the unversioned ``/health`` route and the
        versioned one would then collide on the same path, which is a
        configuration bug that is far cheaper to catch at startup than in
        production.
        """
        prefix = "/" + value.strip().strip("/")
        if prefix == "/":
            raise ValueError("api_prefix must name a version segment, for example '/v1'")
        return prefix

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Accept any case, reject names the stdlib does not know."""
        level = value.strip().upper()
        if level not in _VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
            raise ValueError(f"log_level must be one of: {allowed}")
        return level

    @property
    def log_level_number(self) -> int:
        """The configured level as a stdlib ``logging`` integer."""
        return logging.getLevelNamesMapping()[self.log_level]

    @property
    def qdrant_configured(self) -> bool:
        """Whether a plausible Qdrant endpoint is configured.

        This is a *configuration* check, not a connectivity check: it answers
        "was this deployment told where the vector store lives?" without any
        network IO, so readiness stays instant and offline-testable. M1 adds a
        real round-trip on top of this signal.
        """
        url = self.qdrant_url.strip()
        return url.startswith(("http://", "https://"))

    def safe_dump(self) -> dict[str, object]:
        """Settings as a plain dict with every secret value masked.

        Use this — never ``model_dump()`` — when settings are written to a log,
        a debug endpoint or a documentation file.
        """
        data: dict[str, object] = dict(self.model_dump())
        for name in self.SECRET_FIELDS:
            if data.get(name) is not None:
                data[name] = "***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached because parsing ``.env`` on every request would be pure waste and
    because FastAPI keys dependency overrides by the callable itself: tests
    swap the whole object out via ``app.dependency_overrides``, and anything
    constructing settings directly calls ``get_settings.cache_clear()`` first.
    """
    return Settings()
