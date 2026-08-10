"""FastAPI dependencies shared by the routers.

Handlers never call :func:`production_rag.config.get_settings` directly; they
annotate a parameter with :data:`SettingsDep`. That indirection is what lets a
test build an app around arbitrary settings via ``app.dependency_overrides``
without touching the environment or the ``.env`` file on disk.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from production_rag.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Injected application settings.

Usage::

    @router.get("/health")
    async def health(settings: SettingsDep) -> HealthResponse: ...
"""

__all__ = ["Settings", "SettingsDep", "get_settings"]
