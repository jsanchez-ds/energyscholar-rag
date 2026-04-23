"""Qdrant client factory — supports embedded (no server) and server modes."""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from src.utils.config import get_env
from src.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Return a Qdrant client in the mode configured via env."""
    env = get_env()
    if env.qdrant_mode == "server":
        log.info("qdrant.mode.server", url=env.qdrant_url)
        return QdrantClient(url=env.qdrant_url)
    log.info("qdrant.mode.embedded", path=env.qdrant_path)
    return QdrantClient(path=env.qdrant_path)
