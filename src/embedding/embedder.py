"""
Sentence-transformers wrapper with batching + lazy model load.

Kept deliberately minimal: one model, one dimension, CPU-friendly.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.utils.config import get_env
from src.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):  # -> SentenceTransformer (avoid import at module level)
    from sentence_transformers import SentenceTransformer

    log.info("embedder.load", model=model_name)
    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name or get_env().embed_model
        self.batch_size = batch_size

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        model = _load_model(self.model_name)
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        model = _load_model(self.model_name)
        return int(model.get_sentence_embedding_dimension())
