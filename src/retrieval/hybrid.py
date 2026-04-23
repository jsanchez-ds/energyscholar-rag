"""
Hybrid retrieval — combines dense (Qdrant cosine) and sparse (BM25) results
via Reciprocal Rank Fusion, then reranks the top-N with a cross-encoder.

The BM25 index is built in-memory from Qdrant's payloads on first use
(fine for ~10k chunks; swap to a dedicated sparse store when you outgrow it).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from rank_bm25 import BM25Okapi

from src.embedding.embedder import Embedder
from src.utils.config import get_env
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    arxiv_id: str
    page: int
    chunk_index: int
    text: str
    score: float
    title: str = ""
    authors: list[str] | None = None
    pdf_url: str = ""


def _tokenize(s: str) -> list[str]:
    return [t for t in s.lower().split() if t.isalnum() or len(t) > 2]


class HybridRetriever:
    """Dense + sparse + cross-encoder reranker, with Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        collection: str | None = None,
        url: str | None = None,
    ) -> None:
        env = get_env()
        self.collection = collection or env.qdrant_collection
        self.qdrant = QdrantClient(url=url or env.qdrant_url)
        self.embedder = embedder or Embedder()

    # ── Public API ─────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        rrf_k: int = 60,
        rerank: bool = True,
        rerank_model: str | None = None,
    ) -> list[RetrievedChunk]:
        vector_hits = self._vector_search(query, top_k_vector)
        bm25_hits = self._bm25_search(query, top_k_bm25)

        fused = self._reciprocal_rank_fusion(vector_hits, bm25_hits, rrf_k)
        if rerank and fused:
            fused = self._rerank(query, fused, rerank_model)
        return fused[:top_k_final]

    # ── Internals ──────────────────────────────────────────────────────────
    def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        points: list[ScoredPoint] = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qvec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [_to_chunk(p.payload, float(p.score)) for p in points]

    @lru_cache(maxsize=1)
    def _load_all_chunks(self) -> tuple[list[dict], BM25Okapi]:
        """Scroll all payloads from Qdrant once; build in-memory BM25."""
        all_payloads: list[dict] = []
        offset = None
        while True:
            points, offset = self.qdrant.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            all_payloads.extend(p.payload for p in points)
            if offset is None:
                break

        tokenised = [_tokenize(p.get("text", "")) for p in all_payloads]
        bm25 = BM25Okapi(tokenised)
        log.info("bm25.built", docs=len(all_payloads))
        return all_payloads, bm25

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        payloads, bm25 = self._load_all_chunks()
        if not payloads:
            return []
        scores = bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [_to_chunk(payloads[i], float(scores[i])) for i in top_idx]

    def _reciprocal_rank_fusion(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        """RRF: each list contributes 1/(k + rank) to its candidate's fused score."""
        fused: dict[tuple[str, int], tuple[RetrievedChunk, float]] = {}
        for ranked_list in (dense, sparse):
            for rank, chunk in enumerate(ranked_list, start=1):
                key = (chunk.arxiv_id, chunk.chunk_index)
                add = 1.0 / (k + rank)
                if key in fused:
                    prev_chunk, prev_score = fused[key]
                    fused[key] = (prev_chunk, prev_score + add)
                else:
                    fused[key] = (chunk, add)

        merged = [
            RetrievedChunk(
                arxiv_id=c.arxiv_id,
                page=c.page,
                chunk_index=c.chunk_index,
                text=c.text,
                score=score,
                title=c.title,
                authors=c.authors,
                pdf_url=c.pdf_url,
            )
            for c, score in fused.values()
        ]
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        rerank_model: str | None,
    ) -> list[RetrievedChunk]:
        from sentence_transformers import CrossEncoder

        model_name = rerank_model or get_env().rerank_model
        ce = _load_cross_encoder(model_name)
        pairs = [(query, c.text) for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        reranked = sorted(
            (
                RetrievedChunk(
                    arxiv_id=c.arxiv_id,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    score=float(s),
                    title=c.title,
                    authors=c.authors,
                    pdf_url=c.pdf_url,
                )
                for c, s in zip(candidates, scores, strict=True)
            ),
            key=lambda x: x.score,
            reverse=True,
        )
        return reranked


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    log.info("reranker.load", model=name)
    return CrossEncoder(name)


def _to_chunk(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        arxiv_id=payload.get("arxiv_id", ""),
        page=int(payload.get("page", 0)),
        chunk_index=int(payload.get("chunk_index", 0)),
        text=payload.get("text", ""),
        score=score,
        title=payload.get("title", ""),
        authors=payload.get("authors"),
        pdf_url=payload.get("pdf_url", ""),
    )
