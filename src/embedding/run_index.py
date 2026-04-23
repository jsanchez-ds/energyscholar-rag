"""
Entry-point: `python -m src.embedding.run_index`

Reads metadata + chunks every paper in `data/papers/`, embeds each chunk
with a local sentence-transformers model, and pushes them into a Qdrant
collection (recreating it if it doesn't exist with the right dimension).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.embedding.embedder import Embedder
from src.ingestion.arxiv_client import load_metadata
from src.ingestion.pdf_parser import Chunk, chunk_paper
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger
from src.utils.qdrant import get_qdrant

log = get_logger(__name__)


def _ensure_collection(client, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        current_dim = info.config.params.vectors.size
        if current_dim == dim:
            log.info("qdrant.collection.exists", name=name, dim=dim)
            return
        log.warning("qdrant.collection.dim_mismatch", name=name, expected=dim, actual=current_dim)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("qdrant.collection.created", name=name, dim=dim)


def _point_id(chunk: Chunk) -> str:
    """Deterministic UUID so re-runs upsert in place."""
    raw = f"{chunk.arxiv_id}:{chunk.chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def run() -> None:
    env = get_env()
    cfg = load_yaml_config()

    papers = load_metadata(Path(env.cache_dir) / "papers.jsonl")
    log.info("index.start", papers=len(papers))
    if not papers:
        log.warning("index.no_papers — run `python -m src.ingestion.run_arxiv` first")
        return

    chunking = cfg["chunking"]
    all_chunks: list[Chunk] = []
    paper_lookup: dict[str, dict] = {}
    for meta in papers:
        chunks = chunk_paper(
            arxiv_id=meta.arxiv_id,
            pdf_path=meta.pdf_path,
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
            min_chunk_tokens=chunking["min_chunk_tokens"],
        )
        all_chunks.extend(chunks)
        paper_lookup[meta.arxiv_id] = {
            "title": meta.title,
            "authors": meta.authors,
            "published": meta.published,
            "primary_category": meta.primary_category,
            "pdf_url": meta.pdf_url,
        }

    log.info("index.chunks_built", n=len(all_chunks))
    if not all_chunks:
        log.warning("index.no_chunks")
        return

    embedder = Embedder()
    vecs = embedder.embed([c.text for c in all_chunks], normalize=cfg["embedding"]["normalize"])
    log.info("index.embedded", n=len(vecs), dim=vecs.shape[1])

    client = get_qdrant()
    _ensure_collection(client, env.qdrant_collection, dim=vecs.shape[1])

    points = [
        PointStruct(
            id=_point_id(chunk),
            vector=vecs[i].tolist(),
            payload={
                "arxiv_id": chunk.arxiv_id,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                **paper_lookup.get(chunk.arxiv_id, {}),
            },
        )
        for i, chunk in enumerate(all_chunks)
    ]

    # Upsert in batches to keep memory flat
    BATCH = 256
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=env.qdrant_collection, points=points[i : i + BATCH])
    log.info("index.upserted", n=len(points), collection=env.qdrant_collection)


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    run()
