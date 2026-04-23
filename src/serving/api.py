"""
FastAPI app for EnergyScholar.

Endpoints:
  - GET  /health          — liveness
  - GET  /info            — which provider / model is live
  - POST /query           — RAG answer + citations
  - POST /retrieve        — retrieval only (for debugging / eval)
  - GET  /metrics         — Prometheus metrics
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.generation.rag import answer
from src.llm.client import LLMClient
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import get_env
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

REQUESTS_TOTAL = Counter("rag_requests_total", "Total RAG requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "Request latency", ["endpoint"])

app = FastAPI(
    title="EnergyScholar API",
    version="0.1.0",
    description="RAG over arXiv energy-forecasting papers.",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)


class CitationModel(BaseModel):
    arxiv_id: str
    page: int
    title: str
    pdf_url: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    model: str
    provider: str
    n_context_chunks: int


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k_final: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    env = get_env()
    return {
        "provider": env.llm_provider,
        "embed_model": env.embed_model,
        "rerank_model": env.rerank_model,
        "qdrant_collection": env.qdrant_collection,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    with REQUEST_LATENCY.labels(endpoint="query").time():
        try:
            resp = answer(req.question)
        except Exception as exc:
            REQUESTS_TOTAL.labels(endpoint="query", status="error").inc()
            log.exception("query.failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    REQUESTS_TOTAL.labels(endpoint="query", status="ok").inc()
    return QueryResponse(
        answer=resp.answer,
        citations=[CitationModel(**c.__dict__) for c in resp.citations],
        model=resp.model,
        provider=resp.provider,
        n_context_chunks=resp.n_context_chunks,
    )


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> list[dict]:
    with REQUEST_LATENCY.labels(endpoint="retrieve").time():
        retriever = HybridRetriever()
        chunks = retriever.search(req.question, top_k_final=req.top_k_final)
    REQUESTS_TOTAL.labels(endpoint="retrieve", status="ok").inc()
    return [
        {
            "arxiv_id": c.arxiv_id,
            "page": c.page,
            "title": c.title,
            "score": c.score,
            "text_preview": c.text[:400],
        }
        for c in chunks
    ]


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


configure_logging(level=get_env().log_level)
