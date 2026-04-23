"""Tests for the RAG context/citation formatting (no LLM calls)."""

from src.generation.rag import _dedup_citations, _format_context
from src.retrieval.hybrid import RetrievedChunk


def test_format_context_includes_ids_and_pages(sample_retrieved: list[RetrievedChunk]) -> None:
    ctx = _format_context(sample_retrieved)
    assert "arxiv_id=2301.00001" in ctx
    assert "p.1" in ctx
    assert "arxiv_id=2302.00002" in ctx
    assert "p.3" in ctx
    # The separator between blocks must be present
    assert "---" in ctx


def test_dedup_citations_merges_same_paper_same_page() -> None:
    dup = RetrievedChunk(
        arxiv_id="2301.00001", page=1, chunk_index=99, score=0.5,
        text="different chunk, same page", title="Transformers for Energy Forecasting",
        pdf_url="https://arxiv.org/pdf/2301.00001",
    )
    sample = [
        RetrievedChunk(arxiv_id="2301.00001", page=1, chunk_index=0, score=0.9, text="t", title="A"),
        dup,
        RetrievedChunk(arxiv_id="2302.00002", page=3, chunk_index=0, score=0.8, text="u", title="B"),
    ]
    cites = _dedup_citations(sample)
    assert len(cites) == 2
    ids = {(c.arxiv_id, c.page) for c in cites}
    assert ids == {("2301.00001", 1), ("2302.00002", 3)}
