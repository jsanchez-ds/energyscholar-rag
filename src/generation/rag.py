"""
RAG orchestrator: retrieve → format context → call LLM → return answer + citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.llm.client import LLMClient, Message
from src.retrieval.hybrid import HybridRetriever, RetrievedChunk
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """\
You are EnergyScholar, a research assistant for the energy-forecasting literature.

RULES
  1. Answer ONLY from the supplied context. If the context doesn't answer the
     question, say "I don't have enough information in the indexed papers to answer that."
  2. Every factual claim must be followed by an inline citation of the form
     [arxiv_id, p.N] using the exact values from the context blocks.
  3. Prefer concise, structured answers. Bullet points are welcome for lists.
  4. If multiple papers disagree, note the disagreement and cite both.
  5. Never fabricate arxiv IDs or page numbers.
"""


@dataclass
class Citation:
    arxiv_id: str
    page: int
    title: str
    pdf_url: str


@dataclass
class RagResponse:
    answer: str
    citations: list[Citation]
    model: str
    provider: str
    n_context_chunks: int


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] arxiv_id={c.arxiv_id} p.{c.page} — {c.title}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _dedup_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for c in chunks:
        key = (c.arxiv_id, c.page)
        if key not in seen:
            seen.add(key)
            out.append(Citation(arxiv_id=c.arxiv_id, page=c.page, title=c.title, pdf_url=c.pdf_url))
    return out


def answer(
    question: str,
    retriever: HybridRetriever | None = None,
    llm: LLMClient | None = None,
) -> RagResponse:
    cfg = load_yaml_config()
    retriever = retriever or HybridRetriever()
    llm = llm or LLMClient()

    chunks = retriever.search(
        question,
        top_k_vector=cfg["retrieval"]["top_k_vector"],
        top_k_bm25=cfg["retrieval"]["top_k_bm25"],
        top_k_final=cfg["retrieval"]["top_k_final"],
        rrf_k=cfg["retrieval"]["rrf_k"],
        rerank=True,
        rerank_model=cfg["retrieval"]["rerank_model"],
    )
    log.info("rag.retrieved", n=len(chunks), question=question[:120])

    if not chunks:
        return RagResponse(
            answer="I don't have enough information in the indexed papers to answer that.",
            citations=[],
            model=llm.model,
            provider=llm.provider,
            n_context_chunks=0,
        )

    context = _format_context(chunks)
    user_msg = f"QUESTION:\n{question}\n\nCONTEXT BLOCKS:\n{context}"
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_msg),
    ]
    completion = llm.chat(messages)

    return RagResponse(
        answer=completion.text.strip(),
        citations=_dedup_citations(chunks),
        model=completion.model,
        provider=completion.provider,
        n_context_chunks=len(chunks),
    )


def answer_as_dict(question: str) -> dict:
    """Convenience wrapper — JSON-serialisable response."""
    resp = answer(question)
    return {
        "answer": resp.answer,
        "citations": [asdict(c) for c in resp.citations],
        "model": resp.model,
        "provider": resp.provider,
        "n_context_chunks": resp.n_context_chunks,
    }
