"""
PDF → plain text parser (per page) + naive chunker.

Keeps page numbers so the RAG answer can cite `{arxiv_id} p.{page}`.
Uses `pypdf` (pure Python, no poppler binding).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pypdf import PdfReader

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    arxiv_id: str
    page: int
    chunk_index: int
    text: str


def _clean(s: str) -> str:
    """Collapse whitespace and drop boilerplate glyphs from PDF text."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number_1_indexed, page_text)."""
    out: list[tuple[int, str]] = []
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        log.warning("pdf.open.failed", path=pdf_path, error=str(exc))
        return []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: PERF203 — one bad page shouldn't kill the paper
            text = ""
        text = _clean(text)
        if text:
            out.append((idx, text))
    return out


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Whitespace-tokenised sliding-window chunker.

    `chunk_size` / `chunk_overlap` are in pseudo-tokens (whitespace-separated
    units), a good enough proxy for real tokens at this scale — we prioritise
    speed and zero external deps over exactness.
    """
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if len(window) == 0:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_paper(
    arxiv_id: str,
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    pages = parse_pdf_pages(pdf_path)
    if not pages:
        log.warning("pdf.no_pages", arxiv_id=arxiv_id)
        return []

    all_chunks: list[Chunk] = []
    global_idx = 0
    for page_num, page_text in pages:
        for piece in chunk_text(page_text, chunk_size, chunk_overlap):
            if len(piece.split()) < min_chunk_tokens:
                continue
            all_chunks.append(
                Chunk(
                    arxiv_id=arxiv_id,
                    page=page_num,
                    chunk_index=global_idx,
                    text=piece,
                )
            )
            global_idx += 1
    log.info("pdf.chunked", arxiv_id=arxiv_id, pages=len(pages), chunks=len(all_chunks))
    return all_chunks
