"""
arXiv client — fetches papers matching the configured queries, downloads
PDFs into `data/papers/`, and persists metadata.

Uses the `arxiv` Python package (official-friendly wrapper).
Idempotent: skips papers already on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str       # ISO date string
    updated: str
    categories: list[str]
    primary_category: str
    pdf_url: str
    pdf_path: str = ""
    tags: list[str] = field(default_factory=list)


def _slugify(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_one_query(
    query: str,
    max_results: int,
    categories: list[str] | None = None,
) -> list[PaperMeta]:
    """Run a single arXiv search query and return paper metadata."""
    if categories:
        cat_clause = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_clause})"
    else:
        full_query = query

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)

    results: list[PaperMeta] = []
    for r in client.results(search):
        results.append(
            PaperMeta(
                arxiv_id=r.get_short_id(),
                title=r.title.strip(),
                authors=[a.name for a in r.authors],
                abstract=r.summary.strip().replace("\n", " "),
                published=r.published.date().isoformat() if r.published else "",
                updated=r.updated.date().isoformat() if r.updated else "",
                categories=r.categories,
                primary_category=r.primary_category,
                pdf_url=r.pdf_url,
                tags=[query],
            )
        )
    log.info("arxiv.query.done", query=query, hits=len(results))
    return results


def search_papers(
    queries: list[str],
    max_papers_per_query: int,
    categories: list[str] | None = None,
) -> list[PaperMeta]:
    """Aggregate multiple queries and de-duplicate by arxiv_id."""
    seen: dict[str, PaperMeta] = {}
    for q in queries:
        for meta in _fetch_one_query(q, max_papers_per_query, categories):
            if meta.arxiv_id not in seen:
                seen[meta.arxiv_id] = meta
            else:
                # merge tags so we remember every query that surfaced this paper
                seen[meta.arxiv_id].tags = sorted(set(seen[meta.arxiv_id].tags) | set(meta.tags))
    return list(seen.values())


def download_pdfs(papers: list[PaperMeta], papers_dir: str | Path) -> list[PaperMeta]:
    """Download PDFs, writing to `{papers_dir}/{arxiv_id}.pdf`."""
    out_dir = Path(papers_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[PaperMeta] = []
    for meta in papers:
        pdf_path = out_dir / f"{_slugify(meta.arxiv_id)}.pdf"
        if pdf_path.exists():
            log.info("arxiv.download.skip_existing", arxiv_id=meta.arxiv_id)
            meta.pdf_path = str(pdf_path)
            downloaded.append(meta)
            continue
        try:
            _download_one(meta, pdf_path)
            meta.pdf_path = str(pdf_path)
            downloaded.append(meta)
        except Exception as exc:
            log.warning("arxiv.download.failed", arxiv_id=meta.arxiv_id, error=str(exc))
    return downloaded


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def _download_one(meta: PaperMeta, out_path: Path) -> None:
    import httpx

    with httpx.stream("GET", meta.pdf_url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    log.info("arxiv.download.ok", arxiv_id=meta.arxiv_id, size_kb=out_path.stat().st_size // 1024)


def save_metadata(papers: list[PaperMeta], out_path: str | Path) -> None:
    """Persist metadata JSONL — one paper per line."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for meta in papers:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
    log.info("arxiv.metadata.saved", rows=len(papers), path=str(path))


def load_metadata(path: str | Path) -> list[PaperMeta]:
    p = Path(path)
    if not p.exists():
        return []
    papers: list[PaperMeta] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(PaperMeta(**json.loads(line)))
    return papers
