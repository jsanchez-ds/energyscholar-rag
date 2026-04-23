"""
Entry-point: `python -m src.ingestion.run_arxiv [--max-papers N]`

  1. Runs the configured arXiv queries
  2. Downloads missing PDFs into `data/papers/`
  3. Writes the aggregated metadata as JSONL to `data/cache/papers.jsonl`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.arxiv_client import (
    download_pdfs,
    save_metadata,
    search_papers,
)
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch + download arXiv energy papers")
    ap.add_argument("--max-papers", type=int, default=None, help="Override per-query cap")
    return ap.parse_args()


def run(max_papers_per_query: int | None = None) -> None:
    env = get_env()
    cfg = load_yaml_config()

    queries: list[str] = cfg["ingestion"]["arxiv_queries"]
    per_q = max_papers_per_query or cfg["ingestion"]["max_papers_per_query"]
    categories: list[str] | None = cfg["ingestion"].get("categories")

    log.info("arxiv.run.start", n_queries=len(queries), per_query=per_q, categories=categories)

    papers = search_papers(queries, per_q, categories)
    log.info("arxiv.run.unique_papers", n=len(papers))

    papers = download_pdfs(papers, env.papers_dir)

    meta_path = Path(env.cache_dir) / "papers.jsonl"
    save_metadata(papers, meta_path)
    log.info("arxiv.run.done", downloaded=len(papers), metadata=str(meta_path))


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    args = parse_args()
    run(max_papers_per_query=args.max_papers)
