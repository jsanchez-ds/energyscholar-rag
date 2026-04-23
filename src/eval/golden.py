"""Golden Q&A dataset loader (JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenQA:
    question: str
    reference_answer: str
    # Optional gold citations — arxiv_ids that the question is really about.
    # Used for context_recall if the retriever returns those papers.
    expected_arxiv_ids: list[str]


def load_golden(path: str | Path) -> list[GoldenQA]:
    p = Path(path)
    out: list[GoldenQA] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            out.append(
                GoldenQA(
                    question=rec["question"],
                    reference_answer=rec["reference_answer"],
                    expected_arxiv_ids=rec.get("expected_arxiv_ids", []),
                )
            )
    return out
