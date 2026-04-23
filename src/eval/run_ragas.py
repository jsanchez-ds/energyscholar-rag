"""
Entry-point: `python -m src.eval.run_ragas`

Runs the RAG pipeline against the golden Q&A set, computes RAGAS metrics,
and prints a rich table. Saves a JSON report under `evaluation/reports/`.
Fails the process (exit 1) if any metric is below its threshold — used by CI.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.eval.golden import load_golden
from src.generation.rag import answer
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)
console = Console()


def _run_pipeline_for_ragas() -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Run every question in the golden set through the RAG pipeline.

    Returns 4 aligned lists in the shape RAGAS expects:
      questions, answers, contexts (list of str per question), ground_truths.
    """
    cfg = load_yaml_config()
    golden_path = cfg["evaluation"]["golden_set_path"]
    qs = load_golden(golden_path)
    if not qs:
        raise RuntimeError(f"Golden set at {golden_path} is empty.")

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    from src.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()

    for qa in qs:
        chunks = retriever.search(
            qa.question,
            top_k_vector=cfg["retrieval"]["top_k_vector"],
            top_k_bm25=cfg["retrieval"]["top_k_bm25"],
            top_k_final=cfg["retrieval"]["top_k_final"],
            rrf_k=cfg["retrieval"]["rrf_k"],
            rerank=True,
        )
        # call the full RAG for the final answer (reuses retrieval internally —
        # small duplication for a cleaner separation of concerns)
        rag = answer(qa.question, retriever=retriever)

        questions.append(qa.question)
        answers.append(rag.answer)
        contexts.append([c.text for c in chunks])
        ground_truths.append(qa.reference_answer)
        log.info("eval.qa.done", q=qa.question[:80], chunks=len(chunks))

    return questions, answers, contexts, ground_truths


def run() -> None:
    cfg = load_yaml_config()
    env = get_env()

    # Lazy imports — ragas pulls in heavy deps
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    questions, answers, contexts, ground_truths = _run_pipeline_for_ragas()

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    log.info("ragas.evaluate.start", n=len(questions))
    # RAGAS uses its own LLM for judgement — point it at the same provider
    # we use for generation so you don't need a separate eval key.
    result = evaluate(ds, metrics=metrics)

    # Pretty print
    table = Table(title=f"RAGAS · {env.llm_provider}/{env.llm_model_override or 'default'} · n={len(questions)}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Threshold")
    table.add_column("Pass?", justify="center")

    thresholds = cfg["evaluation"]["thresholds"]
    any_fail = False
    for name in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        value = float(result[name]) if name in result else float("nan")
        thr = thresholds.get(name, 0.0)
        passed = value >= thr
        any_fail = any_fail or not passed
        table.add_row(
            name,
            f"{value:.3f}",
            f"{thr:.2f}",
            "✅" if passed else "❌",
        )

    console.print(table)

    # Persist JSON report
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evaluation/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"ragas_{ts}.json"
    payload = {
        "timestamp_utc": ts,
        "provider": env.llm_provider,
        "n_questions": len(questions),
        "metrics": {k: float(result[k]) for k in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"] if k in result},
        "thresholds": thresholds,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\n[bold]Report saved:[/bold] {report_path}")

    if any_fail:
        console.print("[bold red]One or more metrics below threshold.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    run()
