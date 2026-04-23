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


def _run_pipeline_for_ragas(limit: int | None = None) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Run every question in the golden set through the RAG pipeline.

    Returns 4 aligned lists in the shape RAGAS expects:
      questions, answers, contexts (list of str per question), ground_truths.
    """
    cfg = load_yaml_config()
    golden_path = cfg["evaluation"]["golden_set_path"]
    qs = load_golden(golden_path)
    if not qs:
        raise RuntimeError(f"Golden set at {golden_path} is empty.")
    if limit is not None:
        qs = qs[:limit]

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


def run(limit: int | None = None) -> None:
    cfg = load_yaml_config()
    env = get_env()

    # Lazy imports — ragas pulls in heavy deps
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    # Point RAGAS at the same provider we use for generation so the user
    # only has to set one API key. For the JUDGE LLM specifically we use a
    # smaller/faster variant of the model (more headroom on free-tier rate
    # limits), since judging is less demanding than answering.
    base_url = {
        "groq": "https://api.groq.com/openai/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }.get(env.llm_provider)

    # Judge model — 70B gives higher TPM headroom on Groq free tier (12K vs 6K)
    # which matters because faithfulness + context_recall send the full context.
    judge_model = {
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
        "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    }.get(env.llm_provider, env.llm_model_override)

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            api_key=env.active_api_key,
            base_url=base_url,
            temperature=0.0,
            max_retries=5,
            request_timeout=60,
        )
    )

    # Local HF embeddings — Groq has no embeddings endpoint, so we never
    # want the RAGAS default OpenAI embedder.
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=env.embed_model)
    )

    questions, answers, contexts, ground_truths = _run_pipeline_for_ragas(limit=limit)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    # Serial, generous timeouts — trades speed for free-tier rate-limit safety.
    rc = RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60)

    log.info("ragas.evaluate.start", n=len(questions), judge=judge_model)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=False,
        run_config=rc,
    )

    # Extract aggregate scores via pandas — robust across ragas minor versions.
    df = result.to_pandas()
    metric_names = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    agg: dict[str, float] = {}
    for name in metric_names:
        if name in df.columns:
            agg[name] = float(df[name].mean(skipna=True))
        else:
            agg[name] = float("nan")

    # Pretty print
    table = Table(title=f"RAGAS · generator={env.llm_provider}/{env.llm_model_override or 'default'} · judge={judge_model} · n={len(questions)}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Threshold")
    table.add_column("Pass?", justify="center")

    thresholds = cfg["evaluation"]["thresholds"]
    any_fail = False
    for name in metric_names:
        value = agg[name]
        thr = thresholds.get(name, 0.0)
        passed = (value == value) and value >= thr  # NaN-safe
        any_fail = any_fail or not passed
        table.add_row(
            name,
            f"{value:.3f}" if value == value else "nan",
            f"{thr:.2f}",
            "OK" if passed else "FAIL",
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
        "judge_model": judge_model,
        "n_questions": len(questions),
        "metrics": agg,
        "thresholds": thresholds,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\n[bold]Report saved:[/bold] {report_path}")

    if any_fail:
        console.print("[bold red]One or more metrics below threshold.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    configure_logging(level=get_env().log_level)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N questions (useful for rate-limited free tiers)")
    args = ap.parse_args()
    run(limit=args.limit)
