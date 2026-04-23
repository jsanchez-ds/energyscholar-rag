# Architecture

## Why provider-agnostic?

Tying a RAG system to a single LLM provider creates switching cost you will regret. The moment you want to A/B a cheaper model, run offline eval against a stronger one, or dodge a rate-limit, a hard-coded `anthropic.messages.create(...)` call is a refactor.

EnergyScholar's `LLMClient` exposes one `.chat(messages)` method and routes to Groq / OpenAI / OpenRouter via a shared OpenAI-compatible interface, or to Anthropic via its native SDK. Switching providers is a `.env` change.

## Why hybrid retrieval?

Dense (cosine) retrieval handles paraphrases and semantic similarity; BM25 nails rare domain terms and exact matches that embeddings often smooth over. We fuse them via **Reciprocal Rank Fusion** (RRF), a parameter-light way to combine ranked lists without having to calibrate scores across different scales.

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then re-ranks the top ~40 candidates. Cross-encoders are expensive but accurate — a strong hybrid-retrieve-then-rerank pattern typically beats a larger embedding model alone.

## Why a small golden set + RAGAS?

Eyeballing RAG outputs is how you ship regressions. RAGAS gives four metrics that cover the important failure modes:

| Metric | Catches… |
|---|---|
| **Faithfulness** | Hallucinations — answer says things the context doesn't support |
| **Context precision** | Low-quality retrieval stuffing the prompt with irrelevant chunks |
| **Context recall** | Missing the one passage that actually contains the answer |
| **Answer relevance** | Drift — answering a different question than was asked |

20 hand-curated Q&As is more useful than 500 generated ones. CI gates on thresholds declared in `configs/config.yaml`.

## Why Qdrant?

- Rust core, fast on CPU
- Runs locally via one Docker command
- Free managed tier for prod (Qdrant Cloud) with the same API
- Supports payload filters (useful when we scale to per-topic filtering)

## Why sentence-transformers locally?

Embedding calls at ingest scale (thousands of chunks) are the #1 hidden cost of RAG when you use API-provided embeddings. `all-MiniLM-L6-v2` runs fine on CPU, gives a 384-dim vector, and is free forever. Swap to a cloud embedder only if eval shows it matters.

## What's intentionally NOT here

- **LangChain / LlamaIndex**: adds abstractions that obscure the pipeline. Everything here is a few hundred lines of direct code.
- **Agent frameworks**: out of scope for v1. Would be added as a thin tool-use layer on top of `LLMClient` in v2.
- **Fine-tuning**: RAG answers most grounded-factual questions cheaper than fine-tuning does.
