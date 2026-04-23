# Runbook

## First-time setup

```bash
cd "/c/Users/jona2/Nuevo proyecto 2"
bash scripts/bootstrap.sh
source .venv/Scripts/activate
# edit .env and paste your Groq (or other) API key
make qdrant-up       # starts Qdrant on localhost:6333
```

## Build the index (one-time, ~5–10 min on CPU)

```bash
make ingest          # arXiv → data/papers/*.pdf + metadata JSONL
make index           # chunk + embed + upsert into Qdrant
```

## Run the stack

```bash
make serve &         # http://localhost:8000/docs
make dashboard &     # http://localhost:8501
```

## Evaluate

```bash
make eval            # RAGAS on the golden set → evaluation/reports/*.json
```

If any metric is below its threshold (set in `configs/config.yaml`), the command exits non-zero — this is what CI uses to gate PRs labelled `run-ragas`.

## Common operations

| Goal | Command |
|---|---|
| Switch provider | Edit `LLM_PROVIDER` and the matching `*_API_KEY` in `.env` |
| Reindex after chunker change | `make index` (the upsert is idempotent via deterministic UUIDs) |
| Add more topics to ingest | Edit `configs/config.yaml → ingestion.arxiv_queries` |
| Wipe the vector store | `docker compose -f docker/docker-compose.yml down -v` |

## Troubleshooting

**Qdrant connection refused** → `docker ps` to confirm the container is up. `make qdrant-up` if not.

**`RuntimeError: Provider '...' selected but no API key set`** → fill the matching key in `.env`.

**RAGAS hangs on "Evaluating..."** → It's calling the LLM for judgement. On Groq's free tier you may be rate-limited; lower the golden set to ~5 items for smoke testing or switch to a paid provider for the CI run.
