# ═══════════════════════════════════════════════════════════════════════════
# EnergyScholar — developer Makefile
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: help install lint format test test-cov clean \
        qdrant-up qdrant-down ingest index serve dashboard eval docker-up docker-down

help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install deps into active venv (torch CPU first, then the rest)
	pip install --upgrade pip
	pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
	pip install -r requirements.txt

lint:  ## Ruff + black (check)
	ruff check src tests
	black --check src tests

format:  ## Black + ruff --fix
	black src tests
	ruff check --fix src tests

test:  ## Unit tests (no integration / ragas)
	pytest tests/ -v -m "not integration and not ragas"

test-cov:  ## Unit tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── Pipeline ──────────────────────────────────────────────────────────────
qdrant-up:  ## Start Qdrant (+ Langfuse) via docker compose
	docker compose -f docker/docker-compose.yml up -d qdrant

qdrant-down:
	docker compose -f docker/docker-compose.yml down

ingest:  ## Fetch + download arXiv papers
	python -m src.ingestion.run_arxiv

index:  ## Chunk + embed + upsert into Qdrant
	python -m src.embedding.run_index

serve:  ## FastAPI on :8000
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Streamlit UI on :8501
	streamlit run dashboards/app.py

eval:  ## Run RAGAS on the golden set (requires index built)
	python -m src.eval.run_ragas

pipeline:  ## ingest → index → eval
	$(MAKE) ingest && $(MAKE) index && $(MAKE) eval
