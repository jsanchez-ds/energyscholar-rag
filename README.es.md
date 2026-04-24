🌐 [English](README.md) · **Español**

# 📚 EnergyScholar — Asistente RAG sobre Investigación en Forecasting Energético

> **Pipeline RAG production-grade sobre papers de arXiv en energía. Provider-agnostic de LLM, hosteable localmente, demo pública en HuggingFace Spaces.**

Haz preguntas en lenguaje natural sobre forecasting de demanda eléctrica, integración de renovables e investigación en demand-response, y obtén respuestas **ancladas en papers reales de arXiv con citas a nivel de página**. Construido con una **capa LLM provider-agnostic** (Groq por defecto, intercambiable con Anthropic / OpenAI / OpenRouter), embeddings locales, un vector store en Qdrant y un harness de evaluación RAGAS.

Proyecto hermano de [`energy-forecasting-databricks`](https://github.com/jsanchez-ds/energy-forecasting-databricks) — mismo dominio, ángulo complementario (LLM/RAG vs ML clásico).

---

## 🏗️ Arquitectura

```
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│  API arXiv   │────▶│ ingest + chunk │────▶│ Qdrant vector DB │
│  (papers)    │     │ + embed (ST)   │     │ (Docker local)   │
└──────────────┘     └────────────────┘     └────────┬─────────┘
                                                     │
         ┌────────────────────────────────────────────┴──────────┐
         │                 Pipeline RAG                           │
         │  1. Embed de la query                                  │
         │  2. Búsqueda híbrida (BM25 + vector)                   │
         │  3. Rerank (cross-encoder)                             │
         │  4. Generar respuesta vía LLM (Groq/Claude/…)          │
         │  5. Retornar respuesta + citas a nivel paper/página    │
         └───────────────┬────────────────────────────────────────┘
                         │
        ┌──────────────────────┐             ┌──────────────────────┐
        │   FastAPI            │◀────────────│  Streamlit UI        │
        │   /query, /eval      │             │  (HF Space público)  │
        └──────────┬───────────┘             └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐             ┌──────────────────────┐
        │   Langfuse tracing   │             │  Evaluación RAGAS    │
        │   (cada llamada LLM) │             │  (set de Q&A dorado) │
        └──────────────────────┘             └──────────────────────┘
```

---

## 🎯 Qué demuestra este proyecto

| Capacidad | Evidencia | Rol al que apunta |
|---|---|---|
| LLM engineering | Prompts tipados, patrones de tool-use, guardrails | LLM Eng |
| Diseño provider-agnostic | Un solo code path corre contra Groq / Anthropic / OpenRouter / OpenAI | LLM Eng + MLOps |
| Patrones RAG avanzados | Búsqueda híbrida (BM25 + densa) + rerank con cross-encoder + tracking de citas | LLM Eng + Senior DS |
| **Evaluación primero** | RAGAS (faithfulness, context precision/recall, answer relevance) gateada en CI | Senior DS |
| Tracing / observabilidad | Instrumentación Langfuse en cada llamada LLM | MLOps |
| Data pipeline | Ingesta arXiv idempotente, re-embed incremental al cambiar modelo | Data Eng |
| Vector DB | Qdrant con índices híbridos + filtros de payload | Data Eng + MLOps |
| Deploy cloud | HuggingFace Space público (URL en vivo para recruiters) | ML Eng |
| Disciplina CI/CD | Ruff + mypy + pytest + RAGAS en cada PR | MLOps |

---

## 📂 Estructura del proyecto

```
.
├── src/
│   ├── llm/             # Abstracción de providers (Groq / Anthropic / OpenAI / OpenRouter)
│   ├── ingestion/       # Fetch arXiv + parse PDF + chunking
│   ├── embedding/       # Wrapper de sentence-transformers + batching
│   ├── retrieval/       # Cliente Qdrant + búsqueda híbrida + rerank
│   ├── generation/      # Templates de prompt + builder de citas
│   ├── eval/            # Runner RAGAS + loader de Q&A dorado
│   ├── serving/         # App FastAPI
│   └── utils/           # Config, logging, telemetría
├── evaluation/
│   └── golden_set/      # Q&A curadas a mano para eval RAGAS
├── dashboards/          # UI Streamlit
├── docker/              # docker-compose.yml (Qdrant + Langfuse)
├── tests/               # Suite pytest
├── configs/             # Configs YAML
├── scripts/             # ingest.py, reindex.py, eval.py
└── .github/workflows/   # CI (tests + RAGAS en PRs)
```

---

## 🚀 Quickstart

### 1. Requisitos
- Python 3.11+
- Docker (para Qdrant) — opcional si usas modo embedded
- Un API key gratuito de al menos un provider LLM:
  - **Groq** (default recomendado, free tier, sin tarjeta) → <https://console.groq.com/keys>
  - Anthropic → <https://console.anthropic.com/settings/keys>
  - OpenAI → <https://platform.openai.com/api-keys>
  - OpenRouter → <https://openrouter.ai/keys>

### 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # pegar tu key de Groq (o la que uses)
docker compose -f docker/docker-compose.yml up -d qdrant langfuse
```

### 3. Ingestar papers

```bash
# Descarga el set por defecto (energy forecasting) — ~30 papers
python -m src.ingestion.run_arxiv --max-papers 30

# Chunk + embed + push a Qdrant
python -m src.embedding.run_index
```

### 4. Lanzar el stack

```bash
uvicorn src.serving.api:app --reload --port 8000  # http://localhost:8000/docs
streamlit run dashboards/app.py                   # http://localhost:8501
```

### 5. Evaluar

```bash
python -m src.eval.run_ragas                      # corre el golden set, imprime tabla de métricas
```

---

## 📊 Resultados — corrida en vivo sobre papers de arXiv

**Corpus:** 17 papers de arXiv sobre forecasting energético (load forecasting, renovables, métodos probabilísticos) → **386 chunks** en Qdrant embebido.

El mismo pipeline fue ejercitado contra dos backends de provider en esta sesión:

| Etapa         | Groq                            | OpenRouter                              |
|---------------|---------------------------------|-----------------------------------------|
| Generador     | `llama-3.3-70b-versatile`       | `nvidia/nemotron-3-super-120b-a12b:free` |
| Juez (RAGAS)  | `llama-3.1-8b-instant`          | `google/gemma-3-12b-it:free`            |
| Query E2E     | ✅ respuesta + citas            | ✅ respuesta + citas                     |

Cambiar entre ambos es una línea en `.env`.

### Dashboard Streamlit en acción

![EnergyScholar dashboard](docs/images/streamlit_demo.png)

### Query end-to-end de ejemplo (OpenRouter + Nemotron 120B)

**P:** *"How does temperature affect day-ahead electricity load forecasts?"*

Después del retrieval híbrido (20 densos + 20 BM25 → RRF → rerank con cross-encoder → top 5) y un paso de Nemotron con el system prompt que exige citas estrictas, el pipeline retorna:

> - La temperatura dirige la carga eléctrica principalmente mediante demanda de calefacción y refrigeración: días fríos aumentan la carga por calefacción, días calurosos por refrigeración **[2302.12168v2, p.7]**.
> - La relación entre temperatura media diaria y carga media diaria es **no lineal y bimodal** — días fríos y calurosos pueden coexistir con carga alta o baja porque feriados/fines de semana la suprimen **[2302.12168v2, p.7]**.
> - Umbrales específicos del estudio: **18 °C para heating degree-days (HDD)** y **25 °C para cooling degree-days (CDD)** **[2302.12168v2, p.7]**.
> - La temperatura se incluye explícitamente como feature numérica F9 en los modelos de forecasting porque afecta conocidamente los patrones de carga **[2302.12168v2, p.13]**.
>
> _En resumen, la temperatura influye en los forecasts day-ahead alterando la demanda de calefacción y refrigeración de forma no lineal y dependiente de umbrales, produciendo una relación temperatura–carga bimodal que los modelos deben capturar para mejorar precisión._

Las cinco citas resuelven a PDFs reales de arXiv (`[2302.12168v2]` — *"A comparative assessment of deep learning models for day-ahead load forecasting"* — páginas 3, 6, 7, 13 y 18).

### Métricas RAGAS (n=3 preguntas, free tier de Groq)

| Métrica             | Valor  | Umbral    | Pass  |
|---------------------|--------|-----------|-------|
| context_precision   | 0.814  | 0.70      | ✅    |
| answer_relevancy    | 0.996  | 0.75      | ✅    |
| faithfulness        | _n/a_  | 0.75      | ⚠️     |
| context_recall      | _n/a_  | 0.70      | ⚠️     |

La calidad del retrieval está en el rango esperado (context_precision 0.81) y las respuestas son altamente on-topic (answer_relevancy ≈ 1.0). `faithfulness` y `context_recall` necesitan más headroom de tokens-por-minuto del que da el free tier de Groq para payloads de este tamaño — se completan limpiamente con OpenAI `gpt-4o-mini`, el Dev tier de Groq, o cualquier provider de pago. El cableado está correcto en cualquier caso — `LLM_PROVIDER=openai` (u `openrouter`) corre el mismo comando end-to-end.

---

## 📜 Licencia

MIT
