"""Streamlit UI for EnergyScholar."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="EnergyScholar", page_icon="📚", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 EnergyScholar")
    st.caption("RAG over arXiv energy-forecasting papers")

    try:
        info = httpx.get(f"{API_URL}/info", timeout=3).json()
        st.success("API ✓")
        st.json(info)
    except Exception as exc:
        st.error(f"API not reachable: {exc}")

    st.markdown("---")
    st.markdown(
        "**Try asking:**\n\n"
        "- What features matter most for short-term load forecasting?\n"
        "- How do transformer models compare to LSTMs for energy demand?\n"
        "- What is the typical MAPE range for day-ahead forecasts?"
    )

# ── Main ────────────────────────────────────────────────────────────────────
st.title("Ask the energy-forecasting literature")

question = st.text_area(
    "Your question",
    placeholder="e.g. How do transformers compare to LSTMs for day-ahead load forecasting?",
    height=100,
)

col1, col2 = st.columns([1, 6])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    show_retrieval = st.checkbox("Show retrieved passages", value=False)

if submit and question.strip():
    with st.spinner("Searching papers + generating answer..."):
        try:
            resp = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=120).json()
        except Exception as exc:
            st.error(f"Call failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(resp["answer"])

    st.subheader("Citations")
    for c in resp["citations"]:
        st.markdown(
            f"- **[{c['arxiv_id']}] p.{c['page']}** — "
            f"[{c['title']}]({c['pdf_url']})"
        )

    cols = st.columns(3)
    cols[0].metric("Provider", resp["provider"])
    cols[1].metric("Model", resp["model"])
    cols[2].metric("Context chunks", resp["n_context_chunks"])

    if show_retrieval:
        with st.spinner("Fetching raw retrieval..."):
            raw = httpx.post(
                f"{API_URL}/retrieve",
                json={"question": question, "top_k_final": 8},
                timeout=120,
            ).json()
        st.subheader("Raw retrieval (top-8 after rerank)")
        for hit in raw:
            with st.expander(f"{hit['arxiv_id']} p.{hit['page']} — {hit['title']}  ·  score {hit['score']:.3f}"):
                st.text(hit["text_preview"])

st.caption(
    "Built with sentence-transformers · Qdrant · RAGAS · Langfuse · FastAPI · Streamlit. "
    "LLM provider swappable via env var."
)
