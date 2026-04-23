"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.ingestion.pdf_parser import Chunk
from src.retrieval.hybrid import RetrievedChunk


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(arxiv_id="2301.00001", page=1, chunk_index=0,
              text="Transformer models can beat LSTMs on long horizons for load forecasting"),
        Chunk(arxiv_id="2301.00001", page=1, chunk_index=1,
              text="We use hour-of-day and weekday as calendar features"),
        Chunk(arxiv_id="2302.00002", page=3, chunk_index=0,
              text="Temperature is the dominant weather feature for short-term forecasting"),
    ]


@pytest.fixture
def sample_retrieved() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            arxiv_id="2301.00001", page=1, chunk_index=0, score=0.91,
            text="Transformer models can beat LSTMs on long horizons",
            title="Transformers for Energy Forecasting", authors=["A. Smith"],
            pdf_url="https://arxiv.org/pdf/2301.00001",
        ),
        RetrievedChunk(
            arxiv_id="2302.00002", page=3, chunk_index=0, score=0.78,
            text="Temperature is the dominant weather driver",
            title="Weather Features in Load Forecasting", authors=["B. Jones"],
            pdf_url="https://arxiv.org/pdf/2302.00002",
        ),
    ]
