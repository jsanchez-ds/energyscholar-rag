"""Unit tests for the chunker."""

from src.ingestion.pdf_parser import chunk_text


def test_chunk_text_basic() -> None:
    text = " ".join([f"word{i}" for i in range(2000)])
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # Each chunk must respect the size cap
    for c in chunks:
        assert len(c.split()) <= 500


def test_chunk_overlap_preserved() -> None:
    text = " ".join([f"tok{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
    # With step=20 and len=100, expect ~5 chunks (0-30, 20-50, 40-70, 60-90, 80-100)
    assert 4 <= len(chunks) <= 6
    # Adjacent chunks must share 10 tokens
    first_tail = chunks[0].split()[-10:]
    second_head = chunks[1].split()[:10]
    assert first_tail == second_head


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []
