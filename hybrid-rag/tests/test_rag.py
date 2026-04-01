"""
Test suite for the Hybrid RAG system.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# --- Unit tests: chunking ---

def test_chunk_text_basic():
    from src.ingestion import chunk_text
    text = "This is sentence one. This is sentence two. " * 50
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) >= 10


def test_chunk_text_short_doc():
    from src.ingestion import chunk_text
    text = "Short document with few words."
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    # Too short to meet 10-word minimum, returns empty
    assert isinstance(chunks, list)


def test_chunk_overlap():
    from src.ingestion import chunk_text
    text = ". ".join([f"Sentence number {i} has some extra words here" for i in range(100)])
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    # Overlap: last words of chunk N should appear in chunk N+1
    if len(chunks) >= 2:
        last_words_c0 = set(chunks[0].split()[-5:])
        first_words_c1 = set(chunks[1].split()[:15])
        assert len(last_words_c0 & first_words_c1) > 0


# --- Unit tests: RRF fusion ---

def test_rrf_fusion_basic():
    from src.retriever import reciprocal_rank_fusion

    dense = [
        {"id": "a", "text": "doc a", "source": "s1", "dense_score": 0.9},
        {"id": "b", "text": "doc b", "source": "s1", "dense_score": 0.8},
    ]
    sparse = [
        {"id": "b", "text": "doc b", "source": "s1", "sparse_score": 1.0},
        {"id": "c", "text": "doc c", "source": "s1", "sparse_score": 0.7},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=0.6, sparse_weight=0.4)

    assert len(fused) == 3
    ids = [d["id"] for d in fused]
    # "b" appears in both lists, should score highest
    assert ids[0] == "b"
    for doc in fused:
        assert "rrf_score" in doc
        assert doc["rrf_score"] > 0


def test_rrf_empty_sparse():
    from src.retriever import reciprocal_rank_fusion
    dense = [{"id": "a", "text": "doc", "source": "s", "dense_score": 0.9}]
    fused = reciprocal_rank_fusion(dense, [], k=60, dense_weight=0.6, sparse_weight=0.4)
    assert len(fused) == 1
    assert fused[0]["id"] == "a"


def test_rrf_score_ordering():
    from src.retriever import reciprocal_rank_fusion
    dense = [{"id": str(i), "text": f"doc {i}", "source": "s", "dense_score": 0.9 - i * 0.1} for i in range(5)]
    sparse = [{"id": str(i), "text": f"doc {i}", "source": "s", "sparse_score": 0.9 - i * 0.1} for i in range(5)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=0.6, sparse_weight=0.4)
    scores = [d["rrf_score"] for d in fused]
    assert scores == sorted(scores, reverse=True)


# --- Unit tests: sparse vector ---

def test_sparse_vector_basic():
    from src.ingestion import text_to_sparse_vector, build_vocab_from_corpus
    corpus = ["the quick brown fox", "jumped over the lazy dog"]
    vocab = build_vocab_from_corpus(corpus)
    vec = text_to_sparse_vector("the quick fox", vocab)
    assert len(vec.indices) > 0
    assert len(vec.indices) == len(vec.values)
    assert all(v > 0 for v in vec.values)


def test_sparse_vector_oov():
    from src.ingestion import text_to_sparse_vector
    vec = text_to_sparse_vector("xyzzyx qwerty", vocab={})
    assert vec.indices == [0]


# --- Unit tests: cache ---

def test_cache_key_deterministic():
    from src.cache import _cache_key
    k1 = _cache_key("test query", 5, True)
    k2 = _cache_key("test query", 5, True)
    assert k1 == k2


def test_cache_key_different_params():
    from src.cache import _cache_key
    k1 = _cache_key("test query", 5, True)
    k2 = _cache_key("test query", 10, True)
    assert k1 != k2


# --- Integration tests: API ---

@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant" in data
    assert "redis" in data


def test_query_requires_auth(client):
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 403


def test_query_invalid_key(client):
    response = client.post(
        "/query",
        json={"query": "test"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_query_too_short(client):
    from config.settings import get_settings
    settings = get_settings()
    response = client.post(
        "/query",
        json={"query": "hi"},
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    assert response.status_code == 422


@patch("api.main.run_query")
@patch("api.main.get_cached", return_value=None)
def test_query_success(mock_cache, mock_run, client):
    from config.settings import get_settings
    settings = get_settings()

    mock_run.return_value = {
        "answer": "Test answer",
        "sources": [{"text": "context", "source": "doc1", "rerank_score": 0.9}],
        "trace_id": "test-trace-id",
        "route": "rag",
        "timings": {"total_ms": 500, "retrieve_total_ms": 100, "rerank_ms": 50},
        "error": None,
    }

    response = client.post(
        "/query",
        json={"query": "What is hybrid search?"},
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert "sources" in data
    assert "latency_ms" in data
    assert data["cached"] is False


@patch("api.main.ingest_documents")
def test_ingest_success(mock_ingest, client):
    from config.settings import get_settings
    settings = get_settings()

    mock_ingest.return_value = {
        "chunks_indexed": 10,
        "source": "test",
        "collection": "hybrid_rag",
        "duration_ms": 123.4,
    }

    response = client.post(
        "/ingest",
        json={"documents": ["Test document content here."], "source_name": "test"},
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 10
