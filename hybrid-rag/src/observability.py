import time
import functools
from typing import Optional, Callable

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from config.settings import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# --- Prometheus metrics ---

REGISTRY = CollectorRegistry(auto_describe=True)

query_counter = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["status", "route"],
    registry=REGISTRY,
)

query_latency = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY,
)

retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval (dense+sparse+RRF) latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
    registry=REGISTRY,
)

rerank_latency = Histogram(
    "rag_rerank_latency_seconds",
    "Reranking latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
    registry=REGISTRY,
)

cache_hits = Counter(
    "rag_cache_hits_total",
    "Number of cache hits",
    registry=REGISTRY,
)

cache_misses = Counter(
    "rag_cache_misses_total",
    "Number of cache misses",
    registry=REGISTRY,
)

top_rerank_score = Histogram(
    "rag_top_rerank_score",
    "Top rerank score for each query (0-1 for Cohere, unbounded for local)",
    buckets=[-2, -1, 0, 1, 2, 5, 10],
    registry=REGISTRY,
)

chunks_indexed = Counter(
    "rag_chunks_indexed_total",
    "Total number of chunks ingested",
    registry=REGISTRY,
)


def get_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def record_query(
    status: str,
    route: str,
    total_ms: float,
    retrieval_ms: float,
    rerank_ms: float,
    top_score: float,
    cached: bool,
):
    query_counter.labels(status=status, route=route).inc()
    query_latency.observe(total_ms / 1000)
    retrieval_latency.observe(retrieval_ms / 1000)
    rerank_latency.observe(rerank_ms / 1000)
    top_rerank_score.observe(top_score)

    if cached:
        cache_hits.inc()
    else:
        cache_misses.inc()


def record_ingestion(chunk_count: int):
    chunks_indexed.inc(chunk_count)


# --- Langfuse tracing ---

_langfuse = None


def get_langfuse():
    global _langfuse
    if _langfuse is None and settings.langfuse_public_key:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("langfuse_connected")
        except Exception as e:
            logger.warning("langfuse_unavailable", error=str(e))
    return _langfuse


def trace_query(
    trace_id: str,
    query: str,
    answer: str,
    sources: list[dict],
    timings: dict,
    cached: bool,
):
    lf = get_langfuse()
    if not lf:
        return

    try:
        trace = lf.trace(
            id=trace_id,
            name="hybrid-rag-query",
            input={"query": query},
            output={"answer": answer},
            metadata={
                "cached": cached,
                "num_sources": len(sources),
                **timings,
            },
        )
        lf.flush()
    except Exception as e:
        logger.warning("langfuse_trace_error", error=str(e))
