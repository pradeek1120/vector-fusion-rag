import time
from typing import Optional

import cohere
from sentence_transformers import CrossEncoder

from config.settings import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# --- Singletons ---
_local_reranker: Optional[CrossEncoder] = None
_cohere_client: Optional[cohere.Client] = None


def get_local_reranker() -> CrossEncoder:
    global _local_reranker
    if _local_reranker is None:
        logger.info("Loading cross-encoder reranker...")
        _local_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _local_reranker


def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


# --- Local cross-encoder reranking ---

def rerank_local(
    query: str,
    candidates: list[dict],
    top_n: int,
) -> list[dict]:
    """
    Score each (query, doc) pair with the cross-encoder.
    MiniLM-L-6 runs ~50ms for 20 docs on CPU.
    """
    reranker = get_local_reranker()
    pairs = [(query, c["text"]) for c in candidates]

    scores = reranker.predict(pairs, show_progress_bar=False)

    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# --- Cohere Rerank API ---

def rerank_cohere(
    query: str,
    candidates: list[dict],
    top_n: int,
) -> list[dict]:
    """
    Cohere's rerank-english-v3.0 model.
    Prefer in production for best quality (~100ms latency).
    """
    co = get_cohere_client()
    docs = [c["text"] for c in candidates]

    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_n,
        )
        reranked = []
        for r in response.results:
            doc = candidates[r.index].copy()
            doc["rerank_score"] = float(r.relevance_score)
            reranked.append(doc)
        return reranked

    except Exception as e:
        logger.warning("cohere_rerank_failed", error=str(e), fallback="local")
        return rerank_local(query, candidates, top_n)


# --- Public interface ---

def rerank(
    query: str,
    candidates: list[dict],
    top_n: Optional[int] = None,
) -> tuple[list[dict], float]:
    """
    Choose reranker based on config. Returns (reranked_docs, latency_ms).
    Falls back to local reranker if Cohere is unavailable.
    """
    top_n = top_n or settings.top_n_rerank
    t0 = time.time()

    if settings.use_cohere_rerank and settings.cohere_api_key:
        result = rerank_cohere(query, candidates, top_n)
        strategy = "cohere"
    else:
        result = rerank_local(query, candidates, top_n)
        strategy = "local_cross_encoder"

    latency_ms = round((time.time() - t0) * 1000, 2)

    logger.info(
        "rerank_done",
        strategy=strategy,
        input_count=len(candidates),
        output_count=len(result),
        latency_ms=latency_ms,
    )

    return result, latency_ms
