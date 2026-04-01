import time
from collections import defaultdict
from typing import Optional

from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
)
from openai import OpenAI

from config.settings import get_settings
from src.ingestion import (
    get_dense_model,
    get_qdrant_client,
    get_vocab,
    text_to_sparse_vector,
)
from src.logger import get_logger
from src.models import RetrievedDoc

logger = get_logger(__name__)
settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)


# --- HyDE (Hypothetical Document Embeddings) ---

def generate_hyde_doc(query: str) -> str:
    """
    Ask the LLM to write a hypothetical answer.
    We embed this instead of the raw query for better dense recall.
    """
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Write a short, factual, "
                        "2-3 sentence passage that directly answers the user's question. "
                        "Do not say you don't know. Make your best guess."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("hyde_failed", error=str(e), fallback="raw_query")
        return query


# --- Dense search ---

def dense_search(
    query: str,
    collection: str,
    top_k: int,
    use_hyde: bool,
) -> list[dict]:
    client = get_qdrant_client()
    model = get_dense_model()

    text_to_embed = generate_hyde_doc(query) if use_hyde else query
    query_vec = model.encode(text_to_embed, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=collection,
        query_vector=NamedVector(name="dense", vector=query_vec),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id": str(r.id),
            "text": r.payload["text"],
            "source": r.payload.get("source", ""),
            "dense_score": float(r.score),
        }
        for r in results
    ]


# --- Sparse (BM25) search ---

def sparse_search(
    query: str,
    collection: str,
    top_k: int,
) -> list[dict]:
    client = get_qdrant_client()
    vocab = get_vocab()

    if not vocab:
        logger.warning("vocab_empty", msg="Sparse search skipped — no vocab built yet")
        return []

    sparse_vec = text_to_sparse_vector(query, vocab)

    results = client.search(
        collection_name=collection,
        query_vector=NamedSparseVector(name="sparse", vector=sparse_vec),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id": str(r.id),
            "text": r.payload["text"],
            "source": r.payload.get("source", ""),
            "sparse_score": float(r.score),
        }
        for r in results
    ]


# --- RRF Fusion ---

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[dict]:
    """
    RRF formula: score(d) = Σ weight / (k + rank(d))
    k=60 is the standard constant from the original paper.
    """
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        scores[doc_id] += dense_weight / (k + rank + 1)
        payloads[doc_id] = doc

    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        scores[doc_id] += sparse_weight / (k + rank + 1)
        if doc_id not in payloads:
            payloads[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for doc_id, rrf_score in ranked:
        doc = payloads[doc_id].copy()
        doc["rrf_score"] = round(rrf_score, 6)
        doc.setdefault("dense_score", 0.0)
        doc.setdefault("sparse_score", 0.0)
        fused.append(doc)

    return fused


# --- Main retriever ---

def hybrid_retrieve(
    query: str,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    use_hyde: Optional[bool] = None,
) -> tuple[list[dict], dict]:
    """
    Full hybrid retrieval pipeline.
    Returns (retrieved_docs, timing_info).
    """
    collection = collection or settings.qdrant_collection
    top_k = top_k or settings.top_k_retrieve
    use_hyde = use_hyde if use_hyde is not None else settings.hyde_enabled

    timings = {}

    # Dense search
    t0 = time.time()
    dense_res = dense_search(query, collection, top_k, use_hyde)
    timings["dense_ms"] = round((time.time() - t0) * 1000)

    # Sparse search
    t1 = time.time()
    sparse_res = sparse_search(query, collection, top_k)
    timings["sparse_ms"] = round((time.time() - t1) * 1000)

    # RRF Fusion
    fused = reciprocal_rank_fusion(
        dense_res,
        sparse_res,
        k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
    )

    logger.info(
        "hybrid_retrieve_done",
        query_preview=query[:60],
        dense_count=len(dense_res),
        sparse_count=len(sparse_res),
        fused_count=len(fused),
        **timings,
    )

    return fused, timings
