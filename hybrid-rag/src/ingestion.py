import uuid
import re
import time
from typing import Optional
import fitz  # PyMuPDF

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from src.logger import get_logger
from src.models import Chunk

logger = get_logger(__name__)
settings = get_settings()

# --- Singletons (loaded once at startup) ---
_dense_model: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None
_vocab: dict = {}


def get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        logger.info("Loading dense embedding model...")
        _dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _dense_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _qdrant_client = QdrantClient(**kwargs)
    return _qdrant_client


def get_vocab() -> dict:
    return _vocab


# --- Chunking ---

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sentence-aware chunker. Splits on sentence boundaries,
    then merges into chunks with token overlap.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue

        proposed = (current + " " + sentence).strip()

        if len(proposed.split()) > chunk_size and current:
            chunks.append(current.strip())
            # Overlap: keep last N words from current chunk
            overlap_words = current.split()[-overlap:]
            current = " ".join(overlap_words) + " " + sentence
        else:
            current = proposed

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c.split()) >= 10]  # drop tiny chunks


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return "\n\n".join(pages)


# --- Sparse vector (BM25-style) ---

def build_vocab_from_corpus(texts: list[str]) -> dict:
    """Build a term → index vocabulary from the full corpus."""
    global _vocab
    all_tokens = set()
    for text in texts:
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        all_tokens.update(tokens)
    _vocab = {word: idx for idx, word in enumerate(sorted(all_tokens))}
    logger.info("vocab_built", size=len(_vocab))
    return _vocab


def text_to_sparse_vector(text: str, vocab: dict) -> SparseVector:
    """
    TF-weighted sparse vector. Indices = term IDs, values = term freq.
    Use SPLADE for learned sparse weights in production upgrade.
    """
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    freq: dict[int, float] = {}
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            freq[idx] = freq.get(idx, 0.0) + 1.0

    if not freq:
        return SparseVector(indices=[0], values=[0.0])

    return SparseVector(
        indices=list(freq.keys()),
        values=list(freq.values()),
    )


# --- Qdrant collection management ---

def ensure_collection(client: QdrantClient, collection: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("collection_created", name=collection)
    else:
        logger.info("collection_exists", name=collection)


# --- Main ingestion ---

def ingest_documents(
    texts: list[str],
    source_name: str = "manual",
    metadata: dict = {},
    collection: Optional[str] = None,
) -> dict:
    t0 = time.time()
    collection = collection or settings.qdrant_collection

    client = get_qdrant_client()
    model = get_dense_model()

    ensure_collection(client, collection)

    # 1. Chunk all documents
    all_chunks: list[Chunk] = []
    for doc_text in texts:
        chunks_text = chunk_text(
            doc_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for i, chunk in enumerate(chunks_text):
            all_chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk,
                    source=source_name,
                    chunk_index=i,
                    metadata=metadata,
                )
            )

    logger.info("chunks_created", count=len(all_chunks))

    # 2. Build vocab from all chunks (for sparse vectors)
    vocab = build_vocab_from_corpus([c.text for c in all_chunks])

    # 3. Dense embeddings in batches
    texts_to_embed = [c.text for c in all_chunks]
    dense_vectors = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 4. Build Qdrant points
    points = []
    for chunk, dvec in zip(all_chunks, dense_vectors):
        sparse_vec = text_to_sparse_vector(chunk.text, vocab)
        points.append(
            PointStruct(
                id=chunk.id,
                vector={
                    "dense": dvec.tolist(),
                    "sparse": sparse_vec,
                },
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                },
            )
        )

    # 5. Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)

    duration_ms = (time.time() - t0) * 1000
    logger.info(
        "ingestion_complete",
        chunks=len(points),
        duration_ms=round(duration_ms),
    )

    return {
        "chunks_indexed": len(points),
        "source": source_name,
        "collection": collection,
        "duration_ms": round(duration_ms, 2),
    }
