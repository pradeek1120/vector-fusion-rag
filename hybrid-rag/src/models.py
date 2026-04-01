from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    id: str
    text: str
    source: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1)
    source_name: str = "manual"
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str
    collection: str
    duration_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_hyde: Optional[bool] = None
    filters: dict = Field(default_factory=dict)


class Source(BaseModel):
    text: str
    source: str
    rerank_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    latency_ms: float
    trace_id: str
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    redis: str
    timestamp: str


class EvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_questions: int
    timestamp: str
