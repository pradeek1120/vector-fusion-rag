from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Reranker
    cohere_api_key: str = ""
    use_cohere_rerank: bool = False

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_rag"
    qdrant_api_key: str = ""

    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # RAG tuning
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_n_rerank: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    hyde_enabled: bool = True

    # API
    api_key: str = "dev-secret"
    rate_limit: str = "60/minute"
    log_level: str = "INFO"
    environment: Literal["development", "production"] = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
