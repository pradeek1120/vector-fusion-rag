import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Security, Request, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import get_settings
from src.logger import configure_logging, get_logger
from src.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    Source,
)
from src.ingestion import (
    ingest_documents,
    get_qdrant_client,
    get_dense_model,
)
from src.graph import run_query
from src.cache import get_cached, set_cached
from src.observability import (
    get_metrics,
    record_query,
    record_ingestion,
    trace_query,
)
from src.retriever import hybrid_retrieve
from src.reranker import rerank
from openai import OpenAI

# --- Setup ---

configure_logging()
settings = get_settings()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up models and connections
    logger.info("startup_begin", environment=settings.environment)
    get_dense_model()          # pre-load embedding model
    get_qdrant_client()        # test qdrant connection
    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Production Hybrid RAG API",
    description="Dense + Sparse + Reranking RAG system with observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth ---

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


# --- Request logging middleware ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - t0) * 1000)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# --- Health ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    qdrant_status = "ok"
    redis_status = "ok"

    try:
        get_qdrant_client().get_collections()
    except Exception:
        qdrant_status = "unavailable"

    from src.cache import get_redis
    try:
        r = get_redis()
        if r:
            r.ping()
        else:
            redis_status = "unavailable"
    except Exception:
        redis_status = "unavailable"

    return HealthResponse(
        status="ok" if qdrant_status == "ok" else "degraded",
        qdrant=qdrant_status,
        redis=redis_status,
        timestamp=datetime.utcnow().isoformat(),
    )


# --- Prometheus metrics endpoint ---

@app.get("/metrics", tags=["System"])
async def metrics():
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)


# --- Ingest ---

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit("10/minute")
async def ingest(
    request: Request,
    body: IngestRequest,
    _: str = Depends(verify_api_key),
):
    """
    Ingest text documents into the vector store.
    Handles chunking, dense + sparse embedding, and upsert into Qdrant.
    """
    try:
        result = ingest_documents(
            texts=body.documents,
            source_name=body.source_name,
            metadata=body.metadata,
        )
        record_ingestion(result["chunks_indexed"])
        return IngestResponse(**result)
    except Exception as e:
        logger.error("ingest_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit("5/minute")
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    source_name: str = "uploaded_file",
    _: str = Depends(verify_api_key),
):
    """
    Ingest a .txt or .pdf file directly.
    """
    content = await file.read()

    if file.filename.endswith(".pdf"):
        import fitz
        import io
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
    elif file.filename.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=400, detail="Only .txt and .pdf files are supported"
        )

    try:
        result = ingest_documents(
            texts=[text],
            source_name=source_name or file.filename,
        )
        record_ingestion(result["chunks_indexed"])
        return IngestResponse(**result)
    except Exception as e:
        logger.error("ingest_file_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Query ---

@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit(settings.rate_limit)
async def query(
    request: Request,
    body: QueryRequest,
    _: str = Depends(verify_api_key),
):
    """
    Run a hybrid RAG query.
    Returns the generated answer, sources, and timing metadata.
    """
    t0 = time.time()

    # Check cache first
    cached_result = get_cached(body.query, body.top_k, body.use_hyde)
    if cached_result:
        record_query(
            status="success",
            route="rag",
            total_ms=1.0,
            retrieval_ms=0,
            rerank_ms=0,
            top_score=0,
            cached=True,
        )
        cached_result["cached"] = True
        return QueryResponse(**cached_result)

    try:
        result = run_query(
            query=body.query,
            top_k=body.top_k,
            use_hyde=body.use_hyde,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        timings = result.get("timings", {})
        total_ms = round((time.time() - t0) * 1000, 2)
        sources = [Source(**s) for s in result.get("sources", [])]
        top_score = sources[0].rerank_score if sources else 0.0

        response_data = {
            "query": body.query,
            "answer": result["answer"],
            "sources": sources,
            "latency_ms": total_ms,
            "trace_id": result["trace_id"],
            "cached": False,
        }

        # Async observability (don't block response)
        record_query(
            status="success",
            route=result.get("route", "rag"),
            total_ms=total_ms,
            retrieval_ms=timings.get("retrieve_total_ms", 0),
            rerank_ms=timings.get("rerank_ms", 0),
            top_score=top_score,
            cached=False,
        )

        trace_query(
            trace_id=result["trace_id"],
            query=body.query,
            answer=result["answer"],
            sources=result.get("sources", []),
            timings=timings,
            cached=False,
        )

        # Cache the result
        cacheable = {
            "query": body.query,
            "answer": result["answer"],
            "sources": [s.model_dump() for s in sources],
            "latency_ms": total_ms,
            "trace_id": result["trace_id"],
        }
        set_cached(body.query, cacheable, body.top_k, body.use_hyde)

        return QueryResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_error", query=body.query[:60], error=str(e))
        record_query("error", "rag", 0, 0, 0, 0, False)
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Streaming Query ---

@app.post("/query/stream", tags=["Query"])
@limiter.limit(settings.rate_limit)
async def query_stream(
    request: Request,
    body: QueryRequest,
    _: str = Depends(verify_api_key),
):
    """
    Streaming variant of the RAG query.
    Sends tokens as they are generated.
    """

    def event_stream():
        t0 = time.time()
        client = OpenAI(api_key=settings.openai_api_key)

        # 1) Hybrid retrieve + rerank (same components as graph)
        docs, timings = hybrid_retrieve(
            query=body.query,
            top_k=body.top_k,
            use_hyde=body.use_hyde,
        )
        candidates = docs[: settings.top_k_retrieve]
        reranked, rerank_ms = rerank(body.query, candidates)

        # 2) Build context (mirrors src.graph.node_generate)
        if not reranked:
            yield "data: " + "I could not find relevant information to answer your question.\n\n"
            return

        context_parts = []
        for i, doc in enumerate(reranked):
            context_parts.append(
                f"[Source {i + 1}] ({doc.get('source', 'unknown')})\n{doc['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a precise, helpful assistant. Answer the user's question "
            "based strictly on the provided context passages.\n\n"
            "Rules:\n"
            "- Use only information from the context. Do not hallucinate.\n"
            "- If the context does not contain enough information, say: "
            "\"I don't have enough information in my knowledge base to answer that.\"\n"
            "- Cite specific parts of the context when possible.\n"
            "- Be concise and direct. Avoid filler phrases.\n"
            "- Format your response clearly using markdown if the answer is complex."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {body.query}",
            },
        ]

        # 3) Stream tokens from OpenAI
        try:
            stream = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    # SSE format: lines starting with "data: "
                    yield "data: " + delta.content + "\n\n"

            total_ms = round((time.time() - t0) * 1000, 2)
            record_query(
                status="success",
                route="rag",
                total_ms=total_ms,
                retrieval_ms=timings.get("dense_ms", 0) + timings.get("sparse_ms", 0),
                rerank_ms=rerank_ms,
                top_score=0.0,
                cached=False,
            )
        except Exception as e:
            logger.error("query_stream_error", query=body.query[:60], error=str(e))
            record_query("error", "rag", 0, 0, 0, 0, False)
            yield "data: " + "Internal server error\n\n"

    # Server-Sent Events (SSE) stream
    return StreamingResponse(event_stream(), media_type="text/event-stream")
