## Production Hybrid RAG System

This project is a **production‑ready Retrieval‑Augmented Generation (RAG) system** that I built end‑to‑end.  
It combines dense vector search, sparse BM25 search, reciprocal rank fusion, and cross‑encoder reranking, exposed
through a FastAPI service with caching, monitoring, and evaluation tooling.

You can link this repository directly from a resume or portfolio as:

> “Production‑grade hybrid RAG search system (dense + sparse + reranking) with FastAPI, Qdrant, Redis, and OpenAI.”

---

## Features

- **Hybrid search pipeline**
  - Dense semantic search with `BAAI/bge-base-en-v1.5` embeddings.
  - Sparse BM25‑style search with TF‑weighted sparse vectors.
  - Reciprocal Rank Fusion (RRF) to combine dense + sparse results.
  - Cross‑encoder reranking (local model or Cohere) to select top answers.
- **RAG generation**
  - Uses OpenAI Chat Completions (GPT‑4o by default) to generate grounded answers.
  - Returns final answer plus ranked source snippets and timing metadata.
- **Production API**
  - FastAPI service with API‑key auth, rate limiting, and CORS.
  - Typed request/response models using Pydantic.
  - Health, metrics, and streaming response endpoints.
- **Infrastructure & observability**
  - Qdrant as vector database (dense + sparse collections).
  - Redis for query result caching.
  - Prometheus metrics and Langfuse traces for monitoring.
  - Dockerized for local dev and cloud deployment (e.g. Cloud Run).
- **Evaluation**
  - Evaluation script to score RAG performance (faithfulness, relevancy, context quality).
  - Config‑driven experiments via `.env` (chunk size, HyDE, weights, etc.).

---

## Architecture (high level)

1. User sends a query to the FastAPI `/query` endpoint.
2. LangGraph‑based pipeline routes the query to the RAG flow.
3. Optional HyDE expansion generates a hypothetical answer to improve dense retrieval.
4. Dense and sparse searches are executed against Qdrant, then fused via RRF.
5. A cross‑encoder reranks the fused candidates to get the top‑N passages.
6. An OpenAI model (e.g. GPT‑4o) generates an answer grounded in those passages.
7. The service returns the answer, sources, latency breakdown, and a trace ID.

---

## Tech Stack

- **Backend / API:** FastAPI, Uvicorn  
- **Orchestration:** LangGraph  
- **Vector DB:** Qdrant (dense + sparse vectors)  
- **Embeddings:** SentenceTransformers – `BAAI/bge-base-en-v1.5`  
- **Reranking:** CrossEncoder `ms-marco-MiniLM-L-6-v2` and optional Cohere rerank  
- **LLM:** OpenAI Chat Completions (GPT‑4o by default, configurable)  
- **Cache:** Redis  
- **Config:** Pydantic Settings + `.env`  
- **Observability:** Prometheus metrics, Langfuse tracing  
- **Packaging:** Docker, `docker-compose`  

---

## Project Structure

```text
hybrid-rag/
├── api/
│   └── main.py              # FastAPI app (routes, auth, rate limiting)
├── src/
│   ├── ingestion.py         # Chunking, embeddings, Qdrant upsert
│   ├── retriever.py         # Dense + sparse retrieval, RRF fusion, HyDE
│   ├── reranker.py          # Cross-encoder / Cohere reranking
│   ├── graph.py             # LangGraph RAG pipeline and routing
│   ├── cache.py             # Redis caching layer
│   ├── observability.py     # Prometheus metrics + Langfuse tracing
│   ├── evaluate.py          # RAG evaluation utilities
│   ├── models.py            # Pydantic data models
│   └── logger.py            # Structured logging
├── config/
│   └── settings.py          # Central config loaded from .env
├── scripts/
│   └── cli.py               # (Optional) CLI helpers for ingest/query/eval
├── monitoring/              # Prometheus / Grafana configs
├── docker/
│   └── Dockerfile           # Production image (used for Cloud Run, etc.)
├── tests/                   # Unit / integration tests
├── docker-compose.yml       # Local dev stack (API + Qdrant + Redis, etc.)
├── requirements.txt
└── .env.example
```

---

## Local Development

### 1. Clone and configure

```bash
git clone https://github.com/pradeek1120/vector-fusion-rag.git
cd vector-fusion-rag

cp .env.example .env
# Edit .env and set at minimum:
#   OPENAI_API_KEY
#   QDRANT_URL (or use docker-compose default)
#   REDIS_URL
#   API_KEY (for protecting the API)
```

### 2. Run with Docker Compose

```bash
docker compose up -d

# API:       http://localhost:8000
# Qdrant:    http://localhost:6333
# Prometheus/Grafana: as configured in monitoring/
```

### 3. Ingest documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "documents": ["Your document text here..."],
        "source_name": "local_docs"
      }'
```

### 4. Ask questions

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What does this knowledge base contain?",
        "top_k": 5,
        "use_hyde": true
      }'
```

The response includes:

- `answer` – final LLM response grounded in retrieved documents.  
- `sources` – list of source snippets with rerank scores.  
- `latency_ms` and `trace_id` – useful for debugging and monitoring.  

---

## Deployment (example: Google Cloud Run)

The repository includes a production Dockerfile under `docker/Dockerfile`.  
Typical steps to deploy to Cloud Run:

1. Build and push the image to Artifact Registry:

   ```bash
   gcloud builds submit \
     --tag REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/hybrid-rag-api \
     --file docker/Dockerfile
   ```

2. Deploy the API:

   ```bash
   gcloud run deploy hybrid-rag-api \
     --image REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/hybrid-rag-api \
     --region REGION \
     --platform managed \
     --allow-unauthenticated \
     --port 8000 \
     --set-env-vars OPENAI_API_KEY=... \
     --set-env-vars QDRANT_URL=...,REDIS_URL=...,API_KEY=...,ENVIRONMENT=production
   ```

3. Point your front‑end or tools at the Cloud Run URL (e.g. `https://hybrid-rag-api-xxxx.run.app/query`).

---

## What I Implemented

This project demonstrates my ability to:

- Design and implement a full hybrid RAG pipeline (dense + sparse + RRF + reranking).
- Build a production‑style FastAPI service with authentication, rate limiting, and health checks.
- Integrate external services (OpenAI, Qdrant, Redis, Langfuse, Prometheus).
- Containerize and deploy the system using Docker and Cloud Run.
- Evaluate and iterate on RAG quality using configurable experiments.

Feel free to clone the repo, run it locally, and use it as a reference implementation of a modern production RAG system.

**Why hybrid over dense-only?** Dense search fails on exact keyword queries (product codes, proper nouns). BM25 handles these. Hybrid gets both.

**Why RRF over score normalization?** BM25 and cosine scores are on incompatible scales. RRF uses only rank position — completely scale-invariant.

**Why rerank after retrieval?** Bi-encoders embed query and doc independently (fast). Cross-encoders see them together (slow but accurate). Retrieve-20/rerank-5 gives you speed + quality.

**Why LangGraph over a simple function?** Stateful graph enables conditional routing, easy addition of new retrieval strategies (SQL, web), and visibility into each step for debugging.
