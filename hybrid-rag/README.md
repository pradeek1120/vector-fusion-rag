# Production Hybrid RAG System

A production-grade Retrieval-Augmented Generation (RAG) system combining **dense vector search**, **sparse BM25 search**, **Reciprocal Rank Fusion**, and **cross-encoder reranking** — the architecture used at companies like Cohere, Vespa, and Elastic.

---

## Architecture

```
User Query
    │
    ▼
Query Router (LangGraph)
    │ route = "rag"
    ▼
HyDE Expansion ──────────────────────────┐
    │                                     │
    ▼                                     ▼
Dense Search (Qdrant)         Sparse BM25 Search (Qdrant)
BAAI/bge-base-en-v1.5         TF-weighted sparse vectors
top-20 results                top-20 results
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
        RRF Score Fusion (k=60)
        weighted merge by rank
                   │
                   ▼
        Cross-encoder Reranker
        ms-marco-MiniLM or Cohere
        top-20 → top-5
                   │
                   ▼
        LLM Generation (GPT-4o)
        grounded answer + sources
                   │
                   ▼
        FastAPI Response
        + Redis cache
        + Prometheus metrics
        + Langfuse trace
```

---

## Stack

| Component | Technology |
|---|---|
| Vector DB | Qdrant (dense + sparse native) |
| Dense embeddings | BAAI/bge-base-en-v1.5 (free, SOTA) |
| Sparse search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | ms-marco-MiniLM-L-6-v2 / Cohere |
| LLM | GPT-4o (configurable) |
| Orchestration | LangGraph |
| API | FastAPI + uvicorn |
| Cache | Redis |
| Observability | Prometheus + Grafana + Langfuse |
| CI/CD | GitHub Actions |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/youruser/hybrid-rag
cd hybrid-rag
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and API_KEY at minimum
```

### 2. Start the stack

```bash
docker compose up -d
# Qdrant:    http://localhost:6333
# API:       http://localhost:8000
# Grafana:   http://localhost:3000  (admin/admin)
# Prometheus:http://localhost:9090
```

### 3. Ingest documents

```bash
# Ingest a PDF
python scripts/cli.py ingest --file path/to/your/doc.pdf --source my_doc

# Ingest a whole directory
python scripts/cli.py ingest --dir ./data/docs/

# Or via API
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Your document text here..."], "source_name": "test"}'
```

### 4. Query

```bash
# CLI
python scripts/cli.py query "What is hybrid search?"

# API
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search?"}'
```

### 5. Check system status

```bash
python scripts/cli.py status
```

---

## API Reference

### `POST /query`

```json
{
  "query": "What is hybrid search?",
  "top_k": 5,
  "use_hyde": true
}
```

Response:
```json
{
  "query": "What is hybrid search?",
  "answer": "Hybrid search combines...",
  "sources": [
    {
      "text": "...",
      "source": "my_doc",
      "rerank_score": 0.9821
    }
  ],
  "latency_ms": 842.3,
  "trace_id": "abc-123",
  "cached": false
}
```

### `POST /ingest`

```json
{
  "documents": ["Full text of document..."],
  "source_name": "my_knowledge_base",
  "metadata": {"author": "John", "date": "2024-01"}
}
```

### `POST /ingest/file`

Upload a PDF or TXT file via multipart form.

### `GET /health`

Returns Qdrant and Redis health status.

### `GET /metrics`

Prometheus metrics endpoint.

---

## Evaluation

```bash
# Run RAGAS evaluation on your golden dataset
python scripts/cli.py eval --input data/eval_questions.json --output results/eval.json

# Or directly
python -m src.evaluate --input data/eval_questions.json
```

Output:
```
RAGAS RESULTS
========================================
  faithfulness           0.8921  █████████████████▉
  answer_relevancy       0.8534  █████████████████
  context_precision      0.7812  ███████████████▋
  context_recall         0.7341  ██████████████▋
```

### Experiment tracking

Change a config variable in `.env`, re-run eval, compare. Recommended experiments:

| Variable | Values to try | Expected impact |
|---|---|---|
| `CHUNK_SIZE` | 256, 512, 1024 | Recall vs. precision tradeoff |
| `HYDE_ENABLED` | true, false | Dense recall improvement |
| `TOP_K_RETRIEVE` | 10, 20, 50 | Context coverage |
| `USE_COHERE_RERANK` | false, true | Answer quality |
| `DENSE_WEIGHT` | 0.4, 0.6, 0.8 | Domain keyword vs. semantic |

---

## Project Structure

```
hybrid-rag/
├── api/
│   └── main.py              # FastAPI app — endpoints, auth, middleware
├── src/
│   ├── ingestion.py         # Chunking, embedding, Qdrant upsert
│   ├── retriever.py         # Dense + sparse + RRF fusion + HyDE
│   ├── reranker.py          # Cross-encoder (local + Cohere)
│   ├── graph.py             # LangGraph agent with query routing
│   ├── cache.py             # Redis caching layer
│   ├── observability.py     # Prometheus metrics + Langfuse traces
│   ├── evaluate.py          # RAGAS evaluation pipeline
│   ├── models.py            # Pydantic request/response schemas
│   └── logger.py            # Structured logging (structlog)
├── config/
│   └── settings.py          # Pydantic-settings config (reads .env)
├── tests/
│   └── test_rag.py          # Unit + integration tests
├── scripts/
│   └── cli.py               # Management CLI
├── monitoring/
│   ├── prometheus.yml        # Prometheus scrape config
│   └── grafana-datasources.yml
├── data/
│   └── eval_questions.json  # Sample evaluation dataset
├── docker/
│   └── Dockerfile
├── .github/
│   └── workflows/ci.yml     # GitHub Actions CI/CD
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Production checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Set `USE_COHERE_RERANK=true` for best reranking quality
- [ ] Configure `LANGFUSE_PUBLIC_KEY` for query tracing
- [ ] Set up Grafana alerts on `rag_query_latency_seconds` p95
- [ ] Run RAGAS eval after any config change (treat it as a regression test)
- [ ] Set up Qdrant Cloud or self-hosted Qdrant with persistent volumes
- [ ] Use Redis Sentinel or Redis Cluster for HA cache

---

## Key concepts for interviews

**Why hybrid over dense-only?** Dense search fails on exact keyword queries (product codes, proper nouns). BM25 handles these. Hybrid gets both.

**Why RRF over score normalization?** BM25 and cosine scores are on incompatible scales. RRF uses only rank position — completely scale-invariant.

**Why rerank after retrieval?** Bi-encoders embed query and doc independently (fast). Cross-encoders see them together (slow but accurate). Retrieve-20/rerank-5 gives you speed + quality.

**Why LangGraph over a simple function?** Stateful graph enables conditional routing, easy addition of new retrieval strategies (SQL, web), and visibility into each step for debugging.
