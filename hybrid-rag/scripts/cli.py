#!/usr/bin/env python3
"""
CLI tool for managing the Hybrid RAG system.

Usage:
    python scripts/cli.py ingest --file path/to/doc.pdf --source my_doc
    python scripts/cli.py ingest --dir path/to/docs/
    python scripts/cli.py query "What is hybrid search?"
    python scripts/cli.py eval --input data/eval_questions.json
    python scripts/cli.py status
"""
import argparse
import json
import os
import sys
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.logger import configure_logging, get_logger

configure_logging()
logger = get_logger("cli")
settings = get_settings()


def cmd_ingest(args):
    from src.ingestion import ingest_documents, extract_text_from_pdf

    files = []
    if args.file:
        files = [args.file]
    elif args.dir:
        files = glob.glob(os.path.join(args.dir, "**/*.pdf"), recursive=True)
        files += glob.glob(os.path.join(args.dir, "**/*.txt"), recursive=True)

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} file(s) to ingest...")

    for path in files:
        source_name = args.source or os.path.basename(path)
        print(f"  Ingesting: {path} → source='{source_name}'")

        if path.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()

        result = ingest_documents(
            texts=[text],
            source_name=source_name,
        )
        print(f"  ✓ {result['chunks_indexed']} chunks indexed in {result['duration_ms']:.0f}ms")

    print("\nIngestion complete.")


def cmd_query(args):
    from src.graph import run_query

    query = args.query
    print(f"\nQuery: {query}\n")
    print("Running hybrid RAG pipeline...")

    t0 = time.time()
    result = run_query(query)
    elapsed = (time.time() - t0) * 1000

    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"SOURCES ({len(result.get('sources', []))} retrieved):")
    print(f"{'='*60}")

    for i, src in enumerate(result.get("sources", []), 1):
        score = src.get("rerank_score", 0)
        source = src.get("source", "unknown")
        preview = src["text"][:120].replace("\n", " ")
        print(f"[{i}] Score={score:.4f} | {source}")
        print(f"    {preview}...")
        print()

    timings = result.get("timings", {})
    print(f"Latency:   {elapsed:.0f}ms total")
    print(f"  Retrieve: {timings.get('retrieve_total_ms', '?')}ms")
    print(f"  Rerank:   {timings.get('rerank_ms', '?')}ms")
    print(f"  Generate: {timings.get('generate_ms', '?')}ms")
    print(f"Trace ID:  {result.get('trace_id', 'n/a')}")


def cmd_eval(args):
    from src.evaluate import run_evaluation

    print(f"Loading eval questions from: {args.input}")
    with open(args.input) as f:
        qa_pairs = json.load(f)

    print(f"Running RAGAS evaluation on {len(qa_pairs)} questions...\n")
    summary = run_evaluation(qa_pairs)

    output_path = args.output or "results/eval_output.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*40}")
    print("RAGAS RESULTS")
    print(f"{'='*40}")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for m in metrics:
        score = summary.get(m, 0)
        bar = "█" * int(score * 20)
        print(f"  {m:<22} {score:.4f}  {bar}")
    print(f"\nSaved to: {output_path}")


def cmd_status(args):
    from src.ingestion import get_qdrant_client
    from src.cache import get_redis

    print("=== System Status ===\n")

    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        collection_name = settings.qdrant_collection
        print(f"Qdrant:     OK ({settings.qdrant_url})")
        print(f"Collections: {[c.name for c in collections]}")

        matching = [c for c in collections if c.name == collection_name]
        if matching:
            info = client.get_collection(collection_name)
            print(f"  '{collection_name}': {info.points_count} points")
        else:
            print(f"  '{collection_name}': NOT FOUND (run ingest first)")
    except Exception as e:
        print(f"Qdrant:     FAILED — {e}")

    try:
        r = get_redis()
        if r:
            info = r.info()
            print(f"\nRedis:      OK ({settings.redis_url})")
            print(f"  Memory: {info.get('used_memory_human', '?')}")
            print(f"  Keys:   {r.dbsize()}")
        else:
            print(f"\nRedis:      UNAVAILABLE")
    except Exception as e:
        print(f"\nRedis:      FAILED — {e}")

    print(f"\nSettings:")
    print(f"  Chunk size:    {settings.chunk_size}")
    print(f"  Top-K:         {settings.top_k_retrieve} → rerank → {settings.top_n_rerank}")
    print(f"  RRF k:         {settings.rrf_k}")
    print(f"  Dense weight:  {settings.dense_weight}")
    print(f"  Sparse weight: {settings.sparse_weight}")
    print(f"  HyDE:          {settings.hyde_enabled}")
    print(f"  Cohere rerank: {settings.use_cohere_rerank}")
    print(f"  Environment:   {settings.environment}")


def main():
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="Hybrid RAG management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("--file", help="Path to a single PDF or TXT file")
    p_ingest.add_argument("--dir", help="Directory to recursively ingest")
    p_ingest.add_argument("--source", help="Source label (defaults to filename)")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = subparsers.add_parser("query", help="Run a RAG query")
    p_query.add_argument("query", help="Natural language question")
    p_query.set_defaults(func=cmd_query)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run RAGAS evaluation")
    p_eval.add_argument("--input", required=True, help="JSON file with QA pairs")
    p_eval.add_argument("--output", help="Output path for results JSON")
    p_eval.set_defaults(func=cmd_eval)

    # status
    p_status = subparsers.add_parser("status", help="Check system status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
