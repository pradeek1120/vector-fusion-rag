"""
Evaluation pipeline using RAGAS.

Usage:
    python -m src.evaluate --input data/eval_questions.json --output results/eval_out.json

Input format (eval_questions.json):
    [{"question": "...", "ground_truth": "..."}, ...]
"""
import json
import argparse
import time
from datetime import datetime

from datasets import Dataset

from config.settings import get_settings
from src.graph import run_query
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_ragas_dataset(qa_pairs: list[dict]) -> Dataset:
    rows = []
    for qa in qa_pairs:
        q = qa["question"]
        logger.info("eval_running_query", question=q[:60])

        result = run_query(q)

        contexts = [s["text"] for s in result.get("sources", [])]
        answer = result.get("answer", "")

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts if contexts else [""],
            "ground_truth": qa.get("ground_truth", ""),
        })

    return Dataset.from_list(rows)


def run_evaluation(qa_pairs: list[dict]) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        logger.error("ragas_not_installed")
        return {}

    logger.info("eval_start", num_questions=len(qa_pairs))
    t0 = time.time()

    dataset = build_ragas_dataset(qa_pairs)

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    df = results.to_pandas()
    summary = {
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
        "context_recall": round(float(df["context_recall"].mean()), 4),
        "num_questions": len(qa_pairs),
        "duration_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "chunk_size": settings.chunk_size,
            "top_k_retrieve": settings.top_k_retrieve,
            "top_n_rerank": settings.top_n_rerank,
            "rrf_k": settings.rrf_k,
            "dense_weight": settings.dense_weight,
            "hyde_enabled": settings.hyde_enabled,
            "use_cohere_rerank": settings.use_cohere_rerank,
        },
    }

    logger.info("eval_complete", **summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON eval questions")
    parser.add_argument("--output", default="results/eval_output.json")
    args = parser.parse_args()

    with open(args.input) as f:
        qa_pairs = json.load(f)

    summary = run_evaluation(qa_pairs)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RAGAS Evaluation Results ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")
