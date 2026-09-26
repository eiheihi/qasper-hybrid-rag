import json
import time
from pathlib import Path

from langchain_chroma import Chroma

from bge_m3_embeddings import BGEM3DenseEmbeddings
from evaluate_dense import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile_95,
    recall_at_k,
)


EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULTS_DIR = Path("data/qasper/results")
RESULT_PATH = RESULTS_DIR / "bge_m3_dense_baseline.json"
DETAIL_PATH = RESULTS_DIR / "bge_m3_dense_baseline_details.jsonl"
TOP_K = 10


def load_eval_data() -> list[dict]:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_data = load_eval_data()

    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    details = []
    latencies = []

    for index, item in enumerate(eval_data, start=1):
        start_time = time.perf_counter()
        documents = vector_store.similarity_search(item["question"], k=TOP_K)
        latency_seconds = time.perf_counter() - start_time
        latencies.append(latency_seconds)

        retrieved_chunk_ids = [document.metadata["chunk_id"] for document in documents]
        gold_chunk_ids = set(item["gold_chunk_ids"])

        details.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_chunk_ids": list(gold_chunk_ids),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "hit_at_1": hit_at_k(retrieved_chunk_ids, gold_chunk_ids, 1),
                "hit_at_3": hit_at_k(retrieved_chunk_ids, gold_chunk_ids, 3),
                "hit_at_5": hit_at_k(retrieved_chunk_ids, gold_chunk_ids, 5),
                "recall_at_5": recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 5),
                "recall_at_10": recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 10),
                "mrr_at_10": mrr_at_k(retrieved_chunk_ids, gold_chunk_ids, 10),
                "ndcg_at_10": ndcg_at_k(retrieved_chunk_ids, gold_chunk_ids, 10),
                "latency_seconds": latency_seconds,
            }
        )
        print(f"已评测 {index}/{len(eval_data)} 条问题")

    metrics = {
        "system": "bge_m3_dense_baseline",
        "questions": len(details),
        "hit_at_1": sum(item["hit_at_1"] for item in details) / len(details),
        "hit_at_3": sum(item["hit_at_3"] for item in details) / len(details),
        "hit_at_5": sum(item["hit_at_5"] for item in details) / len(details),
        "recall_at_5": sum(item["recall_at_5"] for item in details) / len(details),
        "recall_at_10": sum(item["recall_at_10"] for item in details) / len(details),
        "mrr_at_10": sum(item["mrr_at_10"] for item in details) / len(details),
        "ndcg_at_10": sum(item["ndcg_at_10"] for item in details) / len(details),
        "p95_retrieval_latency_seconds": percentile_95(latencies),
    }

    RESULT_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with DETAIL_PATH.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nBGE-M3 Dense 基线结果：")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
