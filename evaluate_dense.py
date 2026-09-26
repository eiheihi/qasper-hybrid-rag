import json
import math
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_dense"
RESULTS_DIR = Path("data/qasper/results")
RESULT_PATH = RESULTS_DIR / "dense_baseline.json"
DETAIL_PATH = RESULTS_DIR / "dense_baseline_details.jsonl"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 10


def load_eval_data() -> list[dict]:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & gold))


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0

    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in gold
    )

    ideal_count = min(len(gold), k)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = max(0, math.ceil(len(values) * 0.95) - 1)
    return values[index]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_data = load_eval_data()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        cache_folder="data/models",
    )

    vector_store = Chroma(
        collection_name="qasper_dense_baseline",
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

        retrieved_chunk_ids = [
            document.metadata["chunk_id"]
            for document in documents
        ]
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
                "recall_at_5": recall_at_k(
                    retrieved_chunk_ids,
                    gold_chunk_ids,
                    5,
                ),
                "recall_at_10": recall_at_k(
                    retrieved_chunk_ids,
                    gold_chunk_ids,
                    10,
                ),
                "mrr_at_10": mrr_at_k(
                    retrieved_chunk_ids,
                    gold_chunk_ids,
                    10,
                ),
                "ndcg_at_10": ndcg_at_k(
                    retrieved_chunk_ids,
                    gold_chunk_ids,
                    10,
                ),
                "latency_seconds": latency_seconds,
            }
        )

        print(f"已评测 {index}/{len(eval_data)} 条问题")

    metrics = {
        "system": "dense_baseline",
        "questions": len(details),
        "hit_at_1": sum(item["hit_at_1"] for item in details) / len(details),
        "hit_at_3": sum(item["hit_at_3"] for item in details) / len(details),
        "hit_at_5": sum(item["hit_at_5"] for item in details) / len(details),
        "recall_at_5": sum(
            item["recall_at_5"] for item in details
        ) / len(details),
        "recall_at_10": sum(
            item["recall_at_10"] for item in details
        ) / len(details),
        "mrr_at_10": sum(item["mrr_at_10"] for item in details) / len(details),
        "ndcg_at_10": sum(
            item["ndcg_at_10"] for item in details
        ) / len(details),
        "p95_retrieval_latency_seconds": percentile_95(latencies),
    }

    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with DETAIL_PATH.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nDense 基线结果：")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()