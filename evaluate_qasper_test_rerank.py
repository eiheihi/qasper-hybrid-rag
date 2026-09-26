import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

CORPUS_PATH = Path("data/qasper_test/corpus.jsonl")
EVAL_PATH = Path("data/qasper_test/eval.jsonl")
CHROMA_DIR = "data/qasper_test/chroma_dense"
COLLECTION_NAME = "qasper_test_dense_baseline"
RESULT_DIR = Path("data/qasper_test/results")

# Frozen validation-selected configuration. Do not tune on the test set.
DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
RRF_K = 60
DENSE_WEIGHT = 0.25
PAIR_BATCH_SIZE = 2


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def weighted_rrf(dense_ids: list[str], bm25_ids: list[str]) -> list[str]:
    scores = defaultdict(float)
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] += DENSE_WEIGHT / (RRF_K + rank)
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] += 1.0 / (RRF_K + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def metric_for_one_query(retrieved_ids: list[str], gold_ids: list[str]) -> dict:
    gold = set(gold_ids)
    ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in gold
    ]

    def hit_at(k: int) -> float:
        return float(any(rank <= k for rank in ranks))

    def recall_at(k: int) -> float:
        return len(set(retrieved_ids[:k]) & gold) / len(gold)

    first_rank = min(ranks) if ranks else None
    mrr_at_10 = 1 / first_rank if first_rank and first_rank <= 10 else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal_count = min(len(gold), 10)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr_at_10": mrr_at_10,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
    }


def average_metrics(metric_list: list[dict]) -> dict:
    return {
        name: sum(item[name] for item in metric_list) / len(metric_list)
        for name in metric_list[0]
    }


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def score_in_small_batches(reranker, question: str, passages: list[str]) -> list[float]:
    scores = []
    for start in range(0, len(passages), PAIR_BATCH_SIZE):
        pairs = [[question, passage] for passage in passages[start:start + PAIR_BATCH_SIZE]]
        batch_scores = reranker.compute_score(pairs)
        if isinstance(batch_scores, (int, float)):
            batch_scores = [batch_scores]
        scores.extend(float(score) for score in batch_scores)
    return scores


def main():
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    corpus_ids = [item["chunk_id"] for item in corpus]

    print(f"语料文本块：{len(corpus)}")
    print(f"test 评测问题：{len(eval_data)}")

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        cache_folder="data/models",
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print(f"加载重排模型：{RERANK_MODEL}")
    reranker = FlagReranker(
        RERANK_MODEL,
        query_max_length=256,
        passage_max_length=512,
        use_fp16=True,
        devices=["cuda:0"],
    )

    rrf_metrics, rerank_metrics = [], []
    rrf_latencies, rerank_latencies = [], []
    details = []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        total_start = time.perf_counter()

        dense_docs = vector_store.similarity_search(question, k=DENSE_CANDIDATES)
        dense_ids = [doc.metadata["chunk_id"] for doc in dense_docs]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        fused_ids = weighted_rrf(dense_ids, bm25_ids)
        rrf_top10 = fused_ids[:FINAL_K]
        rrf_elapsed = time.perf_counter() - total_start

        rerank_input_ids = fused_ids[:RERANK_CANDIDATES]
        passages = [corpus_by_id[chunk_id]["text"] for chunk_id in rerank_input_ids]
        scores = score_in_small_batches(reranker, question, passages)
        reranked_pairs = sorted(
            zip(rerank_input_ids, scores), key=lambda item: item[1], reverse=True
        )
        reranked_ids = [chunk_id for chunk_id, _ in reranked_pairs[:FINAL_K]]
        total_elapsed = time.perf_counter() - total_start

        rrf_result = metric_for_one_query(rrf_top10, item["gold_chunk_ids"])
        rerank_result = metric_for_one_query(reranked_ids, item["gold_chunk_ids"])
        rrf_metrics.append(rrf_result)
        rerank_metrics.append(rerank_result)
        rrf_latencies.append(rrf_elapsed)
        rerank_latencies.append(total_elapsed)

        details.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "gold_chunk_ids": item["gold_chunk_ids"],
                "rrf_top10": rrf_top10,
                "reranked_top10": [
                    {
                        "chunk_id": chunk_id,
                        "reranker_score": score,
                        "title": corpus_by_id[chunk_id]["title"],
                        "section": corpus_by_id[chunk_id]["section"],
                    }
                    for chunk_id, score in reranked_pairs[:FINAL_K]
                ],
                "rrf_metrics": rrf_result,
                "rerank_metrics": rerank_result,
                "end_to_end_latency_seconds": total_elapsed,
            }
        )

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    summary = {
        "split": "test",
        "questions": len(eval_data),
        "parameters": {
            "dense_candidates": DENSE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
            "final_k": FINAL_K,
            "rrf_k": RRF_K,
            "dense_weight": DENSE_WEIGHT,
            "reranker": RERANK_MODEL,
            "pair_batch_size": PAIR_BATCH_SIZE,
        },
        "hybrid_rrf_dense_0.25": average_metrics(rrf_metrics),
        "hybrid_rrf_rerank_bge_v2_m3": average_metrics(rerank_metrics),
    }
    summary["hybrid_rrf_dense_0.25"]["p95_latency_seconds"] = percentile(
        rrf_latencies, 0.95
    )
    summary["hybrid_rrf_rerank_bge_v2_m3"]["p95_end_to_end_latency_seconds"] = percentile(
        rerank_latencies, 0.95
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "hybrid_rerank_bge_v2_m3.json"
    details_path = RESULT_DIR / "hybrid_rerank_bge_v2_m3_details.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nTest RRF 基线：")
    print(json.dumps(summary["hybrid_rrf_dense_0.25"], ensure_ascii=False, indent=2))
    print("\nTest RRF + Reranker：")
    print(json.dumps(summary["hybrid_rrf_rerank_bge_v2_m3"], ensure_ascii=False, indent=2))
    print(f"\n汇总结果：{summary_path}")
    print(f"逐题明细：{details_path}")


if __name__ == "__main__":
    main()
