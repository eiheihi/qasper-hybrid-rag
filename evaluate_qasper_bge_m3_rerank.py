import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from bge_m3_embeddings import BGEM3DenseEmbeddings
from tune_qasper_bge_m3_hybrid import (
    average_metrics,
    metric_for_one_query,
    read_jsonl,
    tokenize,
    weighted_rrf,
)


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_DIR = Path("data/qasper/results")

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
RRF_K = 60
DENSE_WEIGHT = 4.0
PAIR_BATCH_SIZE = 2


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def score_in_small_batches(reranker, question: str, passages: list[str]) -> list[float]:
    all_scores = []
    for start in range(0, len(passages), PAIR_BATCH_SIZE):
        pairs = [
            [question, passage]
            for passage in passages[start:start + PAIR_BATCH_SIZE]
        ]
        scores = reranker.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        all_scores.extend(float(score) for score in scores)
    return all_scores


def main():
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    corpus_ids = [item["chunk_id"] for item in corpus]

    print(f"语料文本块：{len(corpus)}")
    print(f"评测问题：{len(eval_data)}")

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
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
        dense_ids = [document.metadata["chunk_id"] for document in dense_docs]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        fused_ids = weighted_rrf(dense_ids, bm25_ids, DENSE_WEIGHT)
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
        "split": "validation",
        "parameters": {
            "dense_embedding": "BAAI/bge-m3",
            "dense_candidates": DENSE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
            "final_k": FINAL_K,
            "rrf_k": RRF_K,
            "dense_weight": DENSE_WEIGHT,
            "reranker": RERANK_MODEL,
            "pair_batch_size": PAIR_BATCH_SIZE,
        },
        "bge_m3_hybrid_rrf": average_metrics(rrf_metrics),
        "bge_m3_hybrid_rerank": average_metrics(rerank_metrics),
    }
    summary["bge_m3_hybrid_rrf"]["p95_latency_seconds"] = percentile(rrf_latencies, 0.95)
    summary["bge_m3_hybrid_rerank"]["p95_end_to_end_latency_seconds"] = percentile(
        rerank_latencies, 0.95
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "bge_m3_hybrid_rerank.json"
    details_path = RESULT_DIR / "bge_m3_hybrid_rerank_details.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nBGE-M3 Hybrid RRF：")
    print(json.dumps(summary["bge_m3_hybrid_rrf"], ensure_ascii=False, indent=2))
    print("\nBGE-M3 Hybrid + Reranker：")
    print(json.dumps(summary["bge_m3_hybrid_rerank"], ensure_ascii=False, indent=2))
    print(f"\n汇总结果：{summary_path}")
    print(f"逐题明细：{details_path}")


if __name__ == "__main__":
    main()
