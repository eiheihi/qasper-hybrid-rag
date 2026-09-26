"""Independent test evaluation for the frozen BGE-M3 hybrid RAG retriever.

The parameters in this file were selected on QASPER validation only. Do not tune
them after seeing this test result; create a new validation experiment instead.
"""

import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

from FlagEmbedding import FlagReranker
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from bge_m3_embeddings import BGEM3DenseEmbeddings
from tune_qasper_bge_m3_hybrid import (
    average_metrics,
    metric_for_one_query,
    read_jsonl,
    tokenize,
    weighted_rrf,
)


CORPUS_PATH = Path("data/qasper_test/corpus.jsonl")
EVAL_PATH = Path("data/qasper_test/eval.jsonl")
CHROMA_DIR = "data/qasper_test/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_test_bge_m3_dense"
RESULT_DIR = Path("data/qasper_test/results")

# Frozen after validation-set tuning.
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
    scores = []
    for start in range(0, len(passages), PAIR_BATCH_SIZE):
        pairs = [[question, text] for text in passages[start : start + PAIR_BATCH_SIZE]]
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

    if not corpus or not eval_data:
        raise RuntimeError("test 数据为空，请先运行 prepare_qasper_test.py")

    print(f"语料文本块：{len(corpus)}")
    print(f"评测问题：{len(eval_data)}")
    print("使用冻结参数：Dense=30、BM25=30、RRF dense_weight=4.0、Rerank=20")

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

    dense_metrics, rrf_metrics, rerank_metrics = [], [], []
    dense_latencies, rrf_latencies, rerank_latencies = [], [], []
    details = []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]
        start_time = time.perf_counter()

        dense_docs = vector_store.similarity_search(question, k=DENSE_CANDIDATES)
        dense_ids = [document.metadata["chunk_id"] for document in dense_docs]
        dense_elapsed = time.perf_counter() - start_time

        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        fused_ids = weighted_rrf(dense_ids, bm25_ids, DENSE_WEIGHT)
        rrf_top10 = fused_ids[:FINAL_K]
        rrf_elapsed = time.perf_counter() - start_time

        rerank_input_ids = fused_ids[:RERANK_CANDIDATES]
        passages = [corpus_by_id[chunk_id]["text"] for chunk_id in rerank_input_ids]
        scores = score_in_small_batches(reranker, question, passages)
        reranked_pairs = sorted(
            zip(rerank_input_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        reranked_ids = [chunk_id for chunk_id, _ in reranked_pairs[:FINAL_K]]
        total_elapsed = time.perf_counter() - start_time

        dense_result = metric_for_one_query(dense_ids[:FINAL_K], gold_ids)
        rrf_result = metric_for_one_query(rrf_top10, gold_ids)
        rerank_result = metric_for_one_query(reranked_ids, gold_ids)
        dense_metrics.append(dense_result)
        rrf_metrics.append(rrf_result)
        rerank_metrics.append(rerank_result)
        dense_latencies.append(dense_elapsed)
        rrf_latencies.append(rrf_elapsed)
        rerank_latencies.append(total_elapsed)

        details.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "gold_chunk_ids": gold_ids,
                "dense_top10": dense_ids[:FINAL_K],
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
                "dense_metrics": dense_result,
                "rrf_metrics": rrf_result,
                "rerank_metrics": rerank_result,
                "end_to_end_latency_seconds": total_elapsed,
            }
        )

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    summary = {
        "split": "test",
        "protocol": "frozen_parameters_selected_on_validation_only",
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
        "bge_m3_dense": average_metrics(dense_metrics),
        "bge_m3_hybrid_rrf": average_metrics(rrf_metrics),
        "bge_m3_hybrid_rerank": average_metrics(rerank_metrics),
    }
    summary["bge_m3_dense"]["p95_retrieval_latency_seconds"] = percentile(
        dense_latencies, 0.95
    )
    summary["bge_m3_hybrid_rrf"]["p95_latency_seconds"] = percentile(rrf_latencies, 0.95)
    summary["bge_m3_hybrid_rerank"]["p95_end_to_end_latency_seconds"] = percentile(
        rerank_latencies, 0.95
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "bge_m3_hybrid_rerank.json"
    details_path = RESULT_DIR / "bge_m3_hybrid_rerank_details.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as file:
        for detail in details:
            file.write(json.dumps(detail, ensure_ascii=False) + "\n")

    for name in ("bge_m3_dense", "bge_m3_hybrid_rrf", "bge_m3_hybrid_rerank"):
        print(f"\n{name}：")
        print(json.dumps(summary[name], ensure_ascii=False, indent=2))
    print(f"\n汇总结果：{summary_path}")
    print(f"逐题明细：{details_path}")


if __name__ == "__main__":
    main()
