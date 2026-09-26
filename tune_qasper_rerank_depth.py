"""Validation-only quality-latency sweep for reranker candidate depth."""

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
from bge_m3_sparse import BGEM3SparseEncoder, sparse_search
from evaluate_qasper_bge_m3_rerank import score_in_small_batches
from tune_qasper_bge_m3_hybrid import (
    average_metrics,
    metric_for_one_query,
    read_jsonl,
    tokenize,
)
from tune_qasper_bge_m3_three_way import candidate_recall, weighted_rrf_three_way


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_PATH = Path("data/qasper/results/rerank_depth_tuning.json")

SOURCE_CANDIDATES = 30
RERANK_DEPTHS = [10, 20, 30, 40, 60]
FINAL_K = 10
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Frozen first-stage fusion from the best validated and tested three-way system.
DENSE_WEIGHT = 2.0
SPARSE_WEIGHT = 1.0


def percentile(values: list[float], p: float = 0.95) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError("请先运行 build_qasper_bge_m3_sparse_index.py")

    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    passages = {item["chunk_id"]: item["text"] for item in corpus}
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    dense_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
    )
    sparse_encoder = BGEM3SparseEncoder(batch_size=4, max_length=512)
    reranker = FlagReranker(
        RERANK_MODEL,
        query_max_length=256,
        passage_max_length=512,
        use_fp16=True,
        devices=["cuda:0"],
    )
    inverted_index = sparse_payload["inverted_index"]
    systems = {f"rerank_top_{depth}": [] for depth in RERANK_DEPTHS}
    candidate_metrics = {name: [] for name in systems}
    latencies = {name: [] for name in systems}

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]
        dense_ids = [
            document.metadata["chunk_id"]
            for document in dense_store.similarity_search(question, k=SOURCE_CANDIDATES)
        ]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:SOURCE_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]
        sparse_indices = sparse_search(
            sparse_encoder.encode_query(question), inverted_index, SOURCE_CANDIDATES
        )
        sparse_ids = [corpus_ids[i] for i in sparse_indices]
        fused_ids = weighted_rrf_three_way(
            dense_ids,
            bm25_ids,
            sparse_ids,
            dense_weight=DENSE_WEIGHT,
            sparse_weight=SPARSE_WEIGHT,
        )

        for depth in RERANK_DEPTHS:
            candidate_ids = fused_ids[:depth]
            start = time.perf_counter()
            scores = score_in_small_batches(
                reranker, question, [passages[chunk_id] for chunk_id in candidate_ids]
            )
            reranked_ids = [
                chunk_id
                for chunk_id, _ in sorted(
                    zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True
                )[:FINAL_K]
            ]
            name = f"rerank_top_{depth}"
            systems[name].append(metric_for_one_query(reranked_ids, gold_ids))
            candidate_metrics[name].append(candidate_recall(candidate_ids, gold_ids))
            latencies[name].append(time.perf_counter() - start)

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "source_candidates_per_retriever": SOURCE_CANDIDATES,
        "frozen_first_stage": {"dense_weight": DENSE_WEIGHT, "sparse_weight": SPARSE_WEIGHT},
        "systems": {
            name: {
                **average_metrics(values),
                "rerank_candidate_recall": sum(candidate_metrics[name])
                / len(candidate_metrics[name]),
                "p95_rerank_only_latency_seconds": percentile(latencies[name]),
            }
            for name, values in systems.items()
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nReranker 候选深度对比：")
    for name, values in report["systems"].items():
        print(
            f"{name:18} Hit@5={values['hit_at_5']:.4f} "
            f"MRR@10={values['mrr_at_10']:.4f} "
            f"nDCG@10={values['ndcg_at_10']:.4f} "
            f"CandRecall={values['rerank_candidate_recall']:.4f} "
            f"P95={values['p95_rerank_only_latency_seconds']:.3f}s"
        )
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
