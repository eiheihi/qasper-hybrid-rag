"""Validation ablation: provide the reranker with the same document context as dense retrieval."""

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
from contextual_passages import contextual_text_by_chunk_id
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
BASE_CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
BASE_COLLECTION = "qasper_bge_m3_dense"
CONTEXTUAL_CHROMA_DIR = "data/qasper/chroma_bge_m3_contextual"
CONTEXTUAL_COLLECTION = "qasper_bge_m3_contextual"
RESULT_PATH = Path("data/qasper/results/context_aware_rerank_ablation.json")

CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def percentile(values: list[float], p: float = 0.95) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def rerank_ids(reranker, question: str, candidate_ids: list[str], passage_lookup: dict[str, str]):
    start = time.perf_counter()
    scores = score_in_small_batches(
        reranker, question, [passage_lookup[chunk_id] for chunk_id in candidate_ids]
    )
    ranked_ids = [
        chunk_id
        for chunk_id, _ in sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)[
            :FINAL_K
        ]
    ]
    return ranked_ids, time.perf_counter() - start


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError("请先运行 build_qasper_bge_m3_sparse_index.py")

    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    raw_passages = {item["chunk_id"]: item["text"] for item in corpus}
    contextual_passages = contextual_text_by_chunk_id(corpus, window=1)
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
    base_store = Chroma(
        collection_name=BASE_COLLECTION,
        persist_directory=BASE_CHROMA_DIR,
        embedding_function=embeddings,
    )
    contextual_store = Chroma(
        collection_name=CONTEXTUAL_COLLECTION,
        persist_directory=CONTEXTUAL_CHROMA_DIR,
        embedding_function=embeddings,
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

    systems = {
        "old_three_way_raw_rerank": {"metrics": [], "recalls": [], "latencies": []},
        "contextual_candidates_raw_rerank": {"metrics": [], "recalls": [], "latencies": []},
        "contextual_candidates_context_rerank": {"metrics": [], "recalls": [], "latencies": []},
    }

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]
        base_ids = [
            document.metadata["chunk_id"]
            for document in base_store.similarity_search(question, k=CANDIDATES)
        ]
        contextual_ids = [
            document.metadata["chunk_id"]
            for document in contextual_store.similarity_search(question, k=CANDIDATES)
        ]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]
        sparse_indices = sparse_search(
            sparse_encoder.encode_query(question), inverted_index, CANDIDATES
        )
        sparse_ids = [corpus_ids[i] for i in sparse_indices]

        old_candidates = weighted_rrf_three_way(
            base_ids, bm25_ids, sparse_ids, dense_weight=2.0, sparse_weight=1.0
        )[:RERANK_CANDIDATES]
        context_candidates = weighted_rrf_three_way(
            contextual_ids, bm25_ids, sparse_ids, dense_weight=2.0, sparse_weight=1.0
        )[:RERANK_CANDIDATES]
        experiment_inputs = {
            "old_three_way_raw_rerank": (old_candidates, raw_passages),
            "contextual_candidates_raw_rerank": (context_candidates, raw_passages),
            "contextual_candidates_context_rerank": (context_candidates, contextual_passages),
        }

        for name, (candidate_ids, passages) in experiment_inputs.items():
            ranked_ids, elapsed = rerank_ids(reranker, question, candidate_ids, passages)
            systems[name]["metrics"].append(metric_for_one_query(ranked_ids, gold_ids))
            systems[name]["recalls"].append(candidate_recall(candidate_ids, gold_ids))
            systems[name]["latencies"].append(elapsed)

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "systems": {
            name: {
                **average_metrics(values["metrics"]),
                "rerank_candidate_recall_at_20": sum(values["recalls"]) / len(values["recalls"]),
                "p95_rerank_only_latency_seconds": percentile(values["latencies"]),
            }
            for name, values in systems.items()
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n上下文感知重排消融：")
    for name, values in report["systems"].items():
        print(
            f"{name:43} Hit@5={values['hit_at_5']:.4f} "
            f"MRR@10={values['mrr_at_10']:.4f} "
            f"nDCG@10={values['ndcg_at_10']:.4f} "
            f"CandRecall@20={values['rerank_candidate_recall_at_20']:.4f}"
        )
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
