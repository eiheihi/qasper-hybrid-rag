"""Independent test evaluation of the frozen three-way fusion configuration."""

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
    weighted_rrf,
)
from tune_qasper_bge_m3_three_way import weighted_rrf_three_way


CORPUS_PATH = Path("data/qasper_test/corpus.jsonl")
EVAL_PATH = Path("data/qasper_test/eval.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper_test/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper_test/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_test_bge_m3_dense"
RESULT_PATH = Path("data/qasper_test/results/bge_m3_three_way_rerank.json")

# Frozen on validation. These values must not be changed after viewing test results.
DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
SPARSE_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
PAIR_BATCH_SIZE = 2
TWO_WAY_DENSE_WEIGHT = 4.0
THREE_WAY_DENSE_WEIGHT = 2.0
THREE_WAY_SPARSE_WEIGHT = 1.0


def percentile(values: list[float], p: float = 0.95) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def candidate_recall(candidate_ids: list[str], gold_ids: list[str]) -> float:
    return len(set(candidate_ids) & set(gold_ids)) / len(gold_ids)


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError(
            "未找到 test 稀疏索引，请先运行 build_qasper_test_bge_m3_sparse_index.py"
        )

    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))
    if sparse_payload["corpus_size"] != len(corpus):
        raise RuntimeError("test 稀疏索引与当前语料不一致，请重新建索引")

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
    systems = {
        "two_way_baseline": [],
        "three_way_frozen": [],
    }
    candidate_recalls = {name: [] for name in systems}
    latencies = {name: [] for name in systems}
    inverted_index = sparse_payload["inverted_index"]

    print(f"评测问题：{len(eval_data)}")
    print("运行冻结配置：三路 Dense=2.0、Sparse=1.0、BM25=1.0")
    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]
        dense_ids = [
            document.metadata["chunk_id"]
            for document in dense_store.similarity_search(question, k=DENSE_CANDIDATES)
        ]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]
        sparse_indices = sparse_search(
            sparse_encoder.encode_query(question), inverted_index, SPARSE_CANDIDATES
        )
        sparse_ids = [corpus_ids[i] for i in sparse_indices]

        candidate_lists = {
            "two_way_baseline": weighted_rrf(dense_ids, bm25_ids, TWO_WAY_DENSE_WEIGHT)[
                :RERANK_CANDIDATES
            ],
            "three_way_frozen": weighted_rrf_three_way(
                dense_ids,
                bm25_ids,
                sparse_ids,
                dense_weight=THREE_WAY_DENSE_WEIGHT,
                sparse_weight=THREE_WAY_SPARSE_WEIGHT,
            )[:RERANK_CANDIDATES],
        }

        for name, candidate_ids in candidate_lists.items():
            start = time.perf_counter()
            scores = score_in_small_batches(
                reranker,
                question,
                [corpus_by_id[chunk_id]["text"] for chunk_id in candidate_ids],
            )
            reranked = [
                chunk_id
                for chunk_id, _ in sorted(
                    zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True
                )[:FINAL_K]
            ]
            systems[name].append(metric_for_one_query(reranked, gold_ids))
            candidate_recalls[name].append(candidate_recall(candidate_ids, gold_ids))
            latencies[name].append(time.perf_counter() - start)

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "test",
        "protocol": "frozen_after_validation_selection",
        "systems": {
            name: {
                **average_metrics(values),
                "rerank_candidate_recall_at_20": sum(candidate_recalls[name])
                / len(candidate_recalls[name]),
                "p95_rerank_only_latency_seconds": percentile(latencies[name]),
            }
            for name, values in systems.items()
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nIndependent test reranker comparison:")
    for name, values in report["systems"].items():
        print(
            f"{name:18} Hit@5={values['hit_at_5']:.4f} "
            f"MRR@10={values['mrr_at_10']:.4f} "
            f"nDCG@10={values['ndcg_at_10']:.4f} "
            f"CandRecall@20={values['rerank_candidate_recall_at_20']:.4f}"
        )
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
