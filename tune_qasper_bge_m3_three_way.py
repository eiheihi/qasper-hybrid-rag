"""Validation-only tuning for dense + BM25 + BGE-M3 native sparse fusion."""

import json
from collections import defaultdict
from pathlib import Path

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from bge_m3_embeddings import BGEM3DenseEmbeddings
from bge_m3_sparse import BGEM3SparseEncoder, sparse_search
from tune_qasper_bge_m3_hybrid import (
    average_metrics,
    metric_for_one_query,
    read_jsonl,
    tokenize,
)


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_PATH = Path("data/qasper/results/bge_m3_three_way_rrf_tuning.json")

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
SPARSE_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
RRF_K = 60
BM25_WEIGHT = 1.0
DENSE_WEIGHTS = [2.0, 4.0]
SPARSE_WEIGHTS = [0.25, 0.5, 1.0, 2.0]


def weighted_rrf_three_way(
    dense_ids: list[str],
    bm25_ids: list[str],
    sparse_ids: list[str],
    dense_weight: float,
    sparse_weight: float,
) -> list[str]:
    scores = defaultdict(float)
    for candidate_ids, weight in (
        (dense_ids, dense_weight),
        (bm25_ids, BM25_WEIGHT),
        (sparse_ids, sparse_weight),
    ):
        for rank, chunk_id in enumerate(candidate_ids, start=1):
            scores[chunk_id] += weight / (RRF_K + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]


def candidate_recall(candidate_ids: list[str], gold_ids: list[str]) -> float:
    return len(set(candidate_ids) & set(gold_ids)) / len(gold_ids)


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError(
            "未找到原生稀疏索引，请先运行 python build_qasper_bge_m3_sparse_index.py"
        )

    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))
    if sparse_payload["corpus_size"] != len(corpus):
        raise RuntimeError("稀疏索引与当前语料不一致，请重新建索引")

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    dense_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
    )
    sparse_encoder = BGEM3SparseEncoder(batch_size=4, max_length=512)
    inverted_index = sparse_payload["inverted_index"]

    systems = {
        f"three_way_dense_{dense_weight}_sparse_{sparse_weight}": []
        for dense_weight in DENSE_WEIGHTS
        for sparse_weight in SPARSE_WEIGHTS
    }
    candidate_metrics = {name: [] for name in systems}

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

        for dense_weight in DENSE_WEIGHTS:
            for sparse_weight in SPARSE_WEIGHTS:
                name = f"three_way_dense_{dense_weight}_sparse_{sparse_weight}"
                fused_ids = weighted_rrf_three_way(
                    dense_ids,
                    bm25_ids,
                    sparse_ids,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                )
                systems[name].append(metric_for_one_query(fused_ids[:FINAL_K], gold_ids))
                candidate_metrics[name].append(
                    candidate_recall(fused_ids[:RERANK_CANDIDATES], gold_ids)
                )

        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "parameters": {
            "dense_candidates": DENSE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "sparse_candidates": SPARSE_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
            "rrf_k": RRF_K,
            "bm25_weight": BM25_WEIGHT,
        },
        "systems": {
            name: {
                **average_metrics(metric_list),
                "rerank_candidate_recall_at_20": sum(candidate_metrics[name])
                / len(candidate_metrics[name]),
            }
            for name, metric_list in systems.items()
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n三路 RRF 对比（按 nDCG@10 排序）：")
    ranked = sorted(
        report["systems"].items(), key=lambda pair: pair[1]["ndcg_at_10"], reverse=True
    )
    for name, metrics in ranked:
        print(
            f"{name:38} Hit@5={metrics['hit_at_5']:.4f} "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@10={metrics['ndcg_at_10']:.4f} "
            f"CandRecall@20={metrics['rerank_candidate_recall_at_20']:.4f}"
        )
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
