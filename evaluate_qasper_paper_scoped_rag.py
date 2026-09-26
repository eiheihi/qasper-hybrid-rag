"""Evaluate the user-selected-paper RAG mode separately from global search.

This protocol assumes a user has already opened one paper in the interface.
The QASPER paper id is used only to simulate that explicit user-selected scope;
it must never be reported as a global-search result.
"""

import json
import math
import os
import time
from collections import defaultdict
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
from tune_qasper_bge_m3_three_way import weighted_rrf_three_way


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_PATH = Path("data/qasper/results/paper_scoped_rag_validation.json")

SOURCE_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10
DENSE_WEIGHT = 2.0
SPARSE_WEIGHT = 1.0
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


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
    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    paper_rows = defaultdict(list)
    paper_indices = defaultdict(set)
    for index, item in enumerate(corpus):
        paper_rows[item["paper_id"]].append(item)
        paper_indices[item["paper_id"]].add(index)
    bm25_by_paper = {
        paper_id: BM25Okapi([tokenize(item["text"]) for item in rows])
        for paper_id, rows in paper_rows.items()
    }
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))

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

    metrics, latencies = [], []
    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        paper_id = item["paper_id"]
        scoped_rows = paper_rows[paper_id]
        start = time.perf_counter()
        dense_ids = [
            document.metadata["chunk_id"]
            for document in dense_store.similarity_search(
                question,
                k=min(SOURCE_CANDIDATES, len(scoped_rows)),
                filter={"paper_id": paper_id},
            )
        ]
        bm25_scores = bm25_by_paper[paper_id].get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:SOURCE_CANDIDATES]
        bm25_ids = [scoped_rows[i]["chunk_id"] for i in bm25_indices]
        sparse_indices = sparse_search(
            sparse_encoder.encode_query(question),
            inverted_index,
            SOURCE_CANDIDATES,
            allowed_document_indices=paper_indices[paper_id],
        )
        sparse_ids = [corpus[index]["chunk_id"] for index in sparse_indices]
        fused_ids = weighted_rrf_three_way(
            dense_ids,
            bm25_ids,
            sparse_ids,
            dense_weight=DENSE_WEIGHT,
            sparse_weight=SPARSE_WEIGHT,
        )[:RERANK_CANDIDATES]
        scores = score_in_small_batches(
            reranker, question, [corpus_by_id[chunk_id]["text"] for chunk_id in fused_ids]
        )
        reranked_ids = [
            chunk_id
            for chunk_id, _ in sorted(zip(fused_ids, scores), key=lambda pair: pair[1], reverse=True)[
                :FINAL_K
            ]
        ]
        metrics.append(metric_for_one_query(reranked_ids, item["gold_chunk_ids"]))
        latencies.append(time.perf_counter() - start)
        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "protocol": "paper_scoped_user_selected_document_not_global_search",
        "questions": len(eval_data),
        "parameters": {
            "dense_weight": DENSE_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "source_candidates": SOURCE_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
        },
        "paper_scoped_three_way_rerank": average_metrics(metrics),
    }
    report["paper_scoped_three_way_rerank"]["p95_end_to_end_latency_seconds"] = percentile(
        latencies
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n用户选定论文后的 Scoped RAG 结果：")
    print(json.dumps(report["paper_scoped_three_way_rerank"], ensure_ascii=False, indent=2))
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
