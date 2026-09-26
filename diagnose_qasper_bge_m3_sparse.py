"""Validation-only candidate recall diagnosis for BGE-M3 native sparse retrieval."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from bge_m3_embeddings import BGEM3DenseEmbeddings
from bge_m3_sparse import BGEM3SparseEncoder, sparse_search
from tune_qasper_bge_m3_hybrid import read_jsonl, tokenize


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_PATH = Path("data/qasper/results/bge_m3_sparse_candidate_diagnostics.json")
CANDIDATES = 30


def recall_at_candidates(candidate_ids: list[str], gold_ids: list[str]) -> float:
    return len(set(candidate_ids) & set(gold_ids)) / len(gold_ids)


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError(
            "未找到原生稀疏索引，请先运行 python build_qasper_bge_m3_sparse_index.py"
        )

    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))
    if payload["corpus_size"] != len(corpus):
        raise RuntimeError("稀疏索引与当前语料大小不一致，请重新建索引")
    inverted_index = payload["inverted_index"]

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    dense_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
    )
    sparse_encoder = BGEM3SparseEncoder(batch_size=4, max_length=512)

    dense_recalls, bm25_recalls, sparse_recalls = [], [], []
    dense_bm25_unions, dense_sparse_unions, all_unions = [], [], []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        dense_ids = [
            document.metadata["chunk_id"]
            for document in dense_store.similarity_search(question, k=CANDIDATES)
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

        gold_ids = item["gold_chunk_ids"]
        dense_recalls.append(recall_at_candidates(dense_ids, gold_ids))
        bm25_recalls.append(recall_at_candidates(bm25_ids, gold_ids))
        sparse_recalls.append(recall_at_candidates(sparse_ids, gold_ids))
        dense_bm25_unions.append(recall_at_candidates(dense_ids + bm25_ids, gold_ids))
        dense_sparse_unions.append(recall_at_candidates(dense_ids + sparse_ids, gold_ids))
        all_unions.append(recall_at_candidates(dense_ids + bm25_ids + sparse_ids, gold_ids))

        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "candidates_per_source": CANDIDATES,
        "dense_candidate_recall_at_30": sum(dense_recalls) / len(dense_recalls),
        "bm25_candidate_recall_at_30": sum(bm25_recalls) / len(bm25_recalls),
        "bge_m3_sparse_candidate_recall_at_30": sum(sparse_recalls) / len(sparse_recalls),
        "dense_bm25_union_recall_at_up_to_60": sum(dense_bm25_unions) / len(dense_bm25_unions),
        "dense_sparse_union_recall_at_up_to_60": sum(dense_sparse_unions) / len(dense_sparse_unions),
        "three_way_union_recall_at_up_to_90": sum(all_unions) / len(all_unions),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n候选集诊断：")
    for name, value in report.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
