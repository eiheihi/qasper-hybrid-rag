"""Validation-only candidate-recall diagnosis for contextual dense retrieval."""

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
CHROMA_DIR = "data/qasper/chroma_bge_m3_contextual"
COLLECTION_NAME = "qasper_bge_m3_contextual"
RESULT_PATH = Path("data/qasper/results/contextual_candidate_diagnostics.json")
CANDIDATES = 30


def recall(candidate_ids: list[str], gold_ids: list[str]) -> float:
    return len(set(candidate_ids) & set(gold_ids)) / len(gold_ids)


def main():
    if not SPARSE_INDEX_PATH.exists():
        raise RuntimeError("请先运行 build_qasper_bge_m3_sparse_index.py")
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]
    sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))
    if sparse_payload["corpus_size"] != len(corpus):
        raise RuntimeError("稀疏索引与语料不一致，请重新建索引")

    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
    contextual_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    sparse_encoder = BGEM3SparseEncoder(batch_size=4, max_length=512)
    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    inverted_index = sparse_payload["inverted_index"]

    contextual_recalls, sparse_recalls, bm25_recalls = [], [], []
    contextual_sparse_unions, contextual_bm25_unions, three_way_unions = [], [], []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        contextual_ids = [
            document.metadata["chunk_id"]
            for document in contextual_store.similarity_search(question, k=CANDIDATES)
        ]
        sparse_indices = sparse_search(
            sparse_encoder.encode_query(question), inverted_index, CANDIDATES
        )
        sparse_ids = [corpus_ids[i] for i in sparse_indices]
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]
        gold_ids = item["gold_chunk_ids"]

        contextual_recalls.append(recall(contextual_ids, gold_ids))
        sparse_recalls.append(recall(sparse_ids, gold_ids))
        bm25_recalls.append(recall(bm25_ids, gold_ids))
        contextual_sparse_unions.append(recall(contextual_ids + sparse_ids, gold_ids))
        contextual_bm25_unions.append(recall(contextual_ids + bm25_ids, gold_ids))
        three_way_unions.append(recall(contextual_ids + sparse_ids + bm25_ids, gold_ids))

        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "candidates_per_source": CANDIDATES,
        "contextual_dense_candidate_recall_at_30": sum(contextual_recalls)
        / len(contextual_recalls),
        "bge_m3_sparse_candidate_recall_at_30": sum(sparse_recalls) / len(sparse_recalls),
        "bm25_candidate_recall_at_30": sum(bm25_recalls) / len(bm25_recalls),
        "contextual_dense_sparse_union_recall_at_up_to_60": sum(contextual_sparse_unions)
        / len(contextual_sparse_unions),
        "contextual_dense_bm25_union_recall_at_up_to_60": sum(contextual_bm25_unions)
        / len(contextual_bm25_unions),
        "three_way_union_recall_at_up_to_90": sum(three_way_unions) / len(three_way_unions),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n上下文表示候选集诊断：")
    for name, value in report.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
