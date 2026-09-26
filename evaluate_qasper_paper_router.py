"""Measure document-routing recall before building a hierarchical retriever."""

import json
from pathlib import Path

from langchain_chroma import Chroma

from bge_m3_embeddings import BGEM3DenseEmbeddings
from tune_qasper_bge_m3_hybrid import read_jsonl


EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_paper_router"
COLLECTION_NAME = "qasper_paper_router"
RESULT_PATH = Path("data/qasper/results/paper_router_diagnostics.json")
K_VALUES = [1, 3, 5, 10]


def main():
    eval_data = read_jsonl(EVAL_PATH)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
    )
    hits = {k: 0 for k in K_VALUES}
    max_k = max(K_VALUES)

    for index, item in enumerate(eval_data, start=1):
        docs = store.similarity_search(item["question"], k=max_k)
        routed_paper_ids = [document.metadata["paper_id"] for document in docs]
        for k in K_VALUES:
            hits[k] += int(item["paper_id"] in routed_paper_ids[:k])
        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "split": "validation",
        "papers": 100,
        "questions": len(eval_data),
        **{f"paper_route_recall_at_{k}": hits[k] / len(eval_data) for k in K_VALUES},
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n论文路由诊断：")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
