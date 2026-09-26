import json
import math
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from bge_m3_embeddings import BGEM3DenseEmbeddings


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RESULT_PATH = Path("data/qasper/results/bge_m3_hybrid_rrf_tuning.json")

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
FINAL_K = 10
RRF_K = 60
DENSE_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def weighted_rrf(dense_ids: list[str], bm25_ids: list[str], dense_weight: float) -> list[str]:
    scores = defaultdict(float)
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] += dense_weight / (RRF_K + rank)
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] += 1.0 / (RRF_K + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def metric_for_one_query(retrieved_ids: list[str], gold_ids: list[str]) -> dict:
    gold = set(gold_ids)
    ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in gold
    ]

    def hit_at(k: int) -> float:
        return float(any(rank <= k for rank in ranks))

    def recall_at(k: int) -> float:
        return len(set(retrieved_ids[:k]) & gold) / len(gold)

    first_rank = min(ranks) if ranks else None
    mrr_at_10 = 1 / first_rank if first_rank and first_rank <= 10 else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal_count = min(len(gold), 10)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr_at_10": mrr_at_10,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
    }


def average_metrics(metric_list: list[dict]) -> dict:
    return {
        name: sum(item[name] for item in metric_list) / len(metric_list)
        for name in metric_list[0]
    }


def main():
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)
    corpus_ids = [item["chunk_id"] for item in corpus]

    print(f"加载语料：{len(corpus)} 个文本块")
    print(f"加载评测问题：{len(eval_data)} 条")

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    systems = {"bge_m3_dense_top10": [], "bm25_top10": []}
    for weight in DENSE_WEIGHTS:
        systems[f"bge_m3_weighted_rrf_dense_{weight}"] = []

    dense_candidate_recalls = []
    bm25_candidate_recalls = []
    union_candidate_recalls = []
    bm25_unique_gold_rates = []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]
        gold = set(gold_ids)

        dense_docs = vector_store.similarity_search(question, k=DENSE_CANDIDATES)
        dense_ids = [document.metadata["chunk_id"] for document in dense_docs]

        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        systems["bge_m3_dense_top10"].append(metric_for_one_query(dense_ids[:FINAL_K], gold_ids))
        systems["bm25_top10"].append(metric_for_one_query(bm25_ids[:FINAL_K], gold_ids))
        for weight in DENSE_WEIGHTS:
            fused_ids = weighted_rrf(dense_ids, bm25_ids, weight)
            systems[f"bge_m3_weighted_rrf_dense_{weight}"].append(
                metric_for_one_query(fused_ids[:FINAL_K], gold_ids)
            )

        dense_set, bm25_set = set(dense_ids), set(bm25_ids)
        dense_candidate_recalls.append(len(dense_set & gold) / len(gold))
        bm25_candidate_recalls.append(len(bm25_set & gold) / len(gold))
        union_candidate_recalls.append(len((dense_set | bm25_set) & gold) / len(gold))
        bm25_unique_gold_rates.append(len((bm25_set - dense_set) & gold) / len(gold))

        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    report = {
        "candidate_diagnostics": {
            "bge_m3_dense_candidate_recall_at_30": sum(dense_candidate_recalls) / len(dense_candidate_recalls),
            "bm25_candidate_recall_at_30": sum(bm25_candidate_recalls) / len(bm25_candidate_recalls),
            "union_candidate_recall_at_up_to_60": sum(union_candidate_recalls) / len(union_candidate_recalls),
            "bm25_unique_gold_evidence_rate": sum(bm25_unique_gold_rates) / len(bm25_unique_gold_rates),
        },
        "systems": {name: average_metrics(values) for name, values in systems.items()},
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n候选集诊断：")
    for name, value in report["candidate_diagnostics"].items():
        print(f"{name}: {value:.4f}")

    print("\n排序方案对比：")
    for name, metrics in report["systems"].items():
        print(
            f"{name:36} "
            f"Hit@5={metrics['hit_at_5']:.4f} "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@10={metrics['ndcg_at_10']:.4f}"
        )

    print(f"\n完整结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    main()
