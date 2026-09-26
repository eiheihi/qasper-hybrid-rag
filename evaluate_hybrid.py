import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# 与 build_qasper_index.py 保持一致
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_dense"
COLLECTION_NAME = "qasper_dense_baseline"
MODEL_CACHE = "data/models"

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
FINAL_K = 10
RRF_K = 60


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str):
    """保留英文单词、数字和连字符术语，适合 QASPER 的英文论文语料。"""
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def rrf_fusion(dense_ids, bm25_ids):
    """Reciprocal Rank Fusion：合并两个排序列表。"""
    scores = defaultdict(float)

    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] += 1 / (RRF_K + rank)

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] += 1 / (RRF_K + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def query_metrics(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    ranks = [rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in gold_set]

    def hit_at(k):
        return float(any(rank <= k for rank in ranks))

    def recall_at(k):
        return len(set(retrieved_ids[:k]) & gold_set) / len(gold_set)

    mrr_at_10 = 0.0
    first_rank = min(ranks) if ranks else None
    if first_rank is not None and first_rank <= 10:
        mrr_at_10 = 1 / first_rank

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank in ranks
        if rank <= 10
    )
    ideal_count = min(len(gold_set), 10)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg_at_10 = dcg / idcg if idcg else 0.0

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr_at_10": mrr_at_10,
        "ndcg_at_10": ndcg_at_10,
    }


def main():
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)

    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    corpus_ids = [item["chunk_id"] for item in corpus]
    tokenized_corpus = [tokenize(item["text"]) for item in corpus]

    print(f"加载语料：{len(corpus)} 个文本块")
    print(f"加载评测问题：{len(eval_data)} 条")

    print("构建 BM25 索引...")
    bm25 = BM25Okapi(tokenized_corpus)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        cache_folder=MODEL_CACHE,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    all_metrics = []
    latencies = []
    details = []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]

        start = time.perf_counter()

        # 1. Dense 候选
        dense_docs = vectorstore.similarity_search(
            question,
            k=DENSE_CANDIDATES,
        )
        dense_ids = [doc.metadata["chunk_id"] for doc in dense_docs]

        # 2. BM25 候选
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        # 3. RRF 融合
        fused = rrf_fusion(dense_ids, bm25_ids)
        retrieved_ids = [chunk_id for chunk_id, _ in fused[:FINAL_K]]

        elapsed = time.perf_counter() - start
        metrics = query_metrics(retrieved_ids, gold_ids)

        all_metrics.append(metrics)
        latencies.append(elapsed)

        details.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "gold_chunk_ids": gold_ids,
                "dense_candidate_ids": dense_ids,
                "bm25_candidate_ids": bm25_ids,
                "retrieved": [
                    {
                        "chunk_id": chunk_id,
                        "rrf_score": score,
                        "title": corpus_by_id[chunk_id]["title"],
                        "section": corpus_by_id[chunk_id]["section"],
                    }
                    for chunk_id, score in fused[:FINAL_K]
                ],
                "metrics": metrics,
                "retrieval_latency_seconds": elapsed,
            }
        )

        if index % 25 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    summary = {
        "system": "hybrid_bm25_dense_rrf",
        "questions": len(eval_data),
        "parameters": {
            "dense_candidates": DENSE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "final_k": FINAL_K,
            "rrf_k": RRF_K,
        },
    }

    for metric_name in all_metrics[0]:
        summary[metric_name] = sum(x[metric_name] for x in all_metrics) / len(all_metrics)

    summary["p95_retrieval_latency_seconds"] = percentile(latencies, 0.95)

    result_dir = Path("data/qasper/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "hybrid_bm25_dense_rrf.json"
    details_path = result_dir / "hybrid_bm25_dense_rrf_details.jsonl"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nHybrid 检索结果：")
    for key, value in summary.items():
        if key != "parameters":
            print(f"{key}: {value}")

    print(f"\n汇总结果：{summary_path}")
    print(f"逐题明细：{details_path}")


if __name__ == "__main__":
    main()