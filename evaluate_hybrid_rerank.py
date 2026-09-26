import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

# 让首次下载的重排模型也保存在项目 data 目录
os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

CORPUS_PATH = Path("data/qasper/corpus.jsonl")
EVAL_PATH = Path("data/qasper/eval.jsonl")
CHROMA_DIR = "data/qasper/chroma_dense"
COLLECTION_NAME = "qasper_dense_baseline"

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
RERANK_CANDIDATES = 20
FINAL_K = 10

RRF_K = 60
DENSE_WEIGHT = 0.25
PAIR_BATCH_SIZE = 2


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def weighted_rrf(dense_ids, bm25_ids):
    scores = defaultdict(float)

    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] += DENSE_WEIGHT / (RRF_K + rank)

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] += 1.0 / (RRF_K + rank)

    return [
        chunk_id
        for chunk_id, _ in sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]


def metric_for_one_query(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in gold_set
    ]

    def hit_at(k):
        return float(any(rank <= k for rank in ranks))

    def recall_at(k):
        return len(set(retrieved_ids[:k]) & gold_set) / len(gold_set)

    first_rank = min(ranks) if ranks else None
    mrr_at_10 = 1 / first_rank if first_rank and first_rank <= 10 else 0.0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank in ranks
        if rank <= 10
    )
    ideal_count = min(len(gold_set), 10)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr_at_10": mrr_at_10,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
    }


def average_metrics(metric_list):
    return {
        key: sum(item[key] for item in metric_list) / len(metric_list)
        for key in metric_list[0]
    }


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def score_in_small_batches(reranker, question, passages):
    """小批量重排，适配 6GB 显存。"""
    all_scores = []

    for start in range(0, len(passages), PAIR_BATCH_SIZE):
        batch_passages = passages[start:start + PAIR_BATCH_SIZE]
        pairs = [[question, passage] for passage in batch_passages]

        scores = reranker.compute_score(pairs)

        if isinstance(scores, (int, float)):
            scores = [scores]

        all_scores.extend(float(score) for score in scores)

    return all_scores


def main():
    corpus = read_jsonl(CORPUS_PATH)
    eval_data = read_jsonl(EVAL_PATH)

    corpus_by_id = {item["chunk_id"]: item for item in corpus}
    corpus_ids = [item["chunk_id"] for item in corpus]

    print(f"语料文本块：{len(corpus)}")
    print(f"评测问题：{len(eval_data)}")

    bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])

    # 必须与 build_qasper_index.py / evaluate_dense.py 保持一致
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        cache_folder="data/models",
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print(f"加载重排模型：{RERANK_MODEL}")
    reranker = FlagReranker(
        RERANK_MODEL,
        query_max_length=256,
        passage_max_length=512,
        use_fp16=True,
        devices=["cuda:0"],
    )

    rrf_metrics = []
    rerank_metrics = []
    rrf_latencies = []
    rerank_latencies = []
    details = []

    for index, item in enumerate(eval_data, start=1):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]

        total_start = time.perf_counter()

        dense_docs = vectorstore.similarity_search(
            question,
            k=DENSE_CANDIDATES,
        )
        dense_ids = [doc.metadata["chunk_id"] for doc in dense_docs]

        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[:BM25_CANDIDATES]
        bm25_ids = [corpus_ids[i] for i in bm25_indices]

        fused_ids = weighted_rrf(dense_ids, bm25_ids)
        rrf_top10 = fused_ids[:FINAL_K]
        rrf_elapsed = time.perf_counter() - total_start

        rerank_input_ids = fused_ids[:RERANK_CANDIDATES]
        passages = [corpus_by_id[chunk_id]["text"] for chunk_id in rerank_input_ids]
        rerank_scores = score_in_small_batches(reranker, question, passages)

        reranked_pairs = sorted(
            zip(rerank_input_ids, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_ids = [chunk_id for chunk_id, _ in reranked_pairs[:FINAL_K]]
        total_elapsed = time.perf_counter() - total_start

        rrf_metrics.append(metric_for_one_query(rrf_top10, gold_ids))
        rerank_metrics.append(metric_for_one_query(reranked_ids, gold_ids))
        rrf_latencies.append(rrf_elapsed)
        rerank_latencies.append(total_elapsed)

        details.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "gold_chunk_ids": gold_ids,
                "rrf_top10": rrf_top10,
                "reranked_top10": [
                    {
                        "chunk_id": chunk_id,
                        "reranker_score": score,
                        "title": corpus_by_id[chunk_id]["title"],
                        "section": corpus_by_id[chunk_id]["section"],
                    }
                    for chunk_id, score in reranked_pairs[:FINAL_K]
                ],
                "rrf_metrics": rrf_metrics[-1],
                "rerank_metrics": rerank_metrics[-1],
                "end_to_end_latency_seconds": total_elapsed,
            }
        )

        if index % 10 == 0 or index == len(eval_data):
            print(f"已完成：{index}/{len(eval_data)}")

    summary = {
        "parameters": {
            "dense_candidates": DENSE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
            "final_k": FINAL_K,
            "rrf_k": RRF_K,
            "dense_weight": DENSE_WEIGHT,
            "reranker": RERANK_MODEL,
            "pair_batch_size": PAIR_BATCH_SIZE,
        },
        "hybrid_rrf_dense_0.25": average_metrics(rrf_metrics),
        "hybrid_rrf_rerank_bge_v2_m3": average_metrics(rerank_metrics),
    }

    summary["hybrid_rrf_dense_0.25"]["p95_latency_seconds"] = percentile(
        rrf_latencies, 0.95
    )
    summary["hybrid_rrf_rerank_bge_v2_m3"]["p95_end_to_end_latency_seconds"] = percentile(
        rerank_latencies, 0.95
    )

    result_dir = Path("data/qasper/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "hybrid_rerank_bge_v2_m3.json"
    details_path = result_dir / "hybrid_rerank_bge_v2_m3_details.jsonl"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nRRF 基线：")
    print(json.dumps(summary["hybrid_rrf_dense_0.25"], ensure_ascii=False, indent=2))

    print("\nRRF + Reranker：")
    print(json.dumps(
        summary["hybrid_rrf_rerank_bge_v2_m3"],
        ensure_ascii=False,
        indent=2,
    ))

    print(f"\n汇总结果：{summary_path}")
    print(f"逐题明细：{details_path}")


if __name__ == "__main__":
    main()