"""Production-facing global QASPER retrieval and answer service.

This module uses the frozen configuration that was validated independently:
BGE-M3 dense + BGE-M3 native sparse + BM25 -> weighted RRF -> BGE reranker.
"""

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from bge_m3_embeddings import BGEM3DenseEmbeddings
from bge_m3_sparse import BGEM3SparseEncoder, sparse_search
from tune_qasper_bge_m3_hybrid import read_jsonl, tokenize


load_dotenv()

CORPUS_PATH = Path("data/qasper/corpus.jsonl")
SPARSE_INDEX_PATH = Path("data/qasper/bge_m3_sparse_index.json")
CHROMA_DIR = "data/qasper/chroma_bge_m3_dense"
COLLECTION_NAME = "qasper_bge_m3_dense"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
SPARSE_CANDIDATES = 30
RERANK_CANDIDATES = 20
ANSWER_CONTEXT_K = 5
DENSE_WEIGHT = 2.0
SPARSE_WEIGHT = 1.0
RRF_K = 60
PAIR_BATCH_SIZE = 2


def weighted_rrf_three_way(
    dense_ids: list[str], bm25_ids: list[str], sparse_ids: list[str]
) -> list[str]:
    scores = defaultdict(float)
    for candidate_ids, weight in (
        (dense_ids, DENSE_WEIGHT),
        (bm25_ids, 1.0),
        (sparse_ids, SPARSE_WEIGHT),
    ):
        for rank, chunk_id in enumerate(candidate_ids, start=1):
            scores[chunk_id] += weight / (RRF_K + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]


def score_in_small_batches(reranker, question: str, passages: list[str]) -> list[float]:
    scores = []
    for start in range(0, len(passages), PAIR_BATCH_SIZE):
        pairs = [[question, passage] for passage in passages[start : start + PAIR_BATCH_SIZE]]
        batch_scores = reranker.compute_score(pairs)
        if isinstance(batch_scores, (int, float)):
            batch_scores = [batch_scores]
        scores.extend(float(score) for score in batch_scores)
    return scores


class QASPERHybridRetriever:
    def __init__(self):
        if not CORPUS_PATH.exists() or not SPARSE_INDEX_PATH.exists():
            raise RuntimeError(
                "QASPER 索引不完整。请先运行 prepare_qasper.py、"
                "build_qasper_bge_m3_index.py 和 build_qasper_bge_m3_sparse_index.py"
            )

        self.corpus = read_jsonl(CORPUS_PATH)
        self.corpus_ids = [item["chunk_id"] for item in self.corpus]
        self.corpus_by_id = {item["chunk_id"]: item for item in self.corpus}
        self.bm25 = BM25Okapi([tokenize(item["text"]) for item in self.corpus])
        sparse_payload = json.loads(SPARSE_INDEX_PATH.read_text(encoding="utf-8"))
        if sparse_payload["corpus_size"] != len(self.corpus):
            raise RuntimeError("稀疏索引与语料不一致，请重新运行 build_qasper_bge_m3_sparse_index.py")
        self.inverted_index = sparse_payload["inverted_index"]

        self.dense_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
        )
        self.sparse_encoder = BGEM3SparseEncoder(batch_size=4, max_length=512)
        self.reranker = FlagReranker(
            RERANK_MODEL,
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:0"],
            cache_dir="data/models",
        )

    def retrieve(self, question: str) -> list[dict]:
        dense_ids = [
            document.metadata["chunk_id"]
            for document in self.dense_store.similarity_search(question, k=DENSE_CANDIDATES)
        ]
        bm25_scores = self.bm25.get_scores(tokenize(question))
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:BM25_CANDIDATES]
        bm25_ids = [self.corpus_ids[index] for index in bm25_indices]
        sparse_indices = sparse_search(
            self.sparse_encoder.encode_query(question),
            self.inverted_index,
            SPARSE_CANDIDATES,
        )
        sparse_ids = [self.corpus_ids[index] for index in sparse_indices]
        candidate_ids = weighted_rrf_three_way(dense_ids, bm25_ids, sparse_ids)[
            :RERANK_CANDIDATES
        ]
        scores = score_in_small_batches(
            self.reranker,
            question,
            [self.corpus_by_id[chunk_id]["text"] for chunk_id in candidate_ids],
        )
        ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [
            {**self.corpus_by_id[chunk_id], "reranker_score": score}
            for chunk_id, score in ranked[:ANSWER_CONTEXT_K]
        ]


@lru_cache
def get_retriever() -> QASPERHybridRetriever:
    return QASPERHybridRetriever()


@lru_cache
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")
    return ChatOpenAI(
        model="deepseek-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )


def answer_question(question: str) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    documents = get_retriever().retrieve(question)
    if not documents:
        raise RuntimeError("没有检索到可用证据")

    context_parts = []
    sources = []
    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"[来源 {index} | 论文：{document['title']} | 章节：{document['section']}]\n"
            f"{document['text']}"
        )
        sources.append(
            {
                "id": index,
                "paper_id": document["paper_id"],
                "title": document["title"],
                "section": document["section"],
                "reranker_score": round(document["reranker_score"], 4),
                "snippet": " ".join(document["text"].split())[:280],
            }
        )

    prompt = f"""You are a careful scientific-paper question answering assistant.
Answer only from the retrieved evidence. If the evidence is insufficient, say so
explicitly instead of guessing. Cite each important claim as [Source 1], [Source 2], etc.

Question:
{question}

Evidence:
{chr(10).join(context_parts)}
"""
    response = get_llm().invoke(prompt)
    return {"answer": response.content, "sources": sources}


def system_info() -> dict:
    return {
        "mode": "global_qasper_search",
        "pipeline": "BGE-M3 dense + BGE-M3 sparse + BM25 -> weighted RRF -> BGE reranker",
        "source_candidates": {
            "dense": DENSE_CANDIDATES,
            "sparse": SPARSE_CANDIDATES,
            "bm25": BM25_CANDIDATES,
        },
        "rerank_candidates": RERANK_CANDIDATES,
        "answer_context_chunks": ANSWER_CONTEXT_K,
    }
