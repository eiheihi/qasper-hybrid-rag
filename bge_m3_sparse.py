"""BGE-M3 native sparse retrieval utilities.

The inverted index is kept separate from Chroma's dense index so dense, BM25,
and BGE-M3 sparse retrieval can be evaluated as independent candidate sources.
"""

from collections import defaultdict
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel


class BGEM3SparseEncoder:
    def __init__(
        self,
        cache_dir: str = "data/models",
        batch_size: int = 4,
        max_length: int = 512,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            devices="cuda:0",
            cache_dir=str(Path(cache_dir)),
            batch_size=batch_size,
            query_max_length=max_length,
            passage_max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode(self, texts: list[str]) -> list[dict[str, float]]:
        result = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return result["lexical_weights"]

    def encode_query(self, text: str) -> dict[str, float]:
        return self.encode([text])[0]


def build_inverted_index(
    lexical_weights: list[dict[str, float]],
) -> dict[str, list[list[float]]]:
    """Map BGE token ids to document indices and lexical weights."""
    inverted_index = defaultdict(list)
    for document_index, weights in enumerate(lexical_weights):
        for token_id, weight in weights.items():
            inverted_index[token_id].append([document_index, float(weight)])
    return dict(inverted_index)


def sparse_search(
    query_weights: dict[str, float],
    inverted_index: dict[str, list[list[float]]],
    k: int,
    allowed_document_indices: set[int] | None = None,
) -> list[int]:
    scores = defaultdict(float)
    for token_id, query_weight in query_weights.items():
        for document_index, document_weight in inverted_index.get(token_id, []):
            if (
                allowed_document_indices is not None
                and document_index not in allowed_document_indices
            ):
                continue
            scores[document_index] += float(query_weight) * float(document_weight)
    return [
        document_index
        for document_index, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]
    ]
