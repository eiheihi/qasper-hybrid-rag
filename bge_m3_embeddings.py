from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings


class BGEM3DenseEmbeddings(Embeddings):
    """LangChain adapter for BGE-M3 dense vectors only.

    Sparse and multi-vector features are intentionally disabled here so this
    experiment isolates the impact of replacing the MiniLM dense encoder.
    """

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
            normalize_embeddings=True,
            use_fp16=True,
            devices="cuda:0",
            cache_dir=str(Path(cache_dir)),
            batch_size=batch_size,
            query_max_length=max_length,
            passage_max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        result = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
