import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

CORPUS_PATH = Path("data/qasper/corpus.jsonl")
CHROMA_DIR = Path("data/qasper/chroma_dense")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def load_corpus() -> list[Document]:
    documents = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            documents.append(
                Document(
                    page_content=record["text"],
                    metadata={
                        "chunk_id": record["chunk_id"],
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "section": record["section"],
                    },
                )
            )

    return documents


def main():
    documents = load_corpus()
    if not documents:
        raise RuntimeError("语料为空，请先运行 prepare_qasper.py")

    # 只删除可重新生成的 QASPER Dense 向量库，避免重复写入。
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        cache_folder="data/models",
    )

    vector_store = Chroma(
        collection_name="qasper_dense_baseline",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(documents), BATCH_SIZE),
        start=1,
    ):
        batch = documents[start:start + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )

        print(f"已写入第 {batch_number}/{total_batches} 批")

    print(f"\nDense 向量库已完成：{len(documents)} 个文本块")
    print(f"保存位置：{CHROMA_DIR}")


if __name__ == "__main__":
    main()