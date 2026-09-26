import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bge_m3_embeddings import BGEM3DenseEmbeddings
from contextual_passages import contextual_text_by_chunk_id
from tune_qasper_bge_m3_hybrid import read_jsonl


CORPUS_PATH = Path("data/qasper/corpus.jsonl")
CHROMA_DIR = Path("data/qasper/chroma_bge_m3_contextual")
COLLECTION_NAME = "qasper_bge_m3_contextual"
CHROMA_BATCH_SIZE = 64


def main():
    corpus = read_jsonl(CORPUS_PATH)
    if not corpus:
        raise RuntimeError("语料为空，请先运行 prepare_qasper.py")

    contextual_texts = contextual_text_by_chunk_id(corpus, window=1)
    documents = [
        Document(
            page_content=contextual_texts[item["chunk_id"]],
            metadata={
                "chunk_id": item["chunk_id"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "section": item["section"],
            },
        )
        for item in corpus
    ]

    # Only this regenerated experimental index is removed.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print(f"构建上下文增强 BGE-M3 索引：{len(documents)} 个原始段落")
    embeddings = BGEM3DenseEmbeddings(batch_size=4, max_length=512)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    total_batches = (len(documents) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE
    for batch_number, start in enumerate(
        range(0, len(documents), CHROMA_BATCH_SIZE), start=1
    ):
        batch = documents[start : start + CHROMA_BATCH_SIZE]
        vector_store.add_documents(
            documents=batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )
        print(f"已写入第 {batch_number}/{total_batches} 批")

    print(f"\n上下文增强向量库已完成：{len(documents)} 个文本块")
    print(f"保存位置：{CHROMA_DIR}")


if __name__ == "__main__":
    main()
