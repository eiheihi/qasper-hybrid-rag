import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bge_m3_embeddings import BGEM3DenseEmbeddings
from tune_qasper_bge_m3_hybrid import read_jsonl


PROFILE_PATH = Path("data/qasper/paper_profiles.jsonl")
CHROMA_DIR = Path("data/qasper/chroma_paper_router")
COLLECTION_NAME = "qasper_paper_router"


def main():
    profiles = read_jsonl(PROFILE_PATH)
    if not profiles:
        raise RuntimeError("未找到论文 profile，请先运行 prepare_qasper_paper_profiles.py")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    documents = [
        Document(
            page_content=profile["profile_text"],
            metadata={"paper_id": profile["paper_id"], "title": profile["title"]},
        )
        for profile in profiles
    ]
    print(f"构建论文级路由索引：{len(documents)} 篇")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=BGEM3DenseEmbeddings(batch_size=4, max_length=512),
    )
    store.add_documents(documents, ids=[item.metadata["paper_id"] for item in documents])
    print(f"论文级路由索引已完成：{CHROMA_DIR}")


if __name__ == "__main__":
    main()
