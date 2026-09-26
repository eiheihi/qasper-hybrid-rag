from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = Path("data/uploads")
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index() -> None:
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("data/uploads 中没有 PDF，请先通过 /documents 上传资料")

    documents = []
    for pdf_file in pdf_files:
        pages = PyPDFLoader(str(pdf_file)).load()

        for page in pages:
            page.metadata["source_file"] = pdf_file.name

        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        cache_folder="data/models",
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"已读取 {len(pdf_files)} 份 PDF")
    print(f"已生成 {len(chunks)} 个文本块")
    print(f"向量库已保存至 {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()