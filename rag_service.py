import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        cache_folder="data/models",
    )


@lru_cache
def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


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

    documents = get_vector_store().similarity_search(question, k=4)
    if not documents:
        raise RuntimeError("向量库中没有可检索的资料")

    context_parts = []
    sources = []

    for index, document in enumerate(documents, start=1):
        source_file = document.metadata.get("source_file", "未知文件")
        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"[来源 {index} | 文件：{source_file} | 第 {page} 页]\n"
            f"{document.page_content}"
        )
        sources.append(
            {
                "id": index,
                "file": source_file,
                "page": page,
                "snippet": " ".join(document.page_content.split())[:240],
            }
        )

    prompt = f"""你是严谨的遥感研究助手。
请仅依据给出的资料回答问题。资料不足时，明确回答“提供的资料中未找到相关依据”，不要自行补充。
回答中的关键结论必须用 [来源 1]、[来源 2] 这类形式标注依据。

问题：
{question}

资料：
{chr(10).join(context_parts)}
"""

    response = get_llm().invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }