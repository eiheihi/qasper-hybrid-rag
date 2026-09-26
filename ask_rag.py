import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    cache_folder="data/models",
)

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model="deepseek-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

question = input("请输入问题：").strip()
if not question:
    raise RuntimeError("问题不能为空")

documents = vector_store.similarity_search(question, k=4)

context_parts = []
for index, document in enumerate(documents, start=1):
    source_file = document.metadata.get("source_file", "未知文件")
    page = document.metadata.get("page", 0) + 1

    context_parts.append(
        f"[来源 {index} | 文件：{source_file} | 第 {page} 页]\n"
        f"{document.page_content}"
    )

context = "\n\n".join(context_parts)

prompt = f"""你是严谨的遥感研究助手。
请仅依据给出的资料回答问题。资料不足时，明确回答“提供的资料中未找到相关依据”，不要自行补充。
回答中的关键结论必须用 [来源 1]、[来源 2] 这类形式标注依据。

问题：
{question}

资料：
{context}
"""

response = llm.invoke(prompt)

print("\n回答：")
print(response.content)

print("\n检索到的来源：")
for index, document in enumerate(documents, start=1):
    source_file = document.metadata.get("source_file", "未知文件")
    page = document.metadata.get("page", 0) + 1
    print(f"[来源 {index}] {source_file}，第 {page} 页")