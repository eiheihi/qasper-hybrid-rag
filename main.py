from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader

from rag_service import answer_question

app = FastAPI(
    title="Remote Sensing RAG API",
    version="0.2.0",
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="要向知识库提问的问题")


@app.get("/")
def root():
    return {
        "message": "RAG project is running",
        "project": "remote-sensing-rag",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    file_path = UPLOAD_DIR / Path(filename).name
    file_path.write_bytes(await file.read())

    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"无法读取该 PDF：{error}",
        ) from error

    return {
        "filename": filename,
        "pages": len(reader.pages),
        "text_characters": len(text),
        "text_preview": text[:500],
        "notice": "上传成功后，请运行 build_index.py 更新向量库",
    }


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error