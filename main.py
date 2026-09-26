from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qasper_rag_service import answer_question, system_info

app = FastAPI(
    title="QASPER Hybrid RAG API",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="要向知识库提问的问题")


@app.get("/")
def root():
    return {
        "message": "QASPER Hybrid RAG API is running",
        "project": "qasper-hybrid-rag",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/system")
def system():
    return system_info()


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
