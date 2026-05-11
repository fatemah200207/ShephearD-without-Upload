import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag_service import (
    PDF_FOLDER,
    create_vector_db,
    list_uploaded_pdfs,
    load_qa_chain,
    smart_answer,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Shepheard Hotel AI Chatbot")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

qa_chain = None


class Query(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    """Automatically index the PDFs already included inside app/data/pdfs."""
    global qa_chain
    os.makedirs(PDF_FOLDER, exist_ok=True)

    if list_uploaded_pdfs():
        create_vector_db()
        qa_chain = load_qa_chain()


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "implemented_files",
        "pdf_count": len(list_uploaded_pdfs()),
        "files": list_uploaded_pdfs(),
    }


@app.get("/files")
def files():
    return {"files": list_uploaded_pdfs()}


@app.post("/reindex")
def reindex():
    """Optional endpoint to rebuild the database after changing project PDFs."""
    global qa_chain

    if not list_uploaded_pdfs():
        return {
            "status": "error",
            "message": "No implemented PDF files found in app/data/pdfs.",
            "files": [],
        }

    message = create_vector_db()
    qa_chain = load_qa_chain()

    return {
        "status": "success",
        "message": message,
        "files": list_uploaded_pdfs(),
    }


@app.post("/chat")
def chat(query: Query):
    global qa_chain

    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not list_uploaded_pdfs():
        return {
            "question": question,
            "answer": "No PDF knowledge files are included yet. Add your PDFs inside app/data/pdfs, push to GitHub, then redeploy.",
        }

    if qa_chain is None:
        create_vector_db()
        qa_chain = load_qa_chain()

    answer = smart_answer(question, qa_chain)

    return {
        "question": question,
        "answer": answer,
    }
