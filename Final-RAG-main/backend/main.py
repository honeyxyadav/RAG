"""
RAG (Retrieval-Augmented Generation) Backend
=============================================
FastAPI server that powers the PDF chatbot.
Handles PDF upload, embedding, and question answering.
"""

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG PDF Chatbot",
    description="Chat with your PDF using Ollama (local) or Groq API (fast inference)",
    version="2.0.0"
)

# Allow frontend (HTML file opened in browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ── Global RAG instance ────────────────────────────────────────────────────────
# One shared pipeline instance — holds the FAISS index in memory
# Using "groq" as default provider (requires GROQ_API_KEY in .env file)
rag = RAGPipeline(vector_store_path=VECTOR_DIR, provider="groq")


# ── Request / Response Models ──────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    chat_history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]
    provider: str = "groq"        # "ollama" for local, "groq" for Groq API


class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list[str]        # Relevant PDF snippets used to answer
    provider: str                   # Which provider was used (ollama or groq)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "running", "message": "RAG PDF Chatbot API is live"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Step 1-5: Upload PDF → Extract text → Split → Embed → Store in FAISS.

    Accepts a PDF file, processes it through the full RAG ingestion pipeline,
    and returns a success message with chunk count.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run the ingestion pipeline (steps 2-5)
        chunk_count = rag.ingest_pdf(save_path)
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunk_count,
            "message": f"PDF processed! {chunk_count} chunks stored in vector DB."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Steps 7-9: Retrieve relevant chunks → Build prompt → Ask LLM → Return answer.

    Takes a question + chat history + provider, retrieves top-k relevant PDF chunks,
    sends them as context to the selected LLM (Ollama or Grok), and returns the answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not rag.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No PDF loaded yet. Please upload a PDF first."
        )

    # Validate provider
    if request.provider not in ["ollama", "groq"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Use 'ollama' or 'groq'."
        )

    try:
        # Switch provider if needed
        rag.set_provider(request.provider)
        
        answer, source_chunks = rag.answer_question(
            question=request.question,
            chat_history=request.chat_history
        )
        return AnswerResponse(answer=answer, source_chunks=source_chunks, provider=request.provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")


@app.delete("/reset")
def reset():
    """Clear the vector store and start fresh (useful for uploading a new PDF)."""
    rag.reset()
    return {"status": "success", "message": "Vector store cleared. Upload a new PDF to begin."}


@app.get("/provider")
def get_provider():
    """Get the current LLM provider."""
    return {"provider": rag.provider}


@app.post("/provider")
def set_provider(provider: str):
    """Set the LLM provider (ollama or groq)."""
    if provider not in ["ollama", "groq"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Use 'ollama' or 'groq'."
        )
    rag.set_provider(provider)
    return {"status": "success", "provider": provider, "message": f"Provider switched to {provider}"}
