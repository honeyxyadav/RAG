# ⬡ RAG PDF Chatbot — Powered by Ollama

> Chat with any PDF document using a fully **local** AI stack. No cloud. No API keys. No data leaving your machine.

---

## 📖 Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [How Ollama Works](#2-how-ollama-works)
3. [Architecture & Flow](#3-architecture--flow)
4. [Project Structure](#4-project-structure)
5. [Setup Instructions](#5-setup-instructions)
6. [Running the App](#6-running-the-app)
7. [Example Queries](#7-example-queries)
8. [Troubleshooting](#8-troubleshooting)
9. [Interview Concepts](#9-interview-concepts)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that makes LLMs answer questions based on *your own data* rather than just their training knowledge.

### The Problem RAG Solves

A standard LLM like GPT-4 or LLaMA:
- Only knows what it was trained on (knowledge cutoff)
- Cannot access your private documents
- May hallucinate facts it doesn't know

### How RAG Fixes This

Instead of relying solely on the LLM's memory, RAG:

```
1. Retrieves relevant snippets from YOUR document (using vector similarity search)
2. Augments the LLM prompt with those snippets as context
3. Generates an answer grounded in the actual document text
```

### RAG vs Fine-Tuning

| Aspect          | RAG                          | Fine-Tuning                    |
|-----------------|------------------------------|--------------------------------|
| Cost            | Low — no GPU training        | High — GPU hours needed        |
| Update speed    | Instant (re-embed new docs)  | Slow (retrain model)           |
| Accuracy        | High for specific docs       | Good for style/domain shift    |
| Hallucination   | Low (grounded in context)    | Still possible                 |
| Use case        | Q&A over documents           | Domain-specific behavior       |

---

## 2. How Ollama Works

**Ollama** lets you run LLMs (like LLaMA 3, Mistral, Gemma) directly on your laptop.

```
┌─────────────────────────────────────────────────┐
│                    Your Machine                  │
│                                                  │
│  ┌──────────────┐        ┌───────────────────┐  │
│  │  Your App    │──HTTP──▶  Ollama Server    │  │
│  │  (FastAPI)   │◀───────│  localhost:11434  │  │
│  └──────────────┘        │                   │  │
│                           │  ┌─────────────┐ │  │
│                           │  │  llama3     │ │  │
│                           │  │  (weights   │ │  │
│                           │  │  in GGUF)   │ │  │
│                           │  └─────────────┘ │  │
│                           └───────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Key Ollama Commands

```bash
ollama serve              # Start the Ollama server
ollama pull llama3        # Download LLaMA 3 model (~4.7GB)
ollama pull mistral       # Alternative: Mistral 7B (~4.1GB)
ollama list               # See downloaded models
ollama run llama3         # Test in terminal
```

Ollama exposes an OpenAI-compatible REST API at `http://localhost:11434`.
LangChain's `Ollama` class calls this API automatically.

---

## 3. Architecture & Flow

### Ingestion Pipeline (Upload Phase)

```
PDF File
   │
   ▼
PyPDFLoader          ← Extracts text page by page
   │
   ▼
RecursiveCharacterTextSplitter
   │  chunk_size=1000, overlap=200
   ▼
[chunk_1, chunk_2, ..., chunk_N]
   │
   ▼
HuggingFaceEmbeddings  ← all-MiniLM-L6-v2 (384-dim vectors)
   │  Each chunk → float[384]
   ▼
FAISS Index            ← Stores vectors + text, saved to disk
```

### Query Pipeline (Chat Phase)

```
User Question
   │
   ▼
HuggingFaceEmbeddings  ← Same model embeds the question
   │
   ▼
FAISS similarity_search(k=4)
   │  Returns top-4 most similar chunks
   ▼
Prompt Builder
   │  "Given this context: [chunks]
   │   Answer: [question]"
   ▼
Ollama (llama3)        ← Local LLM generates answer
   │
   ▼
Answer + Source Chunks → Frontend
```

### Why These Choices?

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | all-MiniLM-L6-v2 | Fast, small (80MB), great quality for retrieval |
| Vector DB | FAISS | In-memory, no server needed, perfect for local dev |
| LLM | llama3 via Ollama | Strong instruction following, runs on CPU |
| Splitter | RecursiveCharacterTextSplitter | Respects paragraph/sentence boundaries |
| Chunk overlap | 200 chars | Prevents context from being cut at boundaries |

---

## 4. Project Structure

```
rag-ollama-project/
│
├── backend/
│   ├── main.py           ← FastAPI app, /upload and /ask endpoints
│   ├── rag_pipeline.py   ← Core RAG logic: ingestion + retrieval + LLM
│   └── requirements.txt  ← Python dependencies
│
├── frontend/
│   ├── index.html        ← Chat UI structure
│   ├── style.css         ← Dark industrial design system
│   └── script.js         ← Upload, chat, history, sources panel
│
├── data/
│   └── uploads/          ← Uploaded PDFs stored here
│
├── vectorstore/          ← FAISS index persisted here (auto-created)
│
└── README.md
```

---

## 5. Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed
- ~6GB free disk (for model weights)

### Step 1 — Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**  
Download from https://ollama.ai/download

### Step 2 — Download the LLM

```bash
ollama pull llama3
# or: ollama pull mistral
```

### Step 3 — Clone & Set Up Python Environment

```bash
git clone <your-repo-url>
cd rag-ollama-project

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat    # Windows CMD
# venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r backend/requirements.txt
```

> **Note:** `sentence-transformers` will download the embedding model (~90MB) on first run. This is automatic.

---

## 6. Running the App

### Terminal 1 — Start Ollama

```bash
ollama serve
```

You should see: `Listening on 127.0.0.1:11434`

### Terminal 2 — Start FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Terminal 3 — Open the Frontend

Simply open `frontend/index.html` in your browser:

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or use VS Code's **Live Server** extension for auto-reload.

### Using the App

1. **Upload a PDF** — Drag and drop or click the upload zone in the sidebar
2. **Wait for indexing** — You'll see chunk count when done (takes 5-30s)
3. **Ask questions** — Type in the chat box and press Enter
4. **View sources** — Click "View source chunks" to see which PDF parts were used

---

## 7. Example Queries

After uploading a research paper or report, try:

```
Summarize the main topics covered in this document.

What are the key findings or conclusions?

List all recommendations mentioned.

Explain the methodology used.

What are the limitations acknowledged by the authors?

What data sources were cited?

What problem does this paper try to solve?

Are there any statistics or numbers mentioned? List them.
```

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on /upload | Make sure `uvicorn` is running on port 8000 |
| `Ollama connection failed` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3` first |
| Slow first question | Normal — model loads into memory on first call |
| `faiss` install fails | Try `pip install faiss-cpu --no-cache-dir` |
| CORS error in browser | Ensure FastAPI is running (CORS is configured for `*`) |
| Empty answers | Try rephrasing; model answers only from PDF content |

### Switching Models

In `backend/rag_pipeline.py`, change:
```python
OLLAMA_MODEL = "llama3"
# to:
OLLAMA_MODEL = "mistral"   # or "gemma", "phi3", etc.
```

Then restart the backend.

---

## 9. Interview Concepts

Key topics interviewers ask about RAG systems:

### Chunking Strategy
- **Why chunk?** LLMs have context limits (4K–128K tokens). PDFs can be 100K+ tokens.
- **Chunk size tradeoff:** Smaller = more precise retrieval; Larger = more context per chunk.
- **Overlap:** Prevents information loss at chunk boundaries.

### Embedding Models
- **What they do:** Convert text → dense float vectors where similar meaning = similar vectors.
- **Why MiniLM?** 384 dimensions, fast CPU inference, great retrieval benchmarks.
- **Alternatives:** `text-embedding-3-small` (OpenAI), `e5-large`, `bge-large`.

### Vector Similarity
- FAISS uses **L2 (Euclidean)** or **cosine** similarity.
- With `normalize_embeddings=True`, dot product ≈ cosine similarity.

### FAISS Index Types
- `IndexFlatL2` (default): Exact search, great for <100K vectors.
- `IndexIVFFlat`: Approximate search, faster for millions of vectors.

### RAG Evaluation Metrics
- **Faithfulness:** Is the answer grounded in the retrieved context?
- **Answer relevancy:** Does the answer address the question?
- **Context recall:** Were the right chunks retrieved?

### Production Improvements
- Replace FAISS → Pinecone / Weaviate / pgvector for scale
- Add re-ranking (cross-encoder) for better retrieval precision
- Implement hybrid search (vector + BM25 keyword)
- Add query expansion / HyDE (Hypothetical Document Embeddings)
- Cache embeddings for repeated queries

---

## Tech Stack Summary

```
Frontend   : HTML + CSS + Vanilla JS (no frameworks)
Backend    : FastAPI (Python)
LLM        : Ollama → LLaMA 3 (local)
Embeddings : HuggingFace sentence-transformers (local)
Vector DB  : FAISS (in-process, no server)
PDF Parse  : LangChain PyPDFLoader (wraps pypdf)
```

**Everything runs 100% locally. Zero cloud, zero API keys, zero cost.**
# Final-RAG
