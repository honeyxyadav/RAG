"""
RAG Pipeline
============
The brain of the application. Handles:
  • PDF ingestion   → text extraction, chunking, embedding, FAISS storage
  • Question answering → retrieval, prompt building, LLM call

Flow Diagram:
  PDF → PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings
      → FAISS index (saved to disk)

  Question → embed query → FAISS similarity search → top-k chunks
           → build prompt → Ollama or Groq → answer
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()


class RAGPipeline:
    """
    Encapsulates the full Retrieval-Augmented Generation pipeline.

    Attributes:
        vector_store_path  : Directory where FAISS index is persisted
        embeddings         : HuggingFace sentence-transformer model
        vector_store       : FAISS index (None until a PDF is ingested)
        llm                : Ollama LLM instance
    """

    # ── Config constants (tweak these to experiment) ────────────────────────────
    CHUNK_SIZE       = 1000   # Characters per chunk
    CHUNK_OVERLAP    = 200    # Overlap between consecutive chunks (preserves context)
    TOP_K_RESULTS    = 4      # Number of chunks retrieved per query
    EMBED_MODEL      = "all-MiniLM-L6-v2"   # Fast, lightweight, 384-dim embeddings
    OLLAMA_MODEL     = "gemma:2b"             # Very small model for systems with limited memory (requires ~1-2 GiB)
    GROQ_MODEL       = "llama-3.1-8b-instant"    # Groq model (supported)
    FAISS_INDEX_NAME = "pdf_index"
    
    # Provider options: "ollama" (local) or "groq" (Groq API)
    DEFAULT_PROVIDER = "ollama"

    def __init__(self, vector_store_path: str, provider: str = DEFAULT_PROVIDER):
        self.vector_store_path = vector_store_path
        self.vector_store: Optional[FAISS] = None
        self.provider = provider

        print("[RAG] Loading embedding model (first run downloads ~90MB)...")
        # HuggingFace embeddings run fully locally — no API key needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBED_MODEL,
            model_kwargs={"device": "cpu"},   # Use "cuda" if you have a GPU
            encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize LLM based on provider
        self._initialize_llm()

        # Load existing FAISS index from disk (if a PDF was previously uploaded)
        self._load_existing_index()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the selected provider."""
        if self.provider == "ollama":
            print(f"[RAG] Connecting to Ollama ({self.OLLAMA_MODEL})...")
            # Ollama must be running locally: `ollama serve`
            self.llm = Ollama(
                model=self.OLLAMA_MODEL,
                temperature=0.1,        # Low temperature = more factual, less creative
                # base_url="http://localhost:11434"  # Default Ollama URL
            )
        elif self.provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file.")
            print(f"[RAG] Connecting to Groq API ({self.GROQ_MODEL})...")
            # Store API key for Groq requests
            self.groq_api_key = groq_api_key
            # Store a wrapper that uses requests for Groq API
            self.llm = self._groq_llm_wrapper
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'ollama' or 'groq'.")
    
    def _groq_llm_wrapper(self, prompt: str) -> str:
        """Wrapper for Groq API calls using requests."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            error_detail = response.text
            print(f"[RAG] Groq API Error: {error_detail}")
            raise ValueError(f"Groq API error: {error_detail}")
    
    def set_provider(self, provider: str):
        """Switch the LLM provider dynamically."""
        if provider != self.provider:
            self.provider = provider
            self._initialize_llm()
            print(f"[RAG] Switched to {provider} provider.")

    # ── Public Methods ──────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> int:
        """
        Full ingestion pipeline: PDF → chunks → embeddings → FAISS.

        Args:
            pdf_path: Absolute path to the uploaded PDF file.

        Returns:
            Number of chunks indexed.
        """
        print(f"\n[RAG] ── Ingestion started: {os.path.basename(pdf_path)}")

        # ── Step 1: Extract text from PDF ──────────────────────────────────────
        print("[RAG] Step 1/4: Extracting text from PDF...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()                       # List[Document], one per page
        print(f"       → {len(pages)} pages extracted")

        # ── Step 2: Split text into chunks ─────────────────────────────────────
        print("[RAG] Step 2/4: Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            # These separators are tried in order; falls back to character split
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks: list[Document] = splitter.split_documents(pages)
        print(f"       → {len(chunks)} chunks created "
              f"(size={self.CHUNK_SIZE}, overlap={self.CHUNK_OVERLAP})")

        # ── Step 3: Embed chunks ────────────────────────────────────────────────
        print("[RAG] Step 3/4: Generating embeddings (this may take a moment)...")
        # FAISS.from_documents embeds all chunks in one batch call
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"       → Embeddings generated using '{self.EMBED_MODEL}'")

        # ── Step 4: Persist FAISS index to disk ─────────────────────────────────
        print("[RAG] Step 4/4: Saving FAISS index to disk...")
        self.vector_store.save_local(
            self.vector_store_path,
            index_name=self.FAISS_INDEX_NAME
        )
        print(f"       → Index saved to {self.vector_store_path}")
        print(f"[RAG] ── Ingestion complete! {len(chunks)} chunks ready for retrieval.\n")

        return len(chunks)

    def answer_question(
        self,
        question: str,
        chat_history: list[dict]
    ) -> tuple[str, list[str]]:
        """
        Retrieve relevant chunks and generate an answer using Ollama.

        Args:
            question    : The user's question.
            chat_history: Previous turns [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            (answer_text, list_of_source_chunk_texts)
        """
        # ── Step 5: Retrieve relevant chunks ───────────────────────────────────
        print(f"\n[RAG] Question: {question!r}")
        print(f"[RAG] Retrieving top {self.TOP_K_RESULTS} chunks from FAISS...")

        results: list[Document] = self.vector_store.similarity_search(
            query=question,
            k=self.TOP_K_RESULTS
        )
        source_chunks = [doc.page_content for doc in results]
        context = "\n\n---\n\n".join(source_chunks)
        print(f"       → Retrieved {len(results)} chunks")

        # ── Step 6: Build prompt ────────────────────────────────────────────────
        # Format recent chat history (last 6 turns to avoid token overflow)
        history_text = self._format_history(chat_history[-6:])

        prompt = f"""You are a helpful assistant that answers questions STRICTLY based on the provided PDF context.

RULES:
- Only use information from the CONTEXT below.
- If the answer is not in the context, say: "I couldn't find that information in the uploaded PDF."
- Be concise and precise.
- Do NOT make up information.

CONTEXT (extracted from the PDF):
{context}

CHAT HISTORY:
{history_text}

USER QUESTION: {question}

ANSWER:"""

        print(f"[RAG] Sending prompt to {self.provider}...")

        # ── Step 7: Query LLM ───────────────────────────────────────────────────
        if self.provider == "groq":
            # Groq uses a function wrapper
            answer = self.llm(prompt)
        else:
            # Ollama uses invoke method
            answer = self.llm.invoke(prompt)
        print(f"[RAG] Answer received ({len(answer)} chars)")

        return answer.strip(), source_chunks

    def is_ready(self) -> bool:
        """Returns True if a FAISS index is loaded and ready for queries."""
        return self.vector_store is not None

    def reset(self):
        """Clear the in-memory vector store."""
        self.vector_store = None
        # Optionally delete the persisted index files here
        print("[RAG] Vector store reset.")

    # ── Private Helpers ─────────────────────────────────────────────────────────

    def _load_existing_index(self):
        """Try to load a previously saved FAISS index from disk."""
        index_file = os.path.join(
            self.vector_store_path,
            f"{self.FAISS_INDEX_NAME}.faiss"
        )
        if os.path.exists(index_file):
            print("[RAG] Found existing FAISS index — loading from disk...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                index_name=self.FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True   # Safe: local files only
            )
            print("[RAG] Index loaded ✓")
        else:
            print("[RAG] No existing index found. Upload a PDF to get started.")

    def _format_history(self, chat_history: list[dict]) -> str:
        """Convert chat history list into a readable string for the prompt."""
        if not chat_history:
            return "(No prior conversation)"
        lines = []
        for turn in chat_history:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)
