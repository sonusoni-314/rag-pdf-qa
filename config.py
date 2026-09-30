"""
Central configuration for the RAG PDF Q&A system.
Every module imports from here instead of hardcoding values,
so you only ever change settings in one place.
"""

import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ---------- Ollama (local LLM) ----------
# Change this to whatever model you've pulled: `ollama list` to check.
# qwen2.5:0.5b is the fast local default. Use phi3:mini for higher quality
# answers when response time is less important.
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 220

# ---------- Embeddings ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # swap to "BAAI/bge-small-en" if you want

# ---------- Chunking ----------
CHUNK_SIZE = 400        # tokens per chunk (prose)
CHUNK_OVERLAP = 50      # token overlap between chunks

# ---------- Retrieval ----------
TOP_K = 4                # number of chunks to retrieve per query

# ---------- ChromaDB ----------
COLLECTION_NAME = "pdf_qa_chunks"

# ---------- Misc ----------
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
