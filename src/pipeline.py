"""
Pipeline: ties together every module into two entry points:

  ingest_pdf(path)  -> classify pages, extract text/OCR, extract tables,
                        chunk everything, embed, store in ChromaDB
  answer_question(q) -> retrieve relevant chunks, generate grounded answer,
                        attach guaranteed-accurate citations

This is what Module 11 (CLI) and Module 12 (Web UI) both call — neither
interface contains any RAG logic itself, they just call these two functions.
"""

import os
import re

from src.pdf_loader import classify_pdf
from src.extraction import extract_text_pages
from src.ocr import extract_ocr_pages
from src.tables import extract_tables
from src.chunking import build_chunks
from src.vector_store import VectorStore
from src.retrieval import retrieve_chunks
from src.generation import generate_answer
from src.citations import format_answer_with_sources


def ingest_pdf(pdf_path: str, vector_store: VectorStore = None, verbose: bool = True):
    """
    Full ingestion pipeline for one PDF. Returns the number of chunks stored.
    """
    if vector_store is None:
        vector_store = VectorStore()

    source_file = os.path.basename(pdf_path)

    if verbose:
        print(f"[1/5] Classifying pages in {source_file}...")
    classification = classify_pdf(pdf_path)
    n_text = sum(1 for p in classification if p["type"] == "text")
    n_scanned = sum(1 for p in classification if p["type"] == "scanned")
    if verbose:
        print(f"      {n_text} text page(s), {n_scanned} scanned page(s)")

    if verbose:
        print("[2/5] Extracting text...")
    text_pages = extract_text_pages(pdf_path)
    if n_scanned > 0:
        if verbose:
            print(f"      Running OCR on {n_scanned} scanned page(s) (this can be slow)...")
        text_pages += extract_ocr_pages(pdf_path)

    if verbose:
        print("[3/5] Detecting tables...")
    table_entries = extract_tables(pdf_path)
    if verbose:
        print(f"      Found {len(table_entries)} table(s)")

    if verbose:
        print("[4/5] Chunking...")
    chunks = build_chunks(source_file, text_pages, table_entries)
    if verbose:
        print(f"      Built {len(chunks)} chunk(s)")

    if verbose:
        print("[5/5] Embedding + storing in ChromaDB...")
    vector_store.add_chunks(chunks)

    if verbose:
        print(f"Done. {source_file} is ready to query.")

    return len(chunks)


def answer_question(query: str, vector_store: VectorStore = None) -> str:
    """
    Full query pipeline: retrieve -> generate -> attach citations.
    Returns the final formatted answer string.
    """
    if vector_store is None:
        vector_store = VectorStore()

    retrieved = retrieve_chunks(query, vector_store)
    raw_answer = generate_answer(query, retrieved)
    cited_pages = {
        int(page) for page in re.findall(r"\[Page\s+(\d+)\]", raw_answer)
    }
    source_chunks = (
        [chunk for chunk in retrieved if chunk["metadata"].get("page_num") in cited_pages]
        if cited_pages
        else retrieved
    )
    return format_answer_with_sources(raw_answer, source_chunks)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <path_to_pdf>")
        sys.exit(1)

    vs = VectorStore()
    ingest_pdf(sys.argv[1], vs)
