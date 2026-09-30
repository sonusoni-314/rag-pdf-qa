"""
Module 4: Chunking

Two chunking strategies:
  - Prose: split into overlapping chunks of ~CHUNK_SIZE tokens (recursive-ish,
    tries to break on paragraph/sentence boundaries rather than mid-word).
  - Tables: kept whole (never split), tagged as their own chunk type, already
    converted to markdown by Module 10.

Every chunk carries metadata: source file, page number, chunk type.
This metadata is what makes citations (Module 8) possible later.
"""

import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP

_encoder = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_prose(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Splits prose into overlapping chunks by token count.
    Tries to break on paragraph boundaries first, falling back to raw
    token slicing if a single paragraph is longer than chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks = []
    current = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _token_len(para)

        # paragraph itself is too big -> hard-split it by tokens
        if para_tokens > chunk_size:
            if current:
                chunks.append(current)
                current = ""
                current_tokens = 0
            tokens = _encoder.encode(para)
            start = 0
            while start < len(tokens):
                end = start + chunk_size
                piece = _encoder.decode(tokens[start:end])
                chunks.append(piece)
                next_start = end - overlap  # overlap window
                if next_start <= start:
                    next_start = end
                start = next_start
            continue

        if current_tokens + para_tokens <= chunk_size:
            current = (current + "\n\n" + para).strip()
            current_tokens += para_tokens
        else:
            if current:
                chunks.append(current)
            # start new chunk; carry a small overlap from end of previous chunk
            overlap_text = ""
            if current and overlap > 0:
                tokens = _encoder.encode(current)
                overlap_text = _encoder.decode(tokens[-overlap:])
            current = (overlap_text + "\n\n" + para).strip()
            current_tokens = _token_len(current)

    if current:
        chunks.append(current)

    return chunks


def build_chunks(source_file: str, text_pages: list, table_entries: list):
    """
    Combines extracted text pages + extracted tables into a unified list of
    chunk dicts, ready for embedding (Module 5).

    text_pages:    output of extraction.py / ocr.py
                   [{"page_num": 1, "text": "...", "type": "text"|"scanned"}, ...]
    table_entries: output of tables.py
                   [{"page_num": 4, "table_markdown": "...", "nearest_heading": "..."}, ...]

    Returns a list of chunk dicts:
    [{
        "id": "sourcefile_p1_c0",
        "text": "...",
        "source_file": "sourcefile.pdf",
        "page_num": 1,
        "chunk_type": "prose" | "table",
        "heading": "..."  # only present for table chunks
    }, ...]
    """
    all_chunks = []

    for page in text_pages:
        pieces = chunk_prose(page["text"])
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "id": f"{source_file}_p{page['page_num']}_c{idx}",
                "text": piece,
                "source_file": source_file,
                "page_num": page["page_num"],
                "chunk_type": "prose",
            })

    for idx, table in enumerate(table_entries):
        all_chunks.append({
            "id": f"{source_file}_p{table['page_num']}_table{idx}",
            "text": table["table_markdown"],
            "source_file": source_file,
            "page_num": table["page_num"],
            "chunk_type": "table",
            "heading": table["nearest_heading"],
        })

    return all_chunks


if __name__ == "__main__":
    sample = ("This is paragraph one. " * 20 + "\n\n" + "This is paragraph two. " * 20)
    chunks = chunk_prose(sample, chunk_size=50, overlap=10)
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({_token_len(c)} tokens) ---")
        print(c[:100], "...\n")
