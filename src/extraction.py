"""
Module 3: Text Extraction (clean, text-based PDFs)

Extracts raw text from pages already classified as "text" by Module 2.
Scanned pages are handled separately in Module 9 (OCR).
"""

import pdfplumber
from src.pdf_loader import classify_page


def extract_text_pages(pdf_path: str):
    """
    Extracts text from every page in the PDF that is classified as "text".

    Returns a list of dicts:
    [{"page_num": 1, "text": "...", "type": "text"}, ...]

    Pages classified as "scanned" are skipped here (Module 9 picks them up).
    """
    extracted = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_type = classify_page(page)
            if page_type == "text":
                text = page.extract_text() or ""
                extracted.append({
                    "page_num": page_num,
                    "text": text,
                    "type": "text",
                })
    return extracted


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extraction.py <path_to_pdf>")
        sys.exit(1)

    pages = extract_text_pages(sys.argv[1])
    for p in pages:
        print(f"--- Page {p['page_num']} ---")
        print(p["text"][:300], "...\n")
