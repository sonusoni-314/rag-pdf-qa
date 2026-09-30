"""
Module 9: OCR Path (scanned PDFs)

For pages classified as "scanned" by Module 2, renders the page as an image
and runs Tesseract OCR to pull text out. Output format matches Module 3's
extraction output so both feed into the same chunking module (Module 4).
"""

import os
import shutil

import pdfplumber
import pytesseract
from src.pdf_loader import classify_page

# Higher resolution = better OCR accuracy but slower. 300 is a good default.
OCR_DPI = 300

_tesseract_candidates = (
    shutil.which("tesseract"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
for _tesseract_path in _tesseract_candidates:
    if _tesseract_path and os.path.isfile(_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path
        break


def ocr_page(page) -> str:
    """Renders a single pdfplumber page to an image and OCRs it."""
    im = page.to_image(resolution=OCR_DPI).original  # PIL image
    text = pytesseract.image_to_string(im)
    return text


def extract_ocr_pages(pdf_path: str):
    """
    Runs OCR on every page classified as "scanned".

    Returns a list of dicts (same shape as extraction.py):
    [{"page_num": 3, "text": "...", "type": "scanned"}, ...]
    """
    extracted = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_type = classify_page(page)
            if page_type == "scanned":
                text = ocr_page(page)
                extracted.append({
                    "page_num": page_num,
                    "text": text,
                    "type": "scanned",
                })
    return extracted


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <path_to_pdf>")
        print("Note: requires Tesseract installed on your system (not just pip package).")
        sys.exit(1)

    pages = extract_ocr_pages(sys.argv[1])
    for p in pages:
        print(f"--- Page {p['page_num']} (OCR) ---")
        print(p["text"][:300], "...\n")
