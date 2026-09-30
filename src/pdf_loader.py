"""
Module 2: PDF Loader & Page Classifier

Loads a PDF, splits it into pages, and classifies each page as either:
  - "text"    -> has a real, extractable text layer
  - "scanned" -> image-only page, needs OCR (handled in Module 9)

Classification heuristic: if pdfplumber can pull a meaningful amount of
text directly from the page, it's a text page. If it comes back empty
(or nearly empty), it's almost certainly a scanned image.
"""

import pdfplumber

# Minimum characters of extracted text before we trust it's a real text page.
MIN_TEXT_CHARS = 20


def load_pdf(pdf_path: str):
    """Opens a PDF and returns the pdfplumber PDF object (caller must close it)."""
    return pdfplumber.open(pdf_path)


def classify_page(page) -> str:
    """
    Returns "text" or "scanned" for a single pdfplumber page.
    """
    text = page.extract_text() or ""
    if len(text.strip()) >= MIN_TEXT_CHARS:
        return "text"
    return "scanned"


def classify_pdf(pdf_path: str):
    """
    Loads a PDF and returns a list of dicts, one per page:
    [{"page_num": 1, "type": "text"}, {"page_num": 2, "type": "scanned"}, ...]

    page_num is 1-indexed to match how humans refer to PDF pages (for citations later).
    """
    results = []
    with load_pdf(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_type = classify_page(page)
            results.append({"page_num": i + 1, "type": page_type})
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    classification = classify_pdf(sys.argv[1])
    for entry in classification:
        print(f"Page {entry['page_num']}: {entry['type']}")
