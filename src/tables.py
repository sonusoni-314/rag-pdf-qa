"""
Module 10: Table Detection & Extraction

Detects tables on each page and converts them to markdown so the LLM can
read structure (rows/columns) instead of getting scrambled flat text.
Each table is tagged with its page number and the nearest preceding heading
found in that page's prose, for better citation context.

Uses pdfplumber's built-in table detection (works well for native/text PDFs
with visible gridlines or consistent spacing). For scanned pages, tables
would need OCR-based detection (unstructured/camelot) — flagged as a
known limitation below rather than silently failing.
"""

import re
import pdfplumber


def table_to_markdown(table_rows) -> str:
    """Converts a list-of-lists table (from pdfplumber) into a markdown table."""
    if not table_rows or not table_rows[0]:
        return ""

    def clean(cell):
        return (cell or "").strip().replace("\n", " ")

    header = [clean(c) for c in table_rows[0]]
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in table_rows[1:]:
        row_cells = [clean(c) for c in row]
        # pad short rows so the markdown table doesn't break
        while len(row_cells) < len(header):
            row_cells.append("")
        md_lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(md_lines)


def find_nearest_heading(page_text: str) -> str:
    """
    Very simple heading heuristic: looks for a short line (<10 words) that's
    either ALL CAPS or Title Case near the top of the page's text, which
    usually corresponds to a section heading. Falls back to "Untitled section".
    """
    if not page_text:
        return "Untitled section"

    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    for line in lines[:8]:  # only look near the top of the page
        word_count = len(line.split())
        if word_count <= 10 and (line.isupper() or line.istitle()):
            return line
    return "Untitled section"


def extract_tables(pdf_path: str):
    """
    Detects and extracts tables from every page of the PDF.

    Returns a list of dicts:
    [{
        "page_num": 4,
        "table_markdown": "| col1 | col2 |\\n|---|---|\\n| a | b |",
        "nearest_heading": "Financial Summary",
        "type": "table"
    }, ...]

    Pages with no detected tables are simply absent from the output.
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            tables = page.extract_tables()  # default = line/gridline-based detection
            if not tables:
                continue

            page_text = page.extract_text() or ""
            heading = find_nearest_heading(page_text)

            for table in tables:
                md = table_to_markdown(table)
                if md.strip():
                    results.append({
                        "page_num": page_num,
                        "table_markdown": md,
                        "nearest_heading": heading,
                        "type": "table",
                    })
    return results


# KNOWN LIMITATIONS (found during real testing, not just theoretical):
#
# 1. Scanned pages with tables aren't handled here at all. pdfplumber needs
#    a real text/vector layer to find gridlines. A scanned table would need
#    OCR + layout detection (e.g. unstructured's hi_res strategy).
#
# 2. This only detects tables with VISIBLE GRIDLINES (drawn border lines).
#    Borderless tables (common in PDFs exported from Word/Google Docs,
#    where columns are just aligned whitespace) will be missed.
#    We tested pdfplumber's "text" strategy as a fallback for this, but it's
#    too aggressive without per-document tuning — it starts misreading
#    ordinary paragraphs as garbled pseudo-tables, which is worse than
#    missing a real table (bad data fed to the LLM beats no data). If you
#    hit this in your own PDFs, the fix is to tune vertical_strategy/
#    horizontal_strategy/snap_tolerance in page.extract_tables() for your
#    specific document, not to leave the aggressive fallback on globally.


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tables.py <path_to_pdf>")
        sys.exit(1)

    tables = extract_tables(sys.argv[1])
    if not tables:
        print("No tables detected.")
    for t in tables:
        print(f"--- Table on page {t['page_num']} (near: '{t['nearest_heading']}') ---")
        print(t["table_markdown"], "\n")
