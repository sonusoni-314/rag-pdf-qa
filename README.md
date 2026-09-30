# RAG-Powered PDF Q&A System (Self-Hosted, OCR & Table-Aware)

Ask natural-language questions over your PDFs — clean text, scanned, or
table-heavy — and get grounded, cited answers. Entirely local: no paid
API keys, no accounts, everything runs on your machine.

## Architecture

```
PDF → classify pages (text vs scanned)
    → extract text (pdfplumber) OR OCR (Tesseract) per page type
    → detect + extract tables → convert to markdown
    → chunk (prose: overlapping token chunks | tables: kept whole)
    → embed (sentence-transformers, local) → store (ChromaDB, local)

User question → embed query → retrieve top-k chunks
             → (optional) hybrid rerank with BM25
             → generate answer (Ollama, local LLM, grounded in context only)
             → attach guaranteed-accurate page citations
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM (generation) | Ollama — default `phi3:mini`, configurable in `config.py` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (local, persisted to `data/chroma_db/`) |
| PDF/text parsing | `pdfplumber` |
| OCR (scanned PDFs) | Tesseract via `pytesseract` |
| Table extraction | `pdfplumber` table detection → markdown |
| Keyword search (stretch) | `rank_bm25` |
| Interface | CLI (`cli.py`) and Web UI (`app.py`, Streamlit) |

## Setup

### 1. Python environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Ollama
Make sure Ollama is running and has a model pulled. This project defaults to
`phi3:mini`. Check what you have:
```bash
ollama list
```
If you want a different model, change `OLLAMA_MODEL` in `config.py`.

### 3. Tesseract (for OCR on scanned PDFs)
`pip install pytesseract` only installs the Python wrapper — the actual
Tesseract program must be installed separately:
- **Windows**: [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

If Tesseract isn't on your PATH after install, set the path explicitly at
the top of `src/ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Usage

### CLI
```bash
python cli.py ingest path\to\document.pdf   # process and store a PDF
python cli.py ask "What was the Q3 revenue?"  # ask a one-off question
python cli.py chat                           # interactive Q&A loop
```

### Web UI
```bash
streamlit run app.py
```
Opens a browser tab — upload a PDF in the sidebar, ask questions in the chat box.

### Hybrid search (stretch)
```bash
python -m src.hybrid_search "your question"
```
Combines vector similarity with BM25 keyword search via reciprocal rank
fusion — generally more robust on numeric/table-heavy queries.

### Evaluation (stretch)
1. Edit `tests/eval_qa_set.json` with real questions about a PDF you've
   ingested, the page number you expect the answer to come from, and
   keywords you expect in the answer.
2. Run:
```bash
python -m src.evaluation
```
Prints retrieval accuracy and a basic answer-faithfulness rate.

## Project Structure
```
rag_pdf_qa/
├── config.py              # all settings (model names, chunk size, paths)
├── cli.py                 # Module 11: CLI interface
├── app.py                 # Module 12: Streamlit web UI
├── requirements.txt
├── src/
│   ├── pdf_loader.py       # Module 2: page classification
│   ├── extraction.py       # Module 3: text extraction
│   ├── chunking.py         # Module 4: chunking
│   ├── vector_store.py     # Module 5: embedding + ChromaDB
│   ├── retrieval.py        # Module 6: retrieval
│   ├── generation.py       # Module 7: Ollama generation
│   ├── citations.py        # Module 8: citation formatting
│   ├── ocr.py               # Module 9: OCR for scanned pages
│   ├── tables.py            # Module 10: table detection + markdown
│   ├── pipeline.py          # orchestrates modules 2-10 end to end
│   ├── hybrid_search.py     # Module 13 (stretch): BM25 + vector fusion
│   └── evaluation.py        # Module 14 (stretch): retrieval/faithfulness eval
├── tests/
│   └── eval_qa_set.json    # your hand-built Q&A test set
└── data/
    ├── uploads/             # PDFs you upload (gitignored)
    └── chroma_db/           # vector store (gitignored)
```

## Known Limitations (be upfront about these — don't oversell)
- **Scanned pages with tables** aren't handled: table detection needs a
  real text/vector layer, which scanned images don't have. A page that's
  both scanned *and* contains a table will get OCR'd as plain text, and
  the table structure will be lost.
- **Borderless tables are missed.** Table detection only catches tables
  with visible gridlines. Tables with no drawn borders (common in PDFs
  exported from Word/Google Docs) won't be detected — tested and confirmed
  during build. A "text alignment" fallback strategy exists in pdfplumber,
  but it's too unreliable without per-document tuning (it starts misreading
  ordinary paragraphs as garbled tables), so it's intentionally left off by
  default. If your PDFs use borderless tables, see the comment block at the
  bottom of `src/tables.py` for how to tune it for your specific document.
- **Heading detection** for tables (`find_nearest_heading` in `tables.py`)
  is a simple heuristic (short ALL-CAPS or Title-Case line near the top
  of the page) — it won't be perfect on documents with unusual formatting.
- **Answer faithfulness checking** (Module 14) is keyword-matching, not
  true semantic faithfulness — a proper version would use a second LLM
  call as a judge. Documented as a possible future improvement.
- The 0.5B/small models in your Ollama list (e.g. `qwen2.5:0.5b`) will
  produce noticeably weaker answers than `phi3:mini` or `gemma2:latest` —
  stick to the larger ones for real use.

## Build Progress
- [x] Module 1: Scaffolding
- [x] Module 2: PDF Loader & Page Classifier
- [x] Module 3: Text Extraction
- [x] Module 4: Chunking
- [x] Module 5: Embedding + Vector Store
- [x] Module 6: Retrieval
- [x] Module 7: Generation (Ollama)
- [x] Module 8: Citation Formatting
- [x] Module 9: OCR Path
- [x] Module 10: Table Extraction
- [x] Module 11: CLI
- [x] Module 12: Web UI
- [x] Module 13: Hybrid Search (stretch)
- [x] Module 14: Evaluation Pipeline (stretch)
