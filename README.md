# RAG PDF Q&A

A self-hosted Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents using local AI models, OCR, and vector search. The project processes text, scanned pages, and table-heavy PDFs, then retrieves the most relevant chunks and generates grounded answers with page-level citations.

## Why this project matters

- Works fully on your local machine
- No paid API keys required for the main workflow
- Supports text PDFs, scanned PDFs, and table extraction
- Gives cited answers grounded in the source document
- Provides both CLI and Streamlit web interfaces

## Tech Stack

- Python
- Streamlit
- Ollama
- sentence-transformers
- ChromaDB
- pdfplumber
- pytesseract
- Tesseract OCR

## Project Architecture

```text
PDF → page classification
  → text extraction or OCR
  → table detection
  → chunking
  → embedding and vector storage
  → retrieval
  → local LLM generation
  → citation-aware final answer
```

## Features

- PDF ingestion and document indexing
- Text extraction for normal PDFs
- OCR for scanned documents
- Table recognition and markdown conversion
- Chunking with metadata for page-level context
- Semantic retrieval using local embeddings
- Hybrid search support with BM25
- Grounded answer generation with source citations
- CLI and browser-based UI

## Repository Structure

```text
rag_pdf_qa/
├── app.py
├── cli.py
├── config.py
├── README.md
├── SETUP.md
├── requirements.txt
├── src/
│   ├── chunking.py
│   ├── citations.py
│   ├── evaluation.py
│   ├── extraction.py
│   ├── generation.py
│   ├── hybrid_search.py
│   ├── ocr.py
│   ├── pdf_loader.py
│   ├── pipeline.py
│   ├── retrieval.py
│   ├── tables.py
│   └── vector_store.py
├── tests/
│   └── eval_qa_set.json
├── data/
│   ├── chroma_db/
│   └── uploads/
└── .gitignore
```

## Setup Guide

### 1) Clone the repository

```bash
git clone https://github.com/sonusoni-314/rag-pdf-qa.git
cd rag-pdf-qa
```

### 2) Create a virtual environment

```bash
python -m venv venv
```

### 3) Activate the environment

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

### 5) Install Ollama

Install Ollama from the official website and make sure the service is running.

Check available models:

```bash
ollama list
```

If you need to start the service manually:

```bash
ollama serve
```

The project defaults to a local model configured in `config.py`.

### 6) Install Tesseract (required for scanned PDFs)

#### Windows
Download and install Tesseract from:
https://github.com/UB-Mannheim/tesseract/wiki

Then verify:

```bash
tesseract --version
```

If needed, configure the path in `src/ocr.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

#### macOS
```bash
brew install tesseract
```

#### Linux
```bash
sudo apt install tesseract-ocr
```

## Run the Application

### Web interface

```bash
streamlit run app.py
```

Then upload a PDF from the sidebar and ask a question in the chat box.

### CLI ingestion

```bash
python cli.py ingest "C:\path\to\your\document.pdf"
```

### CLI question answering

```bash
python cli.py ask "What was the Q3 revenue?"
```

### Interactive chat mode

```bash
python cli.py chat
```

## Example Workflow

1. Upload or ingest a PDF
2. The system classifies pages and extracts text/OCR content
3. Tables are detected and converted into structured text
4. Chunks are embedded and stored in ChromaDB
5. The user asks a question
6. Relevant chunks are retrieved
7. A local LLM generates a grounded answer with source citations

## Notes and Limitations

- Scanned pages with tables can be difficult to reconstruct accurately
- Borderless tables may not be detected reliably
- The evaluation module is a lightweight check, not a full semantic judge
- Performance depends on the model chosen in Ollama

## License

This project is for educational and portfolio purposes.

