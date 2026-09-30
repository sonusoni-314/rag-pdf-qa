# RAG PDF Q&A

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B" alt="Streamlit App" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-4A4A55" alt="Ollama" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-6A5ACD" alt="ChromaDB" />
</p>

A self-hosted Retrieval-Augmented Generation (RAG) application that lets users ask natural-language questions about PDF documents and receive grounded answers with citation-backed page references. The system supports regular PDFs, scanned documents, and table-heavy content while keeping the workflow fully local.

## Overview

This project combines PDF extraction, OCR, table detection, chunking, vector search, and local LLM generation into a complete document Q&A pipeline. It is designed for privacy-focused document intelligence and can be used through a command-line interface or a Streamlit web app.

## Key Features

- Local-first PDF Q&A without paid API dependencies
- Text extraction for standard PDFs
- OCR support for scanned documents
- Table detection and conversion to readable markdown-like structure
- Chunk-based indexing for large documents
- Semantic retrieval using embeddings and vector search
- Hybrid retrieval support with BM25
- Grounded answer generation using local LLMs
- Page-level citations for answer traceability
- CLI and browser-based interfaces

## System Architecture

```text
PDF
  ↓
Page Classification
  ↓
Text Extraction / OCR
  ↓
Table Detection
  ↓
Chunking + Metadata
  ↓
Embedding + ChromaDB Storage
  ↓
Query Retrieval
  ↓
Local LLM Answer Generation
  ↓
Citations + Final Response
```

## Tech Stack

- Python
- Streamlit
- Ollama
- sentence-transformers
- ChromaDB
- pdfplumber
- pytesseract
- Tesseract OCR

## Project Structure

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
├── .gitignore
└── .venv/
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

### 4) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5) Install Ollama

Install Ollama and make sure the local model engine is running.

Check installed models:

```bash
ollama list
```

If needed, start the service manually:

```bash
ollama serve
```

The default model is configured in `config.py`.

### 6) Install Tesseract for OCR

#### Windows
Download and install Tesseract from:
https://github.com/UB-Mannheim/tesseract/wiki

Verify it is installed:

```bash
tesseract --version
```

If needed, set the Tesseract binary path in `src/ocr.py`:

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

## Run the Project

### Web interface

```bash
streamlit run app.py
```

Upload a PDF from the sidebar and ask questions in the chat box.

### Ingest a PDF from the CLI

```bash
python cli.py ingest "C:\path\to\your\document.pdf"
```

### Ask a question from the CLI

```bash
python cli.py ask "What was the Q3 revenue?"
```

### Interactive chat mode

```bash
python cli.py chat
```

## Example Workflow

1. Add a PDF to the project
2. Classify pages and extract text/OCR content
3. Detect and parse tables
4. Chunk and embed the content in ChromaDB
5. Retrieve relevant blocks for a user query
6. Generate a grounded answer with citations

## Limitations

- Scanned pages with tables can be difficult to reconstruct accurately
- Borderless tables may not be detected reliably
- The evaluation module is a lightweight quality check rather than a full semantic judge
- Model quality depends on the Ollama model selected

## License

This project is intended for educational and portfolio use.

