# Setup & Run Guide — RAG PDF Q&A System

Follow these steps top to bottom on your machine.

## Step 1: Check Python is installed

Open Command Prompt and run:
```bash
python --version
```
You need Python 3.9 or higher. If this errors out, download Python from
python.org — **during install, check "Add Python to PATH"** — then restart
your terminal and try again.

## Step 2: Unzip the project

Unzip `rag_pdf_qa_FINAL.zip` somewhere easy to find, e.g. `C:\Users\sonus\rag_pdf_qa`.

## Step 3: Open terminal in that folder

```bash
cd C:\Users\sonus\rag_pdf_qa
```
(adjust the path to wherever you unzipped it)

## Step 4: Create a virtual environment

```bash
python -m venv venv
```

## Step 5: Activate the virtual environment

```bash
venv\Scripts\activate
```
Your prompt should now show `(venv)` at the start. **Run this every time you
open a new terminal to work on this project.**

## Step 6: Install all Python dependencies

```bash
pip install -r requirements.txt
```
Takes a few minutes — `sentence-transformers` pulls in PyTorch (~500MB–1GB).
Let it finish completely.

## Step 7: Install Tesseract (for OCR on scanned PDFs)

This is a separate program, not a Python package — `pip` alone won't install it.

1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the Windows installer (64-bit)
3. Run it — note the install path shown (usually `C:\Program Files\Tesseract-OCR`)
4. Check "Add to PATH" if the installer offers it

**Verify it worked:**
```bash
tesseract --version
```
If this shows a version number, you're good. If it says "not recognized,"
open `rag_pdf_qa\src\ocr.py` and add this line right after the imports:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Step 8: Confirm Ollama is set up

```bash
ollama list
```
You should see `phi3:mini` in the list. If Ollama commands hang or fail,
open a **separate** terminal window and run:
```bash
ollama serve
```
Leave that window open in the background, then return to your original terminal.

## Step 9: Ingest your first PDF

```bash
python cli.py ingest "C:\path\to\your\document.pdf"
```
Expected output:
```
[1/5] Classifying pages in document.pdf...
      3 text page(s), 0 scanned page(s)
[2/5] Extracting text...
[3/5] Detecting tables...
      Found 1 table(s)
[4/5] Chunking...
      Built 5 chunk(s)
[5/5] Embedding + storing in ChromaDB...
Done. document.pdf is ready to query.
```
The first run also triggers a small one-time internet download for
`tiktoken`'s tokenizer file — make sure you have a normal internet
connection for this first run.

## Step 10: Ask a question

```bash
python cli.py ask "give me a summary of this document"
```
You should get an answer with a `Sources:` section listing page numbers.

## Step 11 (optional): Interactive chat mode

```bash
python cli.py chat
```
Keeps asking questions without re-running the command each time.
Type `exit` to quit.

## Step 12 (optional): Web UI

```bash
streamlit run app.py
```
Opens a browser tab automatically. Upload a PDF in the sidebar, then ask
questions in the chat box.

## Troubleshooting

If any step fails, copy the **full, exact error message** from your
terminal (not a paraphrase) and share it — that's the fastest way to
diagnose exactly what's wrong.

Common issues:
- `ModuleNotFoundError` → Step 5 (venv not activated) or Step 6 (deps not installed) was skipped
- `tesseract is not installed or it's not in your PATH` → revisit Step 7
- Ollama connection errors → revisit Step 8, make sure `ollama serve` is running
- `python cli.py ingest` produces 0 chunks → PDF may be entirely scanned images with no OCR run, or a truly blank/corrupt file
