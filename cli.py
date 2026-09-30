"""
Module 11: CLI Interface

Usage:
    python cli.py ingest path/to/file.pdf
    python cli.py ask "What was the Q3 revenue?"
    python cli.py chat              # interactive Q&A loop
"""

import sys
import argparse

from src.pipeline import ingest_pdf, answer_question
from src.vector_store import VectorStore


def cmd_ingest(args):
    vs = VectorStore()
    ingest_pdf(args.pdf_path, vs)


def cmd_ask(args):
    vs = VectorStore()
    answer = answer_question(args.question, vs)
    print("\n" + answer + "\n")


def cmd_chat(args):
    vs = VectorStore()
    print(f"Loaded {vs.count()} chunks. Type your questions (or 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = answer_question(question, vs)
        print(f"\n{answer}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG PDF Q&A CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A loop")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
