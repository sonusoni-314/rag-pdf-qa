"""
Module 14 (Stretch): Evaluation Pipeline

Measures two things against a hand-built Q&A test set (see
tests/eval_qa_set.json for the format):

  1. Retrieval accuracy: did the expected page show up in the top-k
     retrieved chunks?
  2. Answer faithfulness (basic version): does the generated answer
     contain the expected keywords/facts? This is a simple proxy —
     true faithfulness checking would need a second LLM-as-judge pass,
     which is a reasonable future improvement but out of scope here.

Run this after you've ingested the PDFs referenced in your test set.
"""

import json
from src.vector_store import VectorStore
from src.retrieval import retrieve_chunks
from src.generation import generate_answer


def load_qa_set(path: str):
    with open(path, "r") as f:
        return json.load(f)


def evaluate(qa_set_path: str, top_k: int = 4):
    """
    qa_set format (list of dicts):
    [{
        "question": "What was Q3 revenue?",
        "expected_page": 5,
        "expected_keywords": ["4.2 million", "revenue"]
    }, ...]
    """
    qa_set = load_qa_set(qa_set_path)
    vs = VectorStore()

    retrieval_hits = 0
    faithfulness_hits = 0
    results_log = []

    for item in qa_set:
        question = item["question"]
        expected_page = item.get("expected_page")
        expected_keywords = item.get("expected_keywords", [])

        retrieved = retrieve_chunks(question, vs, top_k=top_k)
        retrieved_pages = [r["metadata"]["page_num"] for r in retrieved]

        retrieval_hit = expected_page in retrieved_pages
        if retrieval_hit:
            retrieval_hits += 1

        answer = generate_answer(question, retrieved)
        answer_lower = answer.lower()
        keyword_hit = any(kw.lower() in answer_lower for kw in expected_keywords)
        if keyword_hit:
            faithfulness_hits += 1

        results_log.append({
            "question": question,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "retrieval_hit": retrieval_hit,
            "faithfulness_hit": keyword_hit,
            "answer": answer,
        })

    n = len(qa_set) if qa_set else 1
    summary = {
        "retrieval_accuracy": retrieval_hits / n,
        "faithfulness_rate": faithfulness_hits / n,
        "total_questions": len(qa_set),
    }

    return summary, results_log


if __name__ == "__main__":
    import sys
    qa_path = sys.argv[1] if len(sys.argv) > 1 else "tests/eval_qa_set.json"
    summary, log = evaluate(qa_path)
    print(f"Retrieval accuracy: {summary['retrieval_accuracy']:.0%}")
    print(f"Faithfulness rate:  {summary['faithfulness_rate']:.0%}")
    print(f"(over {summary['total_questions']} questions)")
