"""
Module 6: Retrieval

Thin wrapper around the vector store's query function. Kept as its own
module so retrieval logic (e.g. hybrid search in Module 13) can evolve
independently of the storage layer.
"""

from config import TOP_K
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearcher


def retrieve_chunks(query: str, vector_store: VectorStore, top_k: int = TOP_K):
    """
    Retrieves the top_k most relevant chunks for a query.
    Returns: [{"text": ..., "metadata": {...}, "distance": ...}, ...]
    Lower distance = more similar (Chroma uses distance, not similarity score).
    """
    if vector_store.count() == 0:
        return []

    # Semantic search finds paraphrases; BM25 preserves exact names, dates,
    # model numbers, and table values. RRF combines both signals.
    return HybridSearcher(vector_store).search(
        query,
        top_k=min(top_k, vector_store.count()),
        candidate_pool=min(20, vector_store.count()),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py \"your question here\"")
        sys.exit(1)

    vs = VectorStore()
    results = retrieve_chunks(sys.argv[1], vs)
    for r in results:
        meta = r["metadata"]
        print(f"[page {meta['page_num']} | {meta['chunk_type']}] (dist={r['distance']:.4f})")
        print(r["text"][:200], "...\n")
