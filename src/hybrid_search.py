"""
Module 13 (Stretch): Hybrid Search

Combines vector similarity (semantic) with BM25 (keyword) search. This
helps a lot on numeric/table-heavy queries where exact term matching
("Q3", "4.2 million") matters more than semantic similarity.

Approach: pull a wider candidate pool from BOTH methods, then merge with
reciprocal rank fusion (RRF) — simple, no tuning of weight coefficients needed.
"""

from rank_bm25 import BM25Okapi
from src.vector_store import VectorStore

RRF_K = 60  # standard constant for reciprocal rank fusion


def _tokenize(text: str):
    return text.lower().split()


class HybridSearcher:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Pulls all stored chunks out of Chroma to build a BM25 index over them."""
        raw = self.vector_store.collection.get(include=["documents", "metadatas"])
        self.doc_ids = raw["ids"]
        self.documents = raw["documents"]
        self.metadatas = raw["metadatas"]

        tokenized = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 4, candidate_pool: int = 15):
        if self.bm25 is None or not self.documents:
            return self.vector_store.query(query, top_k=top_k)

        # --- vector candidates ---
        vector_results = self.vector_store.query(query, top_k=candidate_pool)
        vector_ranked_ids = []
        for r in vector_results:
            # match back to doc id via text content (Chroma query doesn't return ids by default here)
            for doc_id, doc in zip(self.doc_ids, self.documents):
                if doc == r["text"]:
                    vector_ranked_ids.append(doc_id)
                    break

        # --- BM25 candidates ---
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_pool]
        bm25_ranked_ids = [self.doc_ids[i] for i in bm25_ranked]

        # --- reciprocal rank fusion ---
        rrf_scores = {}
        for rank, doc_id in enumerate(vector_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(bm25_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

        top_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        results = []
        for doc_id in top_ids:
            idx = id_to_idx[doc_id]
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "distance": None,  # RRF score replaces distance; not directly comparable
            })
        return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_search.py \"your question\"")
        sys.exit(1)

    vs = VectorStore()
    searcher = HybridSearcher(vs)
    results = searcher.search(sys.argv[1])
    for r in results:
        print(f"[page {r['metadata']['page_num']}] {r['text'][:150]}...\n")
