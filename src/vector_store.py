"""
Module 5: Embedding + Vector Store

Embeds chunks using a local sentence-transformers model and stores them
(with metadata) in a persistent local ChromaDB collection. No accounts,
no API keys, everything lives on disk in data/chroma_db/.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME


class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def add_chunks(self, chunks: list):
        """
        Embeds and stores a list of chunk dicts (from chunking.py).
        Each chunk must have: id, text, source_file, page_num, chunk_type.
        """
        if not chunks:
            return

        source_files = {c["source_file"] for c in chunks}
        for source_file in source_files:
            self.collection.delete(where={"source_file": source_file})

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        ids = [c["id"] for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append({
                "source_file": c["source_file"],
                "page_num": c["page_num"],
                "chunk_type": c["chunk_type"],
                "heading": c.get("heading", ""),
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int = 4):
        """
        Embeds the query and returns the top_k most similar chunks.
        Returns a list of dicts: [{"text": ..., "metadata": {...}, "distance": ...}, ...]
        """
        query_embedding = self.embedder.encode([query_text]).tolist()
        result_count = min(top_k, self.collection.count())
        if result_count == 0:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=result_count,
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def count(self):
        return self.collection.count()


if __name__ == "__main__":
    vs = VectorStore()
    print(f"Chunks currently stored: {vs.count()}")
