import uuid
import numpy as np
import chromadb

class VectorStore:
    """A persistent vector store backed by ChromaDB (cosine similarity)."""

    def __init__(self, persist_dir: str, collection_name: str = "rag_documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VSTORE] connected to collection '{collection_name}' at {persist_dir} "
              f"({self.count()} chunk(s) already indexed)")

    def add(self, embeddings: np.ndarray, chunks: list[str], metadata: list[dict]) -> None:
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
        )
        print(f"[VSTORE] added {len(chunks)} chunk(s) -> collection now has {self.count()} chunk(s)")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        if self.count() == 0:
            print("[VSTORE] search skipped — collection is empty")
            return []

        n_results = min(top_k, self.count())
        print(f"[VSTORE] searching top_k={n_results} of {self.count()} chunk(s)...")
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        retrieved = [
            {"chunk": doc, "metadata": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
        for i, r in enumerate(retrieved):
            preview = r["chunk"][:60] + ("..." if len(r["chunk"]) > 60 else "")
            print(f"[VSTORE]   match {i}: score={r['score']:.4f} "
                  f"source={r['metadata']['source']} chunk_index={r['metadata']['chunk_index']} \"{preview}\"")

        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VSTORE] collection '{self.collection_name}' reset")
