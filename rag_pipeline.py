import os
from chunker import chunk_text
from loader import load_documents
from embeddings import embed_texts
from vector_store import VectorStore
from llm import generate_answer

DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

class RAGPipeline:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 3,
                 persist_dir: str = DEFAULT_PERSIST_DIR):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.store = VectorStore(persist_dir=persist_dir)

    def ingest(self, folder_path: str, force: bool = False) -> None:
        """Embed and index documents. Skips re-ingestion if the store is already populated
        (Chroma persists to disk), unless force=True."""
        if not force and self.store.count() > 0:
            print(f"[PIPELINE] vector store already has {self.store.count()} chunk(s) — skipping ingestion (pass force=True to rebuild).")
            return
        if force:
            self.store.reset()

        print(f"[PIPELINE] === INGESTION: loading documents from {folder_path} ===")
        documents = load_documents(folder_path)
        if not documents:
            raise ValueError(f"No .txt documents found in {folder_path}")
        print(f"[PIPELINE] loaded {len(documents)} document(s): {[d['source'] for d in documents]}")

        all_chunks, all_metadata = [], []
        for doc in documents:
            print(f"[PIPELINE] chunking '{doc['source']}'...")
            chunks = chunk_text(doc["text"], self.chunk_size, self.overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": doc["source"], "chunk_index": i})

        print(f"[PIPELINE] {len(documents)} document(s) -> {len(all_chunks)} chunk(s) total, embedding now...")
        embeddings = embed_texts(all_chunks)
        self.store.add(embeddings, all_chunks, all_metadata)
        print("[PIPELINE] === INGESTION COMPLETE ===")

    def query(self, question: str) -> str:
        if self.store.count() == 0:
            return "No documents have been ingested yet."

        print(f"[PIPELINE] === QUERY: \"{question}\" ===")
        query_embedding = embed_texts([question])[0]
        retrieved = self.store.search(query_embedding, top_k=self.top_k)
        answer = generate_answer(question, retrieved)
        print("[PIPELINE] === QUERY COMPLETE ===")
        return answer
