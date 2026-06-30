import os
import sys
from rag_pipeline import RAGPipeline

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")

def main():
    force = "--force" in sys.argv

    pipeline = RAGPipeline(chunk_size=500, overlap=50, top_k=3)
    pipeline.ingest(DOCS_FOLDER, force=force)

    print("\nRAG pipeline ready. Ask questions (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = pipeline.query(question)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()
