import os

def load_documents(folder_path: str) -> list[dict]:
    """Read every .txt file in a folder. Returns [{"source": filename, "text": content}, ...]."""
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents.append({"source": filename, "text": f.read()})
    return documents
