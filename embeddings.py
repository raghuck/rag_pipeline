import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_model: SentenceTransformer | None = None

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[EMBED] loading model '{model_name}'...")
        _model = SentenceTransformer(model_name, token=os.getenv("HF_TOKEN") or None)
        print(f"[EMBED] model loaded (embedding dim={_model.get_sentence_embedding_dimension()})")
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts as L2-normalized vectors (so dot product == cosine similarity)."""
    model = get_embedding_model()

    print(f"[EMBED] tokenizing {len(texts)} text(s)...")
    for i, text in enumerate(texts):
        token_ids = model.tokenizer(text)["input_ids"]
        preview = token_ids[:10]
        suffix = "..." if len(token_ids) > 10 else ""
        print(f"[EMBED]   text {i}: {len(token_ids)} token id(s), ids={preview}{suffix}")

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"[EMBED] produced {embeddings.shape[0]} embedding(s) of dim {embeddings.shape[1]}")
    return embeddings
