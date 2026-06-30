import re

def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+", text)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Group sentences into ~chunk_size-character chunks, never splitting mid-sentence."""
    sentences = split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # carry the tail of the previous chunk forward so context isn't lost at the boundary
            overlap_text = current[-overlap:] if overlap and current else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    print(f"[CHUNK] {len(sentences)} sentence(s) -> {len(chunks)} chunk(s) "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    for i, chunk in enumerate(chunks):
        preview = chunk[:60] + ("..." if len(chunk) > 60 else "")
        print(f"[CHUNK]   chunk {i}: {len(chunk)} chars -> \"{preview}\"")

    return chunks
