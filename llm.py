import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the answer isn't contained in the context, say "I don't have enough information to answer that."
Always cite the source filename(s) you used at the end of your answer.
"""

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['metadata']['source']}]\n{c['chunk']}"
        for c in retrieved_chunks
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    print(f"[LLM] built prompt from {len(retrieved_chunks)} retrieved chunk(s) -> {len(prompt)} char(s)")
    return prompt

def generate_answer(query: str, retrieved_chunks: list[dict], model: str = "llama-3.3-70b-versatile") -> str:
    prompt = build_prompt(query, retrieved_chunks)

    print(f"[LLM] sending request to Groq (model={model})...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content
    print(f"[LLM] received answer ({len(answer)} char(s))")
    return answer
