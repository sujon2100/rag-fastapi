import requests
from sentence_transformers import SentenceTransformer
from app.services.vectorstore import query_vectors

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


def generate_with_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def query_llm(user_query: str):
    # 1. Embed query
    query_embedding = embedder.encode(user_query).tolist()

    # 2. Retrieve context
    results = query_vectors(query_embedding, top_k=3)
    matches = results.get("matches", [])

    if not matches:
        return {"answer": "No relevant documents found."}

    context = " ".join(
        m["metadata"]["text"]
        for m in matches
        if "metadata" in m and "text" in m["metadata"]
    )

    # 3. RAG prompt
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{user_query}

Answer:
"""

    # 4. Generate answer
    answer = generate_with_ollama(prompt)

    return {"answer": answer.strip()}
