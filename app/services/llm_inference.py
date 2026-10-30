import os

import requests
from app.services.embeddings import embed
from app.services.vectorstore import query_vectors, relevant_matches

# Ollama config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


def generate_with_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def query_llm(user_query: str) -> str:
    # 1. Embed query
    query_embedding = embed(user_query)

    # 2. Retrieve context
    results = query_vectors(query_embedding, top_k=3)
    matches = relevant_matches(results)

    if not matches:
        return "No relevant documents found."

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

    # plain string; the route wraps it. Returning {"answer": ...} here made
    # /rag/query respond {"answer": {"answer": ...}} - caught by the
    # functional test against the running API.
    return answer.strip()
