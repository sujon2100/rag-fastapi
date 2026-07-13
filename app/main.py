import os

import requests
from fastapi import FastAPI, HTTPException
from app.routes import router
from app.services.vectorstore import index

app = FastAPI(title="RAG FastAPI Service")
app.include_router(router)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.api_route("/health/live", methods=["GET", "HEAD"])
def health_live():
    """Process is up and answering HTTP. Does not touch Pinecone or Ollama."""
    return {"status": "ok"}


@app.api_route("/health/ready", methods=["GET", "HEAD"])
def health_ready():
    """Checks the actual dependencies the /rag/query path needs."""
    try:
        index.describe_index_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"pinecone not ready: {exc}")

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ollama not ready: {exc}")

    return {"status": "ready"}
