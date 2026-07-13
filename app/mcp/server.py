"""
MCP server exposing the RAG service's retrieval and ingestion as tools.

Wraps the same Pinecone-backed vector store and embedder already used by
app/services/llm_inference.py and app/services/vectorstore.py - no mock
tools, no toy demo data. A separate process from the FastAPI app (same
pattern as any other MCP tool server: agents connect to it over HTTP).

Run standalone:
  python -m app.mcp.server
"""

import os
import uuid

from mcp.server.fastmcp import FastMCP

from app.services.llm_inference import embedder
from app.services.vectorstore import query_vectors, upsert_vectors

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))

server = FastMCP("rag-fastapi-tools", host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)


@server.tool()
def search_documents(query: str, top_k: int = 3) -> str:
    """
    Search the indexed document store for chunks relevant to a query.
    Use this to retrieve grounding context before answering a question
    about previously ingested documents.
    """
    query_embedding = embedder.encode(query).tolist()
    results = query_vectors(query_embedding, top_k=top_k)
    matches = results.get("matches", [])

    if not matches:
        return "No relevant documents found."

    lines = []
    for match in matches:
        metadata = match.get("metadata", {}) or {}
        text = metadata.get("text", "")
        source = metadata.get("source", "unknown")
        score = match.get("score", 0.0)
        lines.append(f"[{source}, score={score:.4f}] {text}")
    return "\n".join(lines)


@server.tool()
def ingest_document(text: str, source: str) -> str:
    """
    Embed a piece of text and store it in the document index under the
    given source label, making it retrievable by search_documents.
    """
    if not text.strip():
        return "Error: text is empty, nothing to ingest."

    doc_id = str(uuid.uuid4())
    vector = embedder.encode(text).tolist()
    upsert_vectors(
        [
            {
                "id": doc_id,
                "values": vector,
                "metadata": {"text": text, "source": source},
            }
        ]
    )
    return f"Ingested document '{source}' as id={doc_id}."


if __name__ == "__main__":
    server.run(transport="streamable-http")
