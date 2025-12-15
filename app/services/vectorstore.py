import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

INDEX_NAME = "rag-demo-st"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

existing = pc.list_indexes().names()

if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)


def upsert_vectors(vectors):
    return index.upsert(vectors=vectors)


def query_vectors(query_embedding, top_k=5):
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
