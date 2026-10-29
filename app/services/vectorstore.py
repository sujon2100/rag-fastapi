"""
Pinecone access, connected lazily on first use.

This used to connect (and create the index if missing) at import time, which
meant nothing could import the app - not even unit tests - without a real
PINECONE_API_KEY, and importing a module could create a cloud resource as a
side effect. Index creation is now an explicit call (ensure_index) that only
setup steps use.

PINECONE_NAMESPACE selects the namespace. Empty means the default namespace,
which is what the deployed service has always used. The eval corpus lives in
its own namespace (eval-v1) so eval runs never mix with anything else.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-demo-st")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


# Results scoring below this (cosine, all-MiniLM-L6-v2) are dropped before
# anything sees them. Picked from the 40-case golden set: answerable questions'
# best match scored >= 0.548, 7 of 8 unanswerable ones <= 0.535, so 0.54 sits
# in that gap. The gap is narrow and was chosen on the same set it is scored
# on, which is why evals/holdout.jsonl exists. 0 turns the filter off.
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.54"))


def relevant_matches(results, min_score: float | None = None) -> list:
    """
    All matches if the best one clears the floor, otherwise none. The floor
    gates the query as a whole (is anything here relevant?), not each chunk:
    lower-ranked chunks can still hold the answer - lg-02's answer chunk
    scored 0.483 behind a 0.548 top match.
    """
    floor = SEARCH_MIN_SCORE if min_score is None else min_score
    matches = results.get("matches", [])
    best = max((m.get("score") or 0.0 for m in matches), default=0.0)
    return matches if best >= floor else []


def get_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "")


@lru_cache(maxsize=1)
def _client():
    from pinecone import Pinecone

    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


@lru_cache(maxsize=1)
def get_index():
    return _client().Index(INDEX_NAME)


def ensure_index():
    """Create the index if it does not exist. Setup only, never on import."""
    from pinecone import ServerlessSpec

    pc = _client()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def upsert_vectors(vectors, namespace: str | None = None):
    ns = get_namespace() if namespace is None else namespace
    return get_index().upsert(vectors=vectors, namespace=ns)


def query_vectors(query_embedding, top_k=5, namespace: str | None = None):
    ns = get_namespace() if namespace is None else namespace
    return get_index().query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=ns,
    )
