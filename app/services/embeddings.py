"""
Sentence-transformer embedder, loaded on first use instead of at import.

Loading torch + the model takes ~20s on a laptop and minutes cold under
Docker Desktop (see RUNBOOK), and unit tests should not pay that.
"""

from functools import lru_cache

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> list[float]:
    return get_embedder().encode(text).tolist()
