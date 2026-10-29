"""
Unit tests for the MCP tool functions in app/mcp/server.py, with the
embedder and Pinecone calls replaced by fakes. The tools are plain functions
under the @server.tool() decorator, so they can be called directly.
"""

import pytest

import app.mcp.server as server


@pytest.fixture
def fake_store(monkeypatch):
    monkeypatch.setattr("app.services.vectorstore.SEARCH_MIN_SCORE", 0.0)
    calls = {"query": [], "upsert": []}
    state = {"matches": []}

    monkeypatch.setattr(server, "embed", lambda text: [0.1] * 384)

    def fake_query(vec, top_k=5):
        calls["query"].append({"vec": vec, "top_k": top_k})
        return {"matches": state["matches"]}

    monkeypatch.setattr(server, "query_vectors", fake_query)
    monkeypatch.setattr(server, "upsert_vectors", lambda vectors: calls["upsert"].append(vectors))
    return calls, state


def test_search_no_matches(fake_store):
    assert server.search_documents("anything") == "No relevant documents found."


def test_search_formats_chunk_id_score_and_text(fake_store):
    calls, state = fake_store
    state["matches"] = [
        {"score": 0.71234, "metadata": {"text": "chunk text", "source": "doc", "chunk_id": "doc#2"}},
        {"score": 0.5, "metadata": {"text": "old vector", "source": "smoke-test"}},
    ]
    out = server.search_documents("q", top_k=2)
    assert out.splitlines() == ["[doc#2, score=0.7123] chunk text", "[smoke-test, score=0.5000] old vector"]
    assert calls["query"][0]["top_k"] == 2


def test_search_tolerates_missing_metadata(fake_store):
    _, state = fake_store
    state["matches"] = [{"score": 0.3}, {"score": 0.2, "metadata": None}]
    assert server.search_documents("q").splitlines() == [
        "[unknown, score=0.3000] ",
        "[unknown, score=0.2000] ",
    ]


def test_search_default_top_k_is_3(fake_store):
    calls, _ = fake_store
    server.search_documents("q")
    assert calls["query"][0]["top_k"] == 3


def test_ingest_rejects_blank_text(fake_store):
    calls, _ = fake_store
    assert server.ingest_document("   \n", "src").startswith("Error")
    assert calls["upsert"] == []


def test_ingest_upserts_one_vector_with_source(fake_store):
    calls, _ = fake_store
    out = server.ingest_document("some text", "notes")
    assert out.startswith("Ingested document 'notes' as id=")
    [vectors] = calls["upsert"]
    assert len(vectors) == 1
    assert vectors[0]["metadata"] == {"text": "some text", "source": "notes"}
    assert len(vectors[0]["values"]) == 384
    assert vectors[0]["id"] in out


def test_min_score_gates_on_the_best_match(fake_store, monkeypatch):
    _, state = fake_store
    monkeypatch.setattr("app.services.vectorstore.SEARCH_MIN_SCORE", 0.54)
    state["matches"] = [
        {"score": 0.70, "metadata": {"text": "keep", "chunk_id": "d#1"}},
        {"score": 0.48, "metadata": {"text": "also kept", "chunk_id": "d#2"}},
    ]
    # best match clears the floor, so lower-ranked chunks stay
    assert server.search_documents("q").splitlines() == [
        "[d#1, score=0.7000] keep", "[d#2, score=0.4800] also kept",
    ]
    state["matches"] = [{"score": 0.40, "metadata": {"text": "drop", "chunk_id": "d#3"}}]
    assert server.search_documents("q") == "No relevant documents found."
