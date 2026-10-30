"""
API-layer tests with FastAPI's TestClient. Pinecone, Ollama and the agent
are replaced with fakes, so these check routing, request validation and
response shapes only.
"""

import pytest
import requests
from fastapi.testclient import TestClient

import app.main as main
import app.routes as routes


@pytest.fixture
def client():
    return TestClient(main.app)


class _FakeIndex:
    def __init__(self, fail=False):
        self.fail = fail

    def describe_index_stats(self):
        if self.fail:
            raise RuntimeError("pinecone down")
        return {}


class _FakeResp:
    def __init__(self, ok=True):
        self.ok = ok

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError("500")


def test_root_and_live(client):
    assert client.get("/").json() == {"status": "ok"}
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.head("/health/live").status_code == 200


def test_ready_ok(client, monkeypatch):
    monkeypatch.setattr(main, "get_index", lambda: _FakeIndex())
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _FakeResp())
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json() == {"status": "ready"}


def test_ready_503_when_pinecone_fails(client, monkeypatch):
    monkeypatch.setattr(main, "get_index", lambda: _FakeIndex(fail=True))
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _FakeResp())
    r = client.get("/health/ready")
    assert r.status_code == 503 and "pinecone" in r.json()["detail"]


def test_ready_503_when_ollama_fails(client, monkeypatch):
    monkeypatch.setattr(main, "get_index", lambda: _FakeIndex())

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(main.requests, "get", boom)
    r = client.get("/health/ready")
    assert r.status_code == 503 and "ollama" in r.json()["detail"]


def test_rag_query_requires_query(client):
    assert client.post("/rag/query", json={}).status_code == 422


def test_rag_query_response_shape(client, monkeypatch):
    monkeypatch.setattr(routes, "query_llm", lambda q: f"echo {q}")
    r = client.post("/rag/query", json={"query": "hi"})
    assert r.status_code == 200
    assert r.json() == {"answer": "echo hi"}


def test_agent_query_passes_role_and_defaults_to_reader(client, monkeypatch):
    seen = []

    async def fake_run_query(q, role="reader"):
        seen.append((q, role))
        return "agent says hi"

    monkeypatch.setattr(routes, "run_query", fake_run_query)
    assert client.post("/agent/query", json={"query": "a"}).json() == {"answer": "agent says hi"}
    client.post("/agent/query", json={"query": "b", "role": "ingest_agent"})
    assert seen == [("a", "reader"), ("b", "ingest_agent")]


def test_agent_query_requires_query(client):
    assert client.post("/agent/query", json={"role": "reader"}).status_code == 422
