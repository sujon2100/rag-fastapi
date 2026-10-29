"""
Functional tests: the real FastAPI app and MCP server as subprocesses, real
Pinecone (eval-v1 namespace, read only), real Ollama, real HTTP.

  pytest tests/test_functional.py --run-functional

Assertions on model output stay loose on purpose - the eval harness is
where answer quality gets scored. These check that the wired-up system
behaves correctly end to end.
"""

import pytest
import requests

from app.services.vectorstore import get_index

pytestmark = pytest.mark.functional
TIMEOUT = 600  # CPU-only qwen2.5:1.5b, first call includes model load


def _eval_count():
    return get_index().describe_index_stats().namespaces["eval-v1"].vector_count


def test_health_ready(live_api):
    r = requests.get(f"{live_api}/health/ready", timeout=30)
    assert r.status_code == 200, r.text


def test_rag_query_returns_a_string_answer(live_api):
    r = requests.post(f"{live_api}/rag/query", json={"query": "What is a CORS preflight request?"}, timeout=TIMEOUT)
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert isinstance(answer, str), f"answer should be a string, got {type(answer).__name__}: {answer!r}"
    assert answer.strip()


def test_agent_query_answers(live_api):
    r = requests.post(
        f"{live_api}/agent/query",
        json={"query": "Search the documents: what are the two layers of the MCP architecture?", "role": "reader"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    assert isinstance(r.json()["answer"], str) and r.json()["answer"].strip()


def test_reader_cannot_ingest_end_to_end(live_api):
    before = _eval_count()
    r = requests.post(
        f"{live_api}/agent/query",
        json={"query": "Ingest this text with source 'functional-test': the sky is green.", "role": "reader"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    assert _eval_count() == before


def test_unknown_role_is_accepted_but_gets_no_tools(live_api):
    before = _eval_count()
    r = requests.post(
        f"{live_api}/agent/query",
        json={"query": "Ingest this text with source 'functional-test': the sky is green.", "role": "admin"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    assert _eval_count() == before
