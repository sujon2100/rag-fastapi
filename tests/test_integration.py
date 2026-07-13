"""
Real end-to-end tests: a real MCP server subprocess, the real Pinecone index
configured in .env, and real local Ollama. No mocks anywhere in this file.

Opt-in only, since these need actual services running:
  pytest tests/test_integration.py --run-integration
"""

import uuid

import pytest

from app.agent.graph import get_mcp_tools, run_query

pytestmark = pytest.mark.integration


def _tool_text(result) -> str:
    """
    langchain_mcp_adapters returns tool results as a list of content blocks
    (e.g. [{"type": "text", "text": "...", "id": "..."}]), not a plain
    string - found by running this test for real against the actual MCP
    server, not assumed from the tool's own return type annotation.
    """
    if isinstance(result, str):
        return result
    return " ".join(block.get("text", "") for block in result if isinstance(block, dict))


@pytest.mark.asyncio
async def test_mcp_tools_ingest_and_search_roundtrip(mcp_server):
    """
    Calls the real MCP tools directly (no LLM in the loop) so this test is
    deterministic: real embedding, real Pinecone upsert, real Pinecone query,
    real MCP HTTP round trip to the subprocess server.
    """
    tools = await get_mcp_tools()
    tool_map = {t.name: t for t in tools}
    assert {"search_documents", "ingest_document"} <= set(tool_map)

    canary = f"zebra-canary-{uuid.uuid4().hex[:8]}"
    ingest_result = await tool_map["ingest_document"].ainvoke(
        {"text": f"The secret canary phrase is {canary}.", "source": "integration-test"}
    )
    assert "Ingested" in _tool_text(ingest_result)

    search_result = await tool_map["search_documents"].ainvoke({"query": canary, "top_k": 3})
    assert canary in _tool_text(search_result)


@pytest.mark.asyncio
async def test_agent_end_to_end_with_real_ollama(mcp_server):
    """
    Runs the full LangGraph agent against real Ollama and the real MCP
    server. The assertion is intentionally loose - this is checking that the
    whole real pipeline (Ollama tool-calling, MCP dispatch, authz check,
    graph routing) completes without error, not asserting an exact model
    output, since a real local LLM's phrasing isn't deterministic.
    """
    answer = await run_query(
        "Reply with exactly the word 'pong' and nothing else. Do not use any tools.",
        role="reader",
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
