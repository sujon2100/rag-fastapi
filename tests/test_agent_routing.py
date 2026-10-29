"""
Unit tests for the LangGraph agent's tool-dispatch and authz wiring, using
stub tools and a stub LLM so nothing here touches a real MCP server or
Ollama. This is the piece that would fail silently if the authz check ever
got removed from tools_node - it does not test check_tool_authorized in
isolation (see test_authz.py), it tests that build_agent actually calls it.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent


class StubTool:
    def __init__(self, name, response):
        self.name = name
        self._response = response
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self._response


class StubLLM:
    """Returns a scripted sequence of AIMessages, ignoring the actual prompt."""

    def __init__(self, script):
        self._script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._script.pop(0)

    async def ainvoke(self, messages):
        return self.invoke(messages)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_authorized_tool_call_reaches_the_tool():
    search_tool = StubTool("search_documents", "found it")
    llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search_documents", "args": {"query": "gdpr"}}],
            ),
            AIMessage(content="final answer"),
        ]
    )
    app = build_agent([search_tool], llm)

    result = _run(
        app.ainvoke(
            {"messages": [HumanMessage(content="q")], "role": "reader"},
            config={"configurable": {"thread_id": "t1"}},
        )
    )

    assert search_tool.calls == [{"query": "gdpr"}]
    assert result["messages"][-1].content == "final answer"


def test_unauthorized_tool_call_is_denied_not_executed():
    ingest_tool = StubTool("ingest_document", "should never see this")
    llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "ingest_document", "args": {"text": "x", "source": "y"}}],
            ),
            AIMessage(content="final answer"),
        ]
    )
    app = build_agent([ingest_tool], llm)

    result = _run(
        app.ainvoke(
            {"messages": [HumanMessage(content="q")], "role": "reader"},
            config={"configurable": {"thread_id": "t2"}},
        )
    )

    # reader role is not on the ingest_document allowlist - the tool must
    # never actually run
    assert ingest_tool.calls == []
    tool_message = result["messages"][-2]
    assert "Denied" in tool_message.content


def test_forced_search_runs_before_the_model_with_the_question():
    search_tool = StubTool("search_documents", "[doc#1, score=0.9] text")
    llm = StubLLM([AIMessage(content="answer from doc#1")])
    app = build_agent([search_tool], llm, force_search=True)

    result = _run(
        app.ainvoke(
            {"messages": [HumanMessage(content="what is a checkpointer?")], "role": "reader"},
            config={"configurable": {"thread_id": "t3"}},
        )
    )

    assert search_tool.calls == [{"query": "what is a checkpointer?"}]
    # the model's single scripted turn came after the tool result
    assert result["messages"][-2].content == "[doc#1, score=0.9] text"
    assert result["messages"][-1].content == "answer from doc#1"


def test_forced_search_still_goes_through_authz():
    search_tool = StubTool("search_documents", "should never see this")
    llm = StubLLM([AIMessage(content="final")])
    app = build_agent([search_tool], llm, force_search=True)

    result = _run(
        app.ainvoke(
            {"messages": [HumanMessage(content="q")], "role": "unknown_role"},
            config={"configurable": {"thread_id": "t4"}},
        )
    )

    assert search_tool.calls == []
    assert result["messages"][-2].content.startswith("Denied")
