"""
LangGraph agent that consumes tools from the MCP server in app/mcp/server.py,
using a local Ollama model for generation (no OpenAI key required, no
FakeChatModel). Every tool call is checked against the authz allowlist in
app/agent/authz.py before it runs.

Uses qwen2.5 - llama3 was tried first (it's what app/services/llm_inference.py
originally used for plain generation) but doesn't support Ollama's
tool-calling API at all (confirmed locally: "does not support tools",
HTTP 400). llama3 is also a 4.7GB model; running it alongside a second
model for the agent risked exceeding a small deployment VM's memory if
both got loaded into Ollama at once, so llm_inference.py was switched to
qwen2.5:1.5b too - one small model for everything now, not two.
qwen2.5:0.5b was tried first here (smaller, faster) but was unreliable at
extracting both required fields for the two-argument ingest_document tool
from a natural-language prompt; qwen2.5:1.5b handled the same prompts
correctly in testing, so that's the default.

Run standalone (MCP server must already be running on port 8001):
  python -m app.agent.graph
"""

import asyncio
import operator
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.authz import ToolNotAuthorized, check_tool_authorized

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5:1.5b")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    role: str


def _tool_result_to_text(result) -> str:
    """
    MCP tool results come back as a list of content blocks (e.g.
    [{"type": "text", "text": "...", "id": "..."}]), not a plain string -
    found by actually running this against the real MCP server. Feeding the
    raw Python repr of that list back to the LLM as a ToolMessage works but
    is noisy and measurably hurt a small model's ability to use the result.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return " ".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


async def get_mcp_tools(url: str = MCP_SERVER_URL) -> list:
    client = MultiServerMCPClient(
        {
            "rag-fastapi-tools": {
                "url": url,
                "transport": "streamable_http",
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list, llm: ChatOllama):
    """
    Graph structure: START -> agent -> [tools | END] -> agent -> ...
    tool_map is used by tools_node to dispatch calls the LLM decided to make.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        role = state.get("role", "reader")
        results = []

        for tool_call in last.tool_calls:
            name, args, call_id = tool_call["name"], tool_call["args"], tool_call["id"]
            try:
                check_tool_authorized(role, name)
                result = _tool_result_to_text(await tool_map[name].ainvoke(args))
            except ToolNotAuthorized as exc:
                result = f"Denied: {exc}"
            except Exception as exc:  # tool execution failure, not an authz issue
                result = f"Error calling tool '{name}': {exc}"
            results.append(ToolMessage(content=result, tool_call_id=call_id, name=name))

        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


async def run_query(question: str, role: str = "reader", thread_id: str = "default") -> str:
    tools = await get_mcp_tools()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    app = build_agent(tools, llm)
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=question)], "role": role},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    return result["messages"][-1].content


async def _main():
    answer = await run_query("Search the documents for information about GDPR.")
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
