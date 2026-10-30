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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.authz import ToolNotAuthorized, check_tool_authorized

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5:1.5b")
# Hard cap on tokens per model reply. Without it qwen2.5:1.5b occasionally
# loops and keeps generating - one eval case reached 7,400+ tokens (~10 min
# of CPU) before it was killed. Applies to the live service too.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))

# The baseline eval (evals/results/2026-10-29_131139_*) showed the agent never
# calling search_documents on its own for a plain question - 0 of 30 - and
# answering from the model's memory instead, often wrongly. Two fixes, each
# switchable so the eval can measure them separately. Measured on the same 40
# cases: the prompt alone changed nothing (still 0/30 searches); prompt +
# forced search took groundedness from 0 to 0.53 (judge), so both are on by
# default. Set either to 0 to get the old behavior.
#   AGENT_GROUNDED_PROMPT  - system prompt: search first, answer only from results
#   AGENT_FORCE_SEARCH     - run search_documents on the question before the
#                            model's first turn, so searching is not optional
AGENT_GROUNDED_PROMPT = os.getenv("AGENT_GROUNDED_PROMPT", "1") == "1"
AGENT_FORCE_SEARCH = os.getenv("AGENT_FORCE_SEARCH", "1") == "1"

# With the search cutoff (SEARCH_MIN_SCORE) the forced search correctly came
# back "No relevant documents found." for 7 of 8 unanswerable eval questions,
# and the model still answered several of them from memory ("16 KB" for
# FastAPI's body limit). So when the forced search finds nothing, reply with
# a fixed refusal in code instead of asking the model.
# Measured: unanswerable questions refused went from 5/8 to 7/8 on the golden
# set, 4/5 on the held-out set, no answerable question refused. On by default.
#   AGENT_NO_CONTEXT_REFUSAL
AGENT_NO_CONTEXT_REFUSAL = os.getenv("AGENT_NO_CONTEXT_REFUSAL", "1") == "1"
NO_CONTEXT_REPLY = "The documents I can search do not cover this."
NO_RESULTS_PREFIX = "No relevant documents found"

GROUNDED_SYSTEM_PROMPT = """You answer questions using a document search tool, not your own memory.

Rules:
1. Before answering any question, call search_documents with a short search query.
2. Answer only with information found in the search results. Do not add facts from memory.
3. If the search results do not contain the answer, say: "The documents I can search do not cover this." Do not guess.
4. Keep answers short and mention which source the answer came from."""


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


def build_agent(tools: list, llm: ChatOllama, force_search: bool = False, no_context_refusal: bool = False):
    """
    Graph structure: START -> agent -> [tools | END] -> agent -> ...
    With force_search: START -> retrieve -> tools -> agent -> ...
    tool_map is used by tools_node to dispatch calls the LLM decided to make.
    The forced search goes through tools_node too, so it gets the same authz
    check as any other call (a role without search gets "Denied").
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        # async so a timeout around the run actually cancels the Ollama
        # request; with the sync invoke() it kept running in a worker thread
        response = await llm_with_tools.ainvoke(state["messages"])
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

    def retrieve_node(state: AgentState) -> dict:
        question = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
        call = {"name": "search_documents", "args": {"query": question}, "id": "forced-search"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    def refuse_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=NO_CONTEXT_REPLY)]}

    def after_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if (
            no_context_refusal
            and isinstance(last, ToolMessage)
            and last.tool_call_id == "forced-search"
            and last.content.startswith(NO_RESULTS_PREFIX)
        ):
            return "refuse"
        return "agent"

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    if force_search:
        graph.add_node("retrieve", retrieve_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "tools")
    else:
        graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_node("refuse", refuse_node)
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "refuse": "refuse"})
    graph.add_edge("refuse", END)

    return graph.compile(checkpointer=MemorySaver())


async def run_query_traced(
    question: str,
    role: str = "reader",
    thread_id: str = "default",
    temperature: float | None = None,
    llm_timeout: float | None = None,
    grounded_prompt: bool | None = None,
    force_search: bool | None = None,
    no_context_refusal: bool | None = None,
) -> dict:
    """
    Same run as run_query, but also returns what happened along the way:
    every tool call the model asked for and what each one returned. The eval
    harness needs this - the HTTP endpoint only returns the final answer.
    temperature=None keeps Ollama's default (what the service uses).
    llm_timeout is the HTTP timeout per Ollama call (a safety net; it does
    not catch a model that keeps streaming tokens - AGENT_MAX_TOKENS does).
    grounded_prompt / force_search default to the AGENT_* env settings.
    """
    grounded_prompt = AGENT_GROUNDED_PROMPT if grounded_prompt is None else grounded_prompt
    force_search = AGENT_FORCE_SEARCH if force_search is None else force_search
    no_context_refusal = AGENT_NO_CONTEXT_REFUSAL if no_context_refusal is None else no_context_refusal
    tools = await get_mcp_tools()
    llm_kwargs = {"num_predict": AGENT_MAX_TOKENS}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if llm_timeout is not None:
        llm_kwargs["client_kwargs"] = {"timeout": llm_timeout}
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, **llm_kwargs)
    app = build_agent(tools, llm, force_search=force_search, no_context_refusal=no_context_refusal)
    start = [SystemMessage(content=GROUNDED_SYSTEM_PROMPT)] if grounded_prompt else []
    result = await app.ainvoke(
        {"messages": start + [HumanMessage(content=question)], "role": role},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    messages = result["messages"]
    tool_calls = [
        {"name": tc["name"], "args": tc["args"], "forced": tc["id"] == "forced-search"}
        for m in messages
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    tool_results = [
        {"name": m.name, "content": m.content} for m in messages if isinstance(m, ToolMessage)
    ]
    return {"answer": messages[-1].content, "tool_calls": tool_calls, "tool_results": tool_results}


async def run_query(question: str, role: str = "reader", thread_id: str = "default") -> str:
    return (await run_query_traced(question, role=role, thread_id=thread_id))["answer"]


async def _main():
    answer = await run_query("Search the documents for information about GDPR.")
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
