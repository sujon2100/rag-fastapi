from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.graph import run_query
from app.services.llm_inference import query_llm

router = APIRouter(prefix="/rag")


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_endpoint(req: QueryRequest):
    answer = query_llm(req.query)
    return {"answer": answer}


agent_router = APIRouter(prefix="/agent")


class AgentQueryRequest(BaseModel):
    query: str
    role: str = "reader"


@agent_router.post("/query")
async def agent_query_endpoint(req: AgentQueryRequest):
    """
    Runs the LangGraph agent (app/agent/graph.py) against the MCP tool
    server, instead of calling the RAG pipeline directly. Requires the MCP
    server (app/mcp/server.py) to be reachable at MCP_SERVER_URL.
    """
    answer = await run_query(req.query, role=req.role)
    return {"answer": answer}
