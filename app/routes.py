from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_inference import query_llm

router = APIRouter(prefix="/rag")


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_endpoint(req: QueryRequest):
    answer = query_llm(req.query)
    return {"answer": answer}
