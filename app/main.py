from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="RAG FastAPI Service")
app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "ok"}
