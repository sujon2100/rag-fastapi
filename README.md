The Architecture

#Architecture Description: 
-FastAPI: handles incoming HTTP requests
-VectorStore module: connects to Pinecone
-LLM interfence module: calls OpenAI or Local LLM
-Docker container: reproducible dev environment
-Architecture Diagram: User-->FastAPI-->VectorStore-->LLM-->Response

Think of your system as a conversation between four specialists, each doing only one job.
A user sends a question to your FastAPI endpoint.
FastAPI itself does not think. It only coordinates. Its job is to move data between the right components in the right order.
The first thing FastAPI does is not call Llama.
Instead, it sends the user's question to a sentence-transformer embedding model (all-MiniLM-L6-v2).
This model converts the question into a 384-dimensional vector.
At this point, the question has no words anymore — only meaning.
That vector is sent to Pinecone.
Pinecone is your system's long-term semantic memory.
It stores embeddings of documents you previously uploaded (FastAPI notes, Pinecone explanations, RAG concepts, etc.).
It compares the query vector against stored vectors and returns the most semantically similar text chunks.
If Pinecone finds matches, it returns the raw text associated with those vectors.
If it finds nothing, the system stops early and refuses to hallucinate.
Only after relevant context is found does Llama get involved.
Llama is running locally on your machine via Ollama.
It is not queried directly by the user.
It is given a carefully constructed prompt that includes:
the retrieved context from Pinecone
the user's original question
an instruction to answer only using the provided context
Llama's role is not memory.
Llama's role is reasoning and explanation.
The final answer is then returned to the user through FastAPI.

Flow:
Client (Postman / UI)
        |
        v
FastAPI (/rag/query)
        |
        v
Query Embedding (SentenceTransformers)
        |
        v
Vector Search (Pinecone)
        |
        v
Context Injection
        |
        v
LLM Generation (Ollama - Llama 3)
        |
        v
Final Answer (JSON)

                      ┌──────────────────────┐
                      │      Frontend /      │
                      │   Query Client UI     │
                      └──────────▲───────────┘
                                 │ HTTP POST
                                 ▼
                     ┌──────────────────────────┐
                     │       FastAPI Backend     │
                     │ (app/main, routes, etc.) │
                     └──────────▲───────────────┘
                                 │ Calls
 ┌───────────────────────────────┴────────────────────────────┐
 │                     RAG Pipeline Components                │
 │                                                            │
 │   ┌─────────────────┐      ┌──────────────────────────┐    │
 │   │   Vector Store   │      │    Local LLM (Ollama)     │  │
 │   │   Pinecone DB    │◀────▶│   Llama3 / Model API     │   │
 │   │  (384 dim index) │      │  (localhost network)      │  │
 │   └─────────────────┘      └──────────────────────────┘     │
 │             ▲                             ▲                 │
 │  embed query│                             │ generate answer│
 │             │                             │                 │
 └────────────────────────────────────────────────────────────┘

## What's been added since: MCP + a LangGraph agent

On top of the RAG pipeline described above, this repo now also has:

- `app/mcp/server.py` - an MCP server (built on the official `mcp` Python
  SDK's `FastMCP` helper, not the third-party `fastmcp` package - see
  `docs/RUNBOOK.md` for why) exposing two tools, `search_documents` and
  `ingest_document`, that call straight into the same Pinecone-backed
  vector store the RAG endpoint above uses. No mock tools, no toy demo
  data - these are the real retrieval and ingestion paths.
- `app/agent/graph.py` - a LangGraph agent (`StateGraph`, conditional
  routing between an `agent` node and a `tools` node) that connects to that
  MCP server over HTTP and uses a local Ollama model to decide when to call
  which tool. Uses `qwen2.5:1.5b`, not `llama3` - `llama3` doesn't support
  Ollama's tool-calling API at all (see RUNBOOK).
- `app/agent/authz.py` - a small static allowlist checked before every tool
  call: which roles may call which tools. `reader` can only search;
  `ingest_agent` can search and ingest. Unknown roles get nothing. This is
  a deliberately small echo of the access-control pattern from other work
  of mine, not a policy engine.

Two API surfaces exist side by side: `POST /rag/query` (the original direct
RAG path described above - no agent involved) and `POST /agent/query`
(routes through the LangGraph agent and the MCP tools, takes a `role`
field).

## Running it locally

### Without Docker

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in a real PINECONE_API_KEY
ollama pull llama3
ollama pull qwen2.5:1.5b

# terminal 1
python -m app.mcp.server

# terminal 2
uvicorn app.main:app --reload

# terminal 3
curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "search the documents for X", "role": "reader"}'
```

### With Docker Compose

```bash
cp .env.example .env   # fill in a real PINECONE_API_KEY
docker compose up -d ollama
docker exec -it $(docker compose ps -q ollama) ollama pull llama3
docker exec -it $(docker compose ps -q ollama) ollama pull qwen2.5:1.5b
docker compose up -d mcp-server api
```

First start is slow - see `docs/RUNBOOK.md` for why and what to expect.

## Tests

```bash
pytest tests/                        # fast, no external services needed
pytest tests/ --run-integration      # real Pinecone + real Ollama + a real
                                      # MCP server subprocess, opt-in only
```

## More detail

`docs/RUNBOOK.md` has the local/Docker setup in more depth, the deployment
notes, and an honest list of the real bugs hit while building this (a
non-functional `.gitignore`, a hardcoded Ollama URL that broke under Docker
networking, a model that silently doesn't support tool-calling, a Docker
zombie-process issue, and others).
