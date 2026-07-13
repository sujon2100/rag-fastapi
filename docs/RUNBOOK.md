# Runbook: local testing and what broke along the way

## Part 1: local, no Docker

### Prerequisites

- Python 3.11 (`python3.11 --version`)
- Ollama installed and running (`ollama serve`, or the menu-bar app)
- A real Pinecone API key (free tier is fine) - `PINECONE_API_KEY=your_key_here`
  in `.env` is a placeholder and will fail fast with
  `PineconeConfigurationError` if left as-is

### Setup

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit in a real PINECONE_API_KEY
ollama pull llama3       # used by /rag/query
ollama pull qwen2.5:1.5b # used by the agent - see "llama3 doesn't do tools" below
```

### Run it

```bash
# terminal 1: the MCP tool server
python -m app.mcp.server

# terminal 2: the FastAPI app
uvicorn app.main:app --reload
```

### Smoke test

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready   # actually checks Pinecone + Ollama

curl -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "anything"}'

curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "ingest a document with source test and text: the sky is blue", "role": "ingest_agent"}'

curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "search the documents for sky", "role": "reader"}'
```

### Tests

```bash
pytest tests/                     # authz + tool-routing logic, no external deps, ~15s
pytest tests/ --run-integration   # spins up a real MCP server subprocess,
                                   # hits real Pinecone and real Ollama
```

## Part 2: Docker Compose

```bash
cp .env.example .env   # real PINECONE_API_KEY
docker compose up -d ollama
docker exec -it $(docker compose ps -q ollama) ollama pull llama3
docker exec -it $(docker compose ps -q ollama) ollama pull qwen2.5:1.5b
docker compose up -d mcp-server api
```

First start of `api` and `mcp-server` is slow - both import `torch` and
`sentence-transformers` at module load time (for the embedder), and on
Docker Desktop for Mac that took 2.5-5 minutes cold in testing here, versus
about 20 seconds for the same import running directly on the host. That
gap is almost certainly Docker Desktop's macOS virtualization layer being
slow to cold-read the large `.so` files those packages ship - a plain
Linux host (which is what the actual cloud deployment target is) should
not have the same penalty, but it hasn't been measured there yet. Poll
`/health/live` rather than assuming a fixed wait:

```bash
until curl -sf http://localhost:8000/health/live >/dev/null; do sleep 5; done
```

### Tear down

```bash
docker compose down          # containers + network, keeps the ollama volume
docker compose down -v       # also wipes the pulled models
```

## Issues found and fixed while building this

Worth keeping in the record rather than glossing over.

1. **`.env` was never actually loaded.** `app/services/vectorstore.py` read
   `PINECONE_API_KEY` with plain `os.getenv()` - there was no
   `load_dotenv()` call anywhere in the codebase. The key in `.env` was
   silently ignored; `os.getenv("PINECONE_API_KEY")` returned `None`, not
   even the placeholder string. Fixed by adding `load_dotenv()` to the top
   of `vectorstore.py`.

2. **The `.gitignore` had never actually worked.** The tracked file was
   literally named `.gitignore  ` - two trailing spaces baked into the
   filename from whenever it was first created. Git only recognizes a file
   named exactly `.gitignore`; `git check-ignore` confirmed nothing was
   being ignored at all. `.env` had been tracked in git since the very
   first commit as a result (checked the full history - only the
   placeholder key was ever committed, no real secret). Fixed by renaming
   the file and running `git rm --cached .env`.

3. **Hardcoded `localhost:11434` for Ollama.** `app/services/llm_inference.py`
   had `OLLAMA_URL = "http://localhost:11434/api/generate"` hardcoded,
   which resolves fine on a host machine but not inside a Docker container
   where Ollama runs as a separate service (`ollama`, not `localhost`).
   Fixed by reading `OLLAMA_BASE_URL` from the environment, same variable
   name that was already sitting unused in `.env`.

4. **`llama3` does not support Ollama's tool-calling API.** The agent's
   first real run against real Ollama failed with
   `ollama._types.ResponseError: registry.ollama.ai/library/llama3:latest
   does not support tools (status code: 400)`. Not every model on Ollama
   implements the tools/function-calling API; `llama3` (the base model)
   doesn't. Switched the agent specifically to `qwen2.5`, which does,
   while leaving `/rag/query`'s plain-generation path on `llama3`
   unchanged since it never calls tools.

5. **`qwen2.5:0.5b` was too small to reliably fill in both required fields**
   on the two-argument `ingest_document` tool call, in testing here - it
   would sometimes call the tool with a missing or malformed argument. The
   underlying tool and MCP wiring were confirmed correct by calling them
   directly, bypassing the LLM. `qwen2.5:1.5b` handled the same prompts
   correctly in every test run performed. This is a real, honest limit of
   running small models for tool use on CPU, not a code bug - later
   testing also saw the 1.5b model occasionally answer a search question
   without actually calling `search_documents`, so this is a "mostly
   reliable," not "always reliable," part of the system, and that's worth
   stating plainly rather than papering over.

6. **Docker: PID 1 zombie process, container couldn't be stopped or
   recreated.** Without an init process, `uvicorn` as PID 1 didn't reap
   child processes correctly on shutdown, leaving a zombie that
   `docker stop`/`docker kill` couldn't touch (`cannot kill container:
   ... PID ... is zombie`). Needed `docker rm -f` to clear it out. Fixed
   by adding `init: true` to both the `api` and `mcp-server` services in
   `docker-compose.yml`, which runs `tini` as PID 1 instead.

7. **MCP tool results are a list of content blocks, not a plain string.**
   `ainvoke()` on a `langchain_mcp_adapters` tool returns something like
   `[{"type": "text", "text": "...", "id": "..."}]`, not `str`. The first
   version of both the agent's `tools_node` and the integration test
   assumed a plain string and either fed the LLM a noisy Python repr of
   that list (which likely made the small-model reliability problem in
   item 5 worse) or failed a test assertion outright. Fixed in both
   places by extracting the actual text out of the content blocks before
   using it.

## Part 3: cloud deployment

Not yet deployed. Planned target: a new GCP Compute Engine VM in the same
`ai-platform-eb2-demo` project used for the `ai-platform-builder` deployment,
kept as a separate VM so the two stay independent. This section will be
filled in with the same level of detail as `ai-platform-builder`'s runbook
(provisioning commands, monitoring setup, synthetic traffic) once that
deployment actually happens.
