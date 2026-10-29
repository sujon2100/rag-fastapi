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
  which tool. Uses `qwen2.5:1.5b` - `llama3` doesn't support Ollama's
  tool-calling API at all, and running two different models risked
  exceeding a small deployment VM's memory, so `/rag/query` was switched
  to `qwen2.5:1.5b` too (see RUNBOOK).
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
docker exec -it $(docker compose ps -q ollama) ollama pull qwen2.5:1.5b
docker compose up -d mcp-server api
```

First start is slow - see `docs/RUNBOOK.md` for why and what to expect.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/                          # unit tests, no services, ~10s
pytest tests/ --cov=app --cov=evals    # same, with coverage
pytest tests/ --run-functional         # real API + MCP server over HTTP (Pinecone, Ollama)
pytest tests/ --run-integration        # real MCP tools, writes to namespace integration-test
```

Unit tests use fakes for Pinecone, Ollama and the agent: authz, tool routing,
forced search, chunking, the MCP tools, the API layer, the eval's scoring
rules and the golden set itself. One test is a strict `xfail` for a known bug
(`/rag/query` returns `{"answer": {"answer": ...}}`), so it fails loudly once
the bug is fixed and the marker should come off.

## Evaluation

`evals/` holds an eval harness for the agent. Details in `evals/README.md`.

- **Corpus:** 12 public docs pages (FastAPI, LangGraph, MCP; MIT, CC-BY-4.0 and
  Apache-2.0, pinned to upstream commits), cleaned and chunked into 236 chunks
  in their own Pinecone namespace, `eval-v1`.
- **Golden set:** 40 questions: 26 answerable, 4 that need two documents, 8 the
  documents don't cover, and 2 where a read-only role is asked to ingest.
- **Held-out set:** 12 more questions (`evals/holdout.jsonl`), written after the
  search cutoff was picked and never used to tune anything.
- **Measured:** whether the agent searched, whether search found the right
  chunk, groundedness and key-fact coverage (LLM judge), refusals, latency.
  Each run writes one JSON file to `evals/results/`.

```bash
python -m scripts.ingest_corpus      # once
python -m evals.run_eval             # retrieval + agent + judge, current defaults
python -m evals.run_eval --grounded-prompt 0 --force-search 0   # the original agent
```

### Results

Agent `qwen2.5:1.5b` at temperature 0, judge `gemma3:4b`, both through Ollama
on a 2020 Intel i5 laptop (16 GB RAM, no GPU). One run per configuration.

| Metric | Original agent | + grounded prompt | + prompt and forced search |
|---|---|---|---|
| Searched before answering (30 answerable) | 0.00 | 0.00 | 1.00 |
| Search found an expected chunk (top 3) | 0.87 ¹ | 0.87 ¹ | 0.87 |
| Groundedness, fully supported (judge) | 0.00 ² | 0.00 ² | 0.53 |
| Groundedness, supported or partial (judge) | 0.00 ² | 0.00 ² | 0.93 |
| Key-fact coverage (judge) | 0.61 | 0.59 | 0.89 |
| Refused on unanswerable (8) | 2 of 8 | 0 of 8 | 2 of 8 |
| Refused an answerable question | 1 of 29 | 0 of 30 | 0 of 30 |
| Latency p50 / p95, seconds | 13.5 / 32.5 | 12.7 / 44.2 | 10.4 / 24.1 |
| Errors | 0 | 0 | 0 |

¹ Retriever called directly with the question; the agent itself never searched.
² Set by rule: nothing was retrieved, so nothing can be supported.

Result files: `evals/results/2026-10-29_131139_judged.json`,
`2026-10-29_135349_fixA-prompt_all.json`, `2026-10-29_143035_fixB-prompt-forced_all.json`.

The search cutoff (`SEARCH_MIN_SCORE`, point 4 below) is on by default but
its eval runs, golden and held-out, are not finished yet; its numbers will be
added here when they are.

### What the eval found, and what changed because of it

1. **The agent never searched.** In the original code it answered all 30
   answerable questions from the model's memory: 0 searches. The answers were
   confident and often invented: a `Response.after` method that doesn't
   exist, MCP described as "Maven Central Repository", "Multi-Cloud Platform"
   and an Azure service, FastAPI's body limit given as "16MB". Earlier smoke
   tests hid this because they all said "search the documents for X".
2. **A system prompt didn't fix it.** "Before answering any question, call
   search_documents" changed nothing with this model: still 0 of 30.
3. **Forcing the search did.** A `retrieve` step now runs `search_documents`
   on the question before the model's first turn, through the same authz
   check as any other tool call. On by default (`AGENT_FORCE_SEARCH`,
   `AGENT_GROUNDED_PROMPT`).
4. **Refusals were weak even with search.** With the documents in front of
   it, the model still stretched unrelated chunks into answers ("use a
   middleware for rate limiting"). Search results are now dropped when the best
   match scores under 0.54 (`SEARCH_MIN_SCORE`), so the model sees "No relevant
   documents found". The cutoff was picked from the golden set's scores, so it
   was also checked on the held-out set (see above).
5. **Runaway generation.** One question made the model generate 7,400+ tokens
   (about 10 minutes of CPU) and hung the first eval run. Replies are now
   capped at 1,024 tokens (`AGENT_MAX_TOKENS`), and the agent's model call is
   async, so a timeout actually cancels it. This affected the live service too.

### How far to trust the judge

`gemma3:4b` is small, and I checked it against hand labels (`evals/labels/`):

- **Refusals:** it agreed with hand labels on 19 of 37. It often wrote a
  reason saying the reply answers the question and then flagged it as a
  refusal. A plain regex agreed on 35 of 37, so refusals are scored with the
  regex and the judge's answer is only recorded.
- **Groundedness:** 10 of 17 agreed. 4 times it was too lenient (including
  one invented code example it called supported), 3 times too strict. On
  those 17 the judge and the hand labels both come out at about half fully
  supported, but individual verdicts are noisy, so read groundedness as a
  rough number.
- **No context:** when nothing was retrieved, groundedness is set by rule to
  "unsupported" instead of asking the judge. The judge had called such answers
  "partial" or "supported" from its own knowledge.

The first plan was `gpt-oss:20b` as judge. On this 16 GB laptop it pushed
swap to 24 of 25.6 GB, and a single call ran over 10 minutes without
finishing, so it was dropped.

### Limits

- One run per configuration at temperature 0. Repeats (`--repeats N`) were
  not run, so there's no variance estimate.
- 40 + 12 questions over 12 pages is small. A few cases move a metric by
  several points.
- The cutoff gap on the golden set is narrow (0.535 vs 0.548), and one
  unanswerable question (GraphQL streaming) scores 0.69 because it sounds
  like the streaming docs, so no cutoff can catch it.
- Latency is from one CPU-only laptop, not the deployed VM.

## More detail

`docs/RUNBOOK.md` has the local/Docker setup in more depth, the deployment
notes, and an honest list of the real bugs hit while building this (a
non-functional `.gitignore`, a hardcoded Ollama URL that broke under Docker
networking, a model that silently doesn't support tool-calling, a Docker
zombie-process issue, and others).
