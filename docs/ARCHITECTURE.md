# rag-fastapi architecture

Last updated: 2026-10-29. Base commit `31a333a` plus uncommitted P1 chunk 1 (lazy loading, chunker, eval corpus in `eval-v1`).

Solid lines show what exists in the repo now. Dashed lines show what P1 plans to add and isn't built yet. Once a planned piece is built, it moves from dashed to solid and gets a changelog entry at the bottom.

## 1. System overview

```mermaid
flowchart TB
    client["Client<br/>curl / Postman / future Salesforce Apex callout"]

    subgraph api["api container: FastAPI, app/main.py, port 8000"]
        health["GET /health/live<br/>GET /health/ready"]
        rag["POST /rag/query<br/>routes.py"]
        agentep["POST /agent/query<br/>body: query, role"]
        llminf["llm_inference.query_llm<br/>embed, top_k=3, prompt, generate"]
        subgraph graph["LangGraph StateGraph, app/agent/graph.py"]
            retrieve["retrieve node, AGENT_FORCE_SEARCH=1 (default on)<br/>search_documents on the question, via tools node"]
            refuse["refuse node, AGENT_NO_CONTEXT_REFUSAL=1 (default on)<br/>forced search found nothing: fixed reply, no LLM call"]
            agentnode["agent node, async ainvoke<br/>ChatOllama.bind_tools, num_predict=AGENT_MAX_TOKENS (1024)<br/>grounded system prompt, AGENT_GROUNDED_PROMPT=1 (default on)"]
            toolsnode["tools node"]
            authz["authz.check_tool_authorized<br/>reader: search<br/>ingest_agent: search + ingest"]
        end
        emb1["embeddings.get_embedder<br/>all-MiniLM-L6-v2, 384 dim<br/>lazy, copy 1"]
    end

    subgraph mcp["mcp-server container: FastMCP, app/mcp/server.py, port 8001"]
        search["tool: search_documents(query, top_k)<br/>prints chunk_id if present, else source<br/>no results if best score < SEARCH_MIN_SCORE (0.54)"]
        ingest["tool: ingest_document(text, source)<br/>whole text = 1 vector, no chunking"]
        emb2["embeddings.get_embedder<br/>lazy, copy 2"]
    end

    subgraph ollama["ollama container, port 11434"]
        qwen["qwen2.5:1.5b<br/>/api/generate and tool calling"]
    end

    pinecone[("Pinecone serverless, index rag-demo-st (PINECONE_INDEX)<br/>default ns: 7 test canaries, no real docs<br/>eval-v1: 236 chunks, 12 docs<br/>ns picked by PINECONE_NAMESPACE")]
    vs["vectorstore.get_index<br/>lazy, no create on import"]

    client --> health
    client --> rag
    client --> agentep
    health -. "describe_index_stats" .-> pinecone
    health -. "GET /api/tags" .-> qwen
    rag --> llminf --> emb1
    llminf -->|"query top_k=3"| vs --> pinecone
    llminf -->|"generate"| qwen
    agentep --> agentnode
    agentep -.->|"if forced"| retrieve
    retrieve -.-> toolsnode
    toolsnode -.->|"forced search: No relevant documents"| refuse
    agentnode <-->|"chat + tool calls"| qwen
    agentnode -->|"tool_calls present"| toolsnode
    toolsnode --> authz
    toolsnode -->|"MultiServerMCPClient<br/>streamable HTTP /mcp"| search
    toolsnode --> ingest
    toolsnode -->|"ToolMessage"| agentnode
    search --> emb2
    ingest --> emb2
    search -->|"query"| vs
    ingest -->|"upsert, uuid id"| vs

    %% P1 built so far
    subgraph p1built["P1 built"]
        corpus["evals/corpus/raw<br/>12 pinned public docs"]
        chunker["app/services/chunking.py<br/>clean MDX, split by heading, 800 chars"]
        ingestscript["scripts/ingest_corpus.py<br/>ids = chunk ids, writes manifest.json"]
    end
    corpus --> chunker --> ingestscript -->|"namespace eval-v1"| pinecone

    %% P1 eval harness: code written, baseline not run yet
    subgraph p1eval["P1 eval harness (runs: baseline, fixes A-D, held-out)"]
        golden["evals/golden.jsonl<br/>40 cases: 26 answerable, 4 multi-doc,<br/>8 unanswerable, 2 authz"]
        holdout["evals/holdout.jsonl<br/>12 held-out cases, never used for tuning"]
        runeval["evals/run_eval.py<br/>phases: retrieval, agent, judge"]
        scoring["evals/scoring.py<br/>plain assertions"]
        mcpeval["MCP server subprocess<br/>port 8011, ns eval-v1"]
        judge["evals/judge.py, gemma3:4b<br/>3 small calls: refusal, grounding, facts<br/>no context = unsupported by rule"]
        labels["evals/labels/baseline_refusals.jsonl<br/>hand labels, judge agreement"]
        results["evals/results/timestamp_phase.json"]
    end
    golden --> runeval
    holdout --> runeval
    runeval -->|"run_query_traced, in process"| graph
    graph -.->|"MCP_SERVER_URL :8011 during eval"| mcpeval
    mcpeval --> pinecone
    runeval -->|"retriever-only: search_documents in process"| pinecone
    runeval --> scoring
    runeval --> judge
    labels --> runeval
    runeval --> results

    subgraph p1ci["P1 CI"]
        thresholds["evals/thresholds.json<br/>from fix B/D runs"]
        ci["GitHub Actions ci.yml: unit tests on push (green since a709d19),<br/>retrieval gate (skipped until PINECONE_API_KEY secret is set).<br/>eval.yml: full eval on demand, never run"]
    end
    thresholds --> runeval
    ci --> runeval
```

### Tests

| Suite | Runs where | Needs |
|---|---|---|
| `test_authz`, `test_agent_routing`, `test_chunking`, `test_mcp_tools`, `test_api`, `test_scoring`, `test_golden` | Every push (ci.yml) | Nothing; fakes stand in for Pinecone, Ollama and the agent |
| `test_functional` (`--run-functional`) | Local | Starts real MCP (:8021) and uvicorn (:8020) on ns eval-v1, real Ollama. 5/5 pass after the `/rag/query` fix |
| `test_integration` (`--run-integration`) | Local | Real MCP (:8001), writes canaries to ns `integration-test` |

## 2. Request flow: `POST /agent/query`

```mermaid
sequenceDiagram
    participant C as Client
    participant F as FastAPI
    participant G as LangGraph agent
    participant O as Ollama qwen2.5:1.5b
    participant A as authz
    participant M as MCP server
    participant P as Pinecone

    C->>F: POST /agent/query {query, role}
    F->>G: run_query(query, role)
    G->>M: get_tools() (new MCP client on every request)
    Note over G: AGENT_FORCE_SEARCH=1 runs search_documents(question) first, through the same authz check
    Note over G: if that search returns "No relevant documents found" (best score < 0.54), reply "The documents I can search do not cover this." without calling the model
    G->>O: messages + tool schemas (system prompt only if AGENT_GROUNDED_PROMPT=1; reply capped at AGENT_MAX_TOKENS)
    O-->>G: AIMessage with tool_calls, or a final answer
    alt tool_calls present
        G->>A: check_tool_authorized(role, tool)
        alt allowed
            G->>M: search_documents(query, top_k)
            M->>P: query(embedding, top_k)
            P-->>M: matches + metadata
            M-->>G: "[source, score] text" lines
        else denied
            A-->>G: "Denied: ..." as ToolMessage (tool never runs)
        end
        G->>O: messages + ToolMessage
        O-->>G: final answer (loop capped by recursion_limit=10)
    end
    G-->>F: last message content
    F-->>C: {"answer": "..."} (no sources, no tool trace)
```

`POST /rag/query` skips the agent completely. It embeds the question in the api process, queries Pinecone for the top 3 matches, sends one prompt to Ollama `/api/generate`, and returns the answer.

## 3. Deployment

| Where | What | Status |
|---|---|---|
| Local, no Docker | MCP server in one terminal, uvicorn in another, Ollama on the host | Works (RUNBOOK Part 1) |
| Local Docker Compose | 3 services: `ollama`, `mcp-server`, `api`. One shared image `rag-fastapi:latest`, `init: true`, `restart: unless-stopped` | Works (RUNBOOK Part 2) |
| GCP `rag-mcp-agent-prod` | `rag-mcp-agent-vm`, e2-medium, static IP 104.198.167.39 | `/health/live` timed out on 2026-10-29. The trial credit ended on 2026-10-08. Not investigated, since we work locally only. |

## 4. Known gaps that affect P1

| # | Gap | Where |
|---|---|---|
| 1 | ~~Pinecone connection and `create_index` run when the module is imported~~ Fixed: lazy `get_index()`, `ensure_index()` is explicit | `app/services/vectorstore.py` |
| 2 | ~~The embedder loads at import time~~ Fixed (lazy, `embeddings.py`). Still two copies at runtime (api and mcp-server) | `llm_inference.py`, `mcp/server.py` |
| 3 | Chunker exists and the eval corpus is chunked, but `ingest_document` still stores one vector per call | `mcp/server.py` |
| 4 | Measured in the baseline: the agent called search on 0 of 30 answerable questions and made up answers (e.g. "MCP (Maven Central Repository)"). Fix written and switchable (`AGENT_GROUNDED_PROMPT`, `AGENT_FORCE_SEARCH`), both off by default until measured | `agent/graph.py` |
| 5 | ~~No score threshold~~ Added: `SEARCH_MIN_SCORE` (0.54) gates on the best match. Measured: refusals 7/8 golden, 4/5 held-out with fix D. Questions that sound like the docs (GraphQL streaming 0.69, MCP billing 0.56) still pass it | `vectorstore.relevant_matches` |
| 6 | ~~`/rag/query` returns a nested `{"answer": {"answer": ...}}`~~ Fixed after the first functional test run; `query_llm` returns a string | `llm_inference.py` |
| 7 | `/agent/query` returns no sources and no tool trace. The eval uses `run_query_traced()` in process instead; the HTTP API is unchanged | `routes.py`, `graph.run_query` |
| 8 | ~~The integration test writes canary docs into the live index~~ Fixed: writes to namespace `integration-test` | `tests/conftest.py` |
| 9 | ~~Index name and namespace are hardcoded~~ Fixed: `PINECONE_INDEX`, `PINECONE_NAMESPACE` | `vectorstore.py` |
| 10 | Retrieval miss seen during a spot check: "run code after returning a response" did not return `fastapi-background-tasks` in the top 3. Not fixed; kept as a baseline for the eval | retrieval |

## Changelog

| Date | Change | Commit |
|---|---|---|
| 2026-10-29 | First version, describes the repo as it stood at `31a333a` | n/a |
| 2026-10-29 | P1 chunk 1: lazy Pinecone and embedder, `PINECONE_INDEX`/`PINECONE_NAMESPACE` env vars, `chunking.py`, `scripts/ingest_corpus.py`, 236 chunks from 12 docs in namespace `eval-v1`, `search_documents` shows chunk ids | uncommitted |
| 2026-10-29 | P1 chunk 2: golden set (40), `evals/scoring.py`, `evals/judge.py`, `evals/run_eval.py`, `run_query_traced()`, 6 new test files, functional/integration namespaces, `ci.yml` + `eval.yml` written (not run) | uncommitted |
| 2026-10-29 | Runaway generation fix: `AGENT_MAX_TOKENS` (default 1024) and async agent node; one baseline case had generated 7,400+ tokens. Eval: per-case timeout, saves after every case | uncommitted |
| 2026-10-29 | Baseline run (`2026-10-29_131139`). Judge v2 (separate refusal call, no-context grounding by rule) after a hand check found judge v1 wrong on 6 of 8 refusals; hand labels added. Agent fix options added (`retrieve` node, grounded prompt), off by default | uncommitted |
| 2026-10-29 | Fix B made the default (forced search, grounded prompt). Search cutoff `SEARCH_MIN_SCORE` = 0.54 on the best match. Held-out set (12). Groundedness hand labels (17, judge agreement 0.59). Copied to `rag-fastapi/docs/ARCHITECTURE.md` | uncommitted |
| 2026-10-29 | First CI run (3b82644) failed at test collection: plain `pytest` could not import `app`/`evals` (local runs used `python -m pytest`, which hid it; the README command never worked). Added `pytest.ini` with `pythonpath = .` | uncommitted |
| 2026-10-29 | Fix D: `refuse` node, fixed reply when the forced search finds nothing (`AGENT_NO_CONTEXT_REFUSAL`, default on). Runs: fix C and fix D on golden and held-out. `/rag/query` nested answer fixed (functional tests 5/5). Refusal threshold added | uncommitted |
