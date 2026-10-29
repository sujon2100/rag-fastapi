# Eval corpus sources

Files in `raw/` are unmodified copies of public documentation, pinned to the commits below so the eval corpus never changes underneath the golden set. Cleaning (removing code blocks, JS variants and MDX tags) happens at ingest time in `app/services/chunking.py`. The raw files are left exactly as downloaded.

Fetched on 2026-10-29.

| File | Upstream path | Repo @ commit | License |
|---|---|---|---|
| fastapi-background-tasks.md | docs/en/docs/tutorial/background-tasks.md | fastapi/fastapi @ 79cab5bed28b24d92c185d236feb33f5783dfeb3 | MIT |
| fastapi-dependencies.md | docs/en/docs/tutorial/dependencies/index.md | same | MIT |
| fastapi-handling-errors.md | docs/en/docs/tutorial/handling-errors.md | same | MIT |
| fastapi-middleware.md | docs/en/docs/tutorial/middleware.md | same | MIT |
| fastapi-cors.md | docs/en/docs/tutorial/cors.md | same | MIT |
| fastapi-testing.md | docs/en/docs/tutorial/testing.md | same | MIT |
| langgraph-persistence.mdx | src/oss/langgraph/persistence.mdx | langchain-ai/docs @ d8802f3fce7a80642f11d548452aeed53b47e33f | MIT |
| langgraph-interrupts.mdx | src/oss/langgraph/interrupts.mdx | same | MIT |
| langgraph-streaming.mdx | src/oss/langgraph/streaming.mdx | same | MIT |
| mcp-architecture.mdx | docs/docs/2026-07-28/learn/architecture.mdx | modelcontextprotocol/modelcontextprotocol @ 84d845729393f79b3fd3d392aa17cb86f5b6f95c | CC-BY-4.0 (docs) |
| mcp-server-concepts.mdx | docs/docs/2026-07-28/learn/server-concepts.mdx | same | CC-BY-4.0 (docs) |
| mcp-transports.mdx | docs/specification/2025-11-25/basic/transports.mdx | same | Apache-2.0 (spec) |

Copyright for FastAPI belongs to Sebastián Ramírez, for LangGraph docs to LangChain, Inc., and for MCP to the Model Context Protocol contributors. The licenses are the ones in each upstream repo at the commit above. MCP's LICENSE file says documentation is CC-BY-4.0 and specification text is Apache-2.0.

Why these pages: they cover the same stack this service is built on (FastAPI, LangGraph, MCP), they are real prose written by other people, and their licenses allow redistribution with attribution. A corpus written by the person who also writes the golden questions would be grading its own exam.
