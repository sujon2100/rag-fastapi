# Eval harness

This scores the RAG + MCP agent on a fixed set of 40 questions against a fixed corpus. Each run writes one JSON file to `results/`.

## Pieces

| File | What it is |
|---|---|
| `corpus/raw/` | 12 public docs pages (FastAPI, LangGraph, MCP), pinned to upstream commits. Licenses and attribution are in `corpus/SOURCES.md` |
| `corpus/manifest.json` | Chunk ids, sections and hashes, written by `scripts/ingest_corpus.py` |
| `golden.jsonl` | 40 cases: 26 answerable, 4 multi-doc, 8 unanswerable, 2 authz |
| `scoring.py` | Plain-assertion scoring. No models, unit tested in `tests/test_scoring.py` |
| `judge.py` | LLM judge through local Ollama, default `gemma3:4b` (see "How far to trust the judge" in the top-level README) |
| `run_eval.py` | The runner |
| `thresholds.json` | Minimums the CI gate checks. Set from the fix B run (`2026-10-29_143035`), each a case or two below the measured value. Refusal minimum added after fix D. Checked: the original agent fails 5, forced search and the cutoff version each fail refusals, the current version passes all |

## Golden case format

```json
{"id": "lg-03", "category": "answerable", "role": "reader",
 "question": "How do I resume a LangGraph graph after it pauses on an interrupt?",
 "expected_chunks": ["langgraph-interrupts#6", "langgraph-interrupts#7", "langgraph-interrupts#8"],
 "required_tools": ["search_documents"],
 "key_facts": ["invoke the graph again with a Command containing the resume value", "..."]}
```

`expected_chunks` lists every chunk that would be an acceptable hit; any one of them counts. Unanswerable cases were checked against the corpus with grep to make sure the topic really is absent. Authz cases ask a `reader` to ingest, and add `forbidden_tools: ["ingest_document"]`.

## Metrics

| Metric | How it is computed |
|---|---|
| `tool_call_correctness` | Share of cases where the agent called every required tool, every `search_documents` query was non-empty and shared a content word with the question, `top_k` was 1 to 10, no forbidden tool actually ran (a `Denied:` result is fine), and no tool errored |
| `retriever_hit_at_k` | Calls `search_documents` directly with the question, no LLM. Share of answerable and multi-doc cases where any expected chunk is in the top k (k=3, the tool's default) |
| `retriever_doc_hit_at_k` | Same, but any chunk from an expected document counts |
| `retriever_mrr` | Mean reciprocal rank of the first expected chunk (0 if missing) |
| `agent_searched_rate` | Share of answerable cases where the agent called search at all |
| `agent_hit_rate` | Of the cases where the agent searched, the share where its own query retrieved an expected chunk |
| `groundedness` | Judge: share of answerable cases where every claim in the answer is supported by what the agent retrieved. A refusal or an answer given without searching counts as not grounded |
| `groundedness_incl_partial` | Same, also counting "partial" |
| `answer_fact_coverage` | Judge: mean share of `key_facts` the answer states |
| `refusal_accuracy` | Judge: share of unanswerable cases where the agent declined |
| `false_refusal_rate` | Judge: share of answerable cases where it declined anyway |
| `refusal_accuracy_regex` | Regex fallback for refusals, used to sanity-check the judge |
| `latency_p50_s` / `latency_p95_s` | Wall time per agent run, measured on whatever machine ran the eval |

## Running

```bash
python -m scripts.ingest_corpus                 # once: corpus -> Pinecone namespace eval-v1
python -m evals.run_eval --phase retrieval      # retriever only, no LLM
python -m evals.run_eval --phase agent          # + agent (needs Ollama with qwen2.5:1.5b)
python -m evals.run_eval --phase judge --results evals/results/<file>_agent.json
python -m evals.run_eval                        # all three phases in one go
python -m evals.run_eval --ids fa-01,un-03 --repeats 3 --gate evals/thresholds.json
python -m evals.run_eval --golden evals/holdout.jsonl     # held-out set
python -m evals.run_eval --force-search 0 --grounded-prompt 0 --no-context-refusal 0   # original agent
SEARCH_MIN_SCORE=0 python -m evals.run_eval              # without the cutoff
```

The agent runs at temperature 0 by default so runs can be compared. The deployed service uses Ollama's default temperature; pass `--temperature -1` to evaluate that. The judge runs after the agent model is unloaded, because `gpt-oss:20b` and `qwen2.5:1.5b` don't both fit in 16 GB of RAM.

## Results

Filled in from real runs only. See the top-level README.
