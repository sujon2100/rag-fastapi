"""
Eval harness for the RAG + MCP agent.

  python -m evals.run_eval                          # retrieval + agent + judge, all 40 cases
  python -m evals.run_eval --phase retrieval        # retriever only, no LLM (fast, used in CI)
  python -m evals.run_eval --phase agent            # retrieval + agent, judge later
  python -m evals.run_eval --phase judge --results evals/results/<file>.json
  python -m evals.run_eval --ids fa-01,un-03 --repeats 3
  python -m evals.run_eval --gate evals/thresholds.json   # exit 1 if under thresholds

What it measures (see evals/README.md for definitions):
  tool_call_correctness, retriever hit@k / doc hit@k / MRR, agent hit rate,
  groundedness and fact coverage (judge), refusal accuracy and false refusal
  rate (judge), latency p50/p95. Every run writes evals/results/<timestamp>.json.

The agent runs in-process (run_query_traced) against a real MCP server
subprocess, real Pinecone (namespace eval-v1) and real Ollama. Judge and
agent models are run one after the other, never loaded together - on a 16GB
laptop gpt-oss:20b and qwen2.5 do not fit side by side.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "evals" / "golden.jsonl"
RESULTS_DIR = REPO_ROOT / "evals" / "results"


def load_golden(ids: list[str] | None = None, path: Path = GOLDEN) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            sys.exit(f"unknown case ids: {sorted(missing)}")
    return cases


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def start_mcp_server(port: int, namespace: str) -> subprocess.Popen:
    if _port_open(port):
        sys.exit(f"port {port} already in use - stop whatever is on it or pass --mcp-port")
    env = {**os.environ, "MCP_SERVER_PORT": str(port), "PINECONE_NAMESPACE": namespace}
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp.server"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if _port_open(port):
            return proc
        if proc.poll() is not None:
            sys.exit("MCP server exited during startup")
        time.sleep(0.5)
    proc.terminate()
    sys.exit(f"MCP server did not open port {port} within 60s")


def run_retrieval(cases, records_by_key, top_k):
    from app.mcp.server import search_documents
    from evals.scoring import CATEGORIES_WITH_SOURCES, retrieval_scores, retrieved_ids

    for case in cases:
        if case["category"] not in CATEGORIES_WITH_SOURCES:
            continue
        ids = retrieved_ids(search_documents(case["question"], top_k=top_k))
        result = {"retrieved": ids, **retrieval_scores(case["expected_chunks"], ids)}
        for rec in records_by_key.values():
            if rec["id"] == case["id"]:
                rec["retriever"] = result
        print(f"  retrieval {case['id']:6} hit={result['hit']!s:5} doc_hit={result['doc_hit']!s:5} {ids}")


async def run_agent(cases, records_by_key, repeats, temperature, case_timeout, save, grounded_prompt, force_search):
    from app.agent.graph import run_query_traced
    from evals.scoring import CATEGORIES_WITH_SOURCES, check_tool_calls, retrieval_scores, retrieved_ids

    for rep in range(repeats):
        for case in cases:
            rec = records_by_key[(case["id"], rep)]
            t0 = time.perf_counter()
            try:
                # llm_timeout bounds each Ollama call, wait_for bounds the whole case
                trace = await asyncio.wait_for(
                    run_query_traced(
                        case["question"], role=case["role"],
                        thread_id=f"eval-{case['id']}-{rep}-{time.time_ns()}", temperature=temperature,
                        llm_timeout=case_timeout / 2,
                        grounded_prompt=grounded_prompt, force_search=force_search,
                    ),
                    timeout=case_timeout,
                )
            except Exception as exc:  # record and keep going; one bad case should not kill the run
                rec.update(error=f"{type(exc).__name__}: {exc}", latency_s=round(time.perf_counter() - t0, 2))
                print(f"  agent {case['id']:6} rep={rep} ERROR {rec['error'][:150]}")
                save()
                continue
            rec["latency_s"] = round(time.perf_counter() - t0, 2)
            rec["answer"] = trace["answer"]
            rec["tool_calls"] = trace["tool_calls"]
            rec["tool_results"] = [{"name": r["name"], "content": r["content"][:4000]} for r in trace["tool_results"]]
            search_texts = [r["content"] for r in trace["tool_results"] if r["name"] == "search_documents"]
            rec["agent_retrieved_text"] = search_texts
            rec["agent_retrieved"] = [i for t in search_texts for i in retrieved_ids(t)]
            if case["category"] in CATEGORIES_WITH_SOURCES and rec["agent_retrieved"]:
                rec["agent_retrieval"] = retrieval_scores(case["expected_chunks"], rec["agent_retrieved"])
            rec["tool_check"] = check_tool_calls(case, trace["tool_calls"], trace["tool_results"])
            print(
                f"  agent {case['id']:6} rep={rep} {rec['latency_s']:6.1f}s tools={[c['name'] for c in trace['tool_calls']]} "
                f"tool_ok={rec['tool_check']['ok']}"
            )
            save()


def run_judge(cases, records, model, save=lambda: None):
    from evals.judge import judge_case

    by_id = {c["id"]: c for c in cases}
    for rec in records:
        if rec.get("error") or rec["category"] == "authz" or "answer" not in rec:
            continue
        try:
            rec["judge"] = judge_case(rec, by_id[rec["id"]], model=model)
        except Exception as exc:
            rec["judge_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  judge {rec['id']:6} rep={rec['repeat']} ERROR {rec['judge_error'][:150]}")
            continue
        j = rec["judge"]
        print(f"  judge {rec['id']:6} rep={rec['repeat']} {j['latency_s']:6.1f}s grounded={j['grounded']:11} refused={j['refused']!s:5} facts={j['fact_coverage']}")
        save()


def build_summary(records, repeats):
    from evals.scoring import summarize

    per_repeat = [summarize([r for r in records if r["repeat"] == i]) for i in range(repeats)]
    if repeats == 1:
        return {"per_repeat": per_repeat, "mean": per_repeat[0]}
    mean, spread = {}, {}
    for key in per_repeat[0]:
        vals = [s[key] for s in per_repeat if s[key] is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            mean[key] = round(statistics.mean(vals), 4)
            spread[key] = [min(vals), max(vals)]
        else:
            mean[key] = None
    return {"per_repeat": per_repeat, "mean": mean, "min_max": spread}


def print_summary(summary):
    print("\nsummary (mean over repeats):")
    for k, v in summary["mean"].items():
        rng = summary.get("min_max", {}).get(k)
        print(f"  {k:28} {v}" + (f"   range {rng}" if rng and rng[0] != rng[1] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "agent", "retrieval", "judge"], default="all")
    ap.add_argument("--results", help="existing results file (for --phase judge)")
    ap.add_argument("--ids", help="comma-separated case ids")
    ap.add_argument("--golden", default=str(GOLDEN), help="case file (evals/holdout.jsonl for the held-out set)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=3, help="retriever-only k; matches the tool's default")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="agent temperature; pass -1 for Ollama's default (what the service uses)")
    ap.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gemma3:4b"))
    ap.add_argument("--namespace", default="eval-v1")
    ap.add_argument("--mcp-port", type=int, default=8011)
    ap.add_argument("--gate", help="thresholds json; exit 1 if the mean summary misses any")
    ap.add_argument("--grounded-prompt", type=int, choices=[0, 1], default=1,
                    help="grounded system prompt on/off (default on, as in the service)")
    ap.add_argument("--force-search", type=int, choices=[0, 1], default=1,
                    help="forced first search on/off (default on, as in the service)")
    ap.add_argument("--label", default="", help="short tag added to the results filename")
    ap.add_argument("--case-timeout", type=float, default=300.0, help="seconds before one agent case is recorded as an error")
    args = ap.parse_args(argv)

    os.environ["PINECONE_NAMESPACE"] = args.namespace
    os.environ["MCP_SERVER_URL"] = f"http://127.0.0.1:{args.mcp_port}/mcp"
    sys.path.insert(0, str(REPO_ROOT))
    temperature = None if args.temperature < 0 else args.temperature

    def save():
        """Partial write after every case, so a crash or hang loses nothing already measured."""
        out_path.write_text(json.dumps({"meta": {**meta, "partial": True}, "records": records}, indent=2) + "\n")

    if args.phase == "judge":
        if not args.results:
            sys.exit("--phase judge needs --results")
        src = Path(args.results).resolve()
        data = json.loads(src.read_text())
        out_path = src.with_name(src.stem.removesuffix("_agent") + "_judged.json")
        cases = load_golden(sorted({r["id"] for r in data["records"]}), Path(data["meta"].get("golden", GOLDEN)))
        records, meta = data["records"], data["meta"]
    else:
        cases = load_golden(args.ids.split(",") if args.ids else None, Path(args.golden))
        repeats = 1 if args.phase == "retrieval" else args.repeats
        records = [
            {"id": c["id"], "category": c["category"], "repeat": r}
            for r in range(repeats) for c in cases
        ]
        meta = {
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "phase": args.phase,
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "namespace": args.namespace,
            "top_k": args.top_k,
            "agent_model": os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5:1.5b"),
            "agent_temperature": temperature if temperature is not None else "ollama default",
            "judge_model": None,
            "repeats": repeats,
            "cases": len(cases),
            "golden": str(Path(args.golden).resolve().relative_to(REPO_ROOT)),
            "host": f"{platform.system()} {platform.machine()} {platform.processor()}",
        }
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{stamp}{'_' + args.label if args.label else ''}_{args.phase}.json"

        meta["case_timeout_s"] = args.case_timeout
        meta["grounded_prompt"] = bool(args.grounded_prompt)
        meta["force_search"] = bool(args.force_search)
        meta["agent_max_tokens"] = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
        records_by_key = {(r["id"], r["repeat"]): r for r in records}
        t0 = time.perf_counter()
        print(f"retrieval-only, top_k={args.top_k}")
        run_retrieval(cases, records_by_key, args.top_k)
        if args.phase in ("all", "agent"):
            proc = start_mcp_server(args.mcp_port, args.namespace)
            try:
                print(f"agent, {repeats} repeat(s), temperature={meta['agent_temperature']}")
                asyncio.run(run_agent(cases, records_by_key, repeats, temperature, args.case_timeout, save,
                                      bool(args.grounded_prompt), bool(args.force_search)))
            finally:
                proc.terminate()
                proc.wait(timeout=10)
        meta["agent_phase_s"] = round(time.perf_counter() - t0, 1)

    if args.phase in ("all", "judge"):
        from evals.judge import unload

        unload(meta["agent_model"])  # free RAM before loading the judge
        t1 = time.perf_counter()
        print(f"judge, model={args.judge_model}")
        run_judge(cases, records, args.judge_model, save)
        meta["judge_model"] = args.judge_model
        meta["judge_phase_s"] = round(time.perf_counter() - t1, 1)
        unload(args.judge_model)

    summary = build_summary(records, max(r["repeat"] for r in records) + 1)
    gl_path = REPO_ROOT / "evals" / "labels" / "fixB_grounding.jsonl"
    if gl_path.exists() and any(r.get("judge") for r in records):
        from evals.scoring import grounding_label_agreement

        gl = [json.loads(line) for line in gl_path.read_text().splitlines() if line.strip()]
        summary["grounding_vs_hand_labels"] = grounding_label_agreement(records, gl)
    labels_path = REPO_ROOT / "evals" / "labels" / "baseline_refusals.jsonl"
    if labels_path.exists() and any(r.get("judge") for r in records):
        from evals.scoring import refusal_label_agreement

        labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
        summary["refusal_vs_hand_labels"] = {
            "regex (used for scoring)": refusal_label_agreement(records, labels, "refused"),
            "judge": refusal_label_agreement(records, labels, "refused_judge"),
        }
    out_path.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2) + "\n")
    print_summary(summary)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")

    if args.gate:
        from evals.scoring import check_thresholds

        failures = check_thresholds(summary["mean"], json.loads(Path(args.gate).read_text()))
        if failures:
            print("GATE FAILED:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("gate passed")


if __name__ == "__main__":
    main()
