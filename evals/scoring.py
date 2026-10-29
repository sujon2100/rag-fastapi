"""
Plain-assertion scoring for the eval harness. No network, no models here -
everything that needs a model lives in judge.py. Kept separate so the scoring
rules themselves have unit tests (tests/test_scoring.py).
"""

import re
import statistics

_RESULT_LINE = re.compile(r"^\[([^,\]]+), score=([0-9.]+)\]", re.M)
_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")
_STOP = {
    "what", "which", "when", "where", "does", "with", "that", "this", "from", "into",
    "should", "would", "could", "about", "there", "their", "have", "your", "them",
    "after", "before", "some", "they", "than", "then", "each", "other", "while", "still",
}

# Fallback only; the judge decides refusals. Used when the judge is skipped.
_REFUSAL = re.compile(
    r"(no relevant documents|not (?:found|mentioned|covered|available|present) in|"
    r"(?:does not|doesn't|do not|don't) (?:contain|mention|include|cover|provide|have)|"
    r"(?:couldn't|could not|unable to|can't|cannot) (?:find|locate|answer)|"
    r"no information|not enough information|i don't know|i do not know)",
    re.I,
)

CATEGORIES_WITH_SOURCES = {"answerable", "multi_doc"}


def retrieved_ids(tool_text: str) -> list[str]:
    """Chunk/source ids in the order search_documents printed them."""
    return [m.group(1).strip() for m in _RESULT_LINE.finditer(tool_text or "")]


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def retrieval_scores(expected: list[str], retrieved: list[str]) -> dict:
    """hit = any expected chunk retrieved; doc_hit = any chunk from an expected doc; rr = 1/rank."""
    exp = set(expected)
    exp_docs = {doc_of(c) for c in expected}
    rank = next((i + 1 for i, r in enumerate(retrieved) if r in exp), None)
    return {
        "hit": rank is not None,
        "doc_hit": any(doc_of(r) in exp_docs for r in retrieved),
        "rr": 1.0 / rank if rank else 0.0,
    }


def check_tool_calls(case: dict, tool_calls: list[dict], tool_results: list[dict]) -> dict:
    """
    Did the agent use its tools sanely for this case?
      - every required tool was called at least once
      - search_documents args: non-empty query sharing a content word with
        the question, top_k (if given) between 1 and 10
      - forbidden tools never actually ran (a "Denied:" result is fine - that
        is the authz layer doing its job)
      - no tool raised an error
    """
    problems = []
    called = [c["name"] for c in tool_calls]

    for name in case.get("required_tools", []):
        if name not in called:
            problems.append(f"did not call {name}")

    q_words = content_words(case["question"])
    for c in tool_calls:
        if c["name"] != "search_documents":
            continue
        args = c.get("args") or {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            problems.append("search_documents called with empty query")
        elif not (content_words(query) & q_words):
            problems.append(f"search query shares no words with question: {query!r}")
        top_k = args.get("top_k")
        if top_k is not None and not (isinstance(top_k, int) and 1 <= top_k <= 10):
            problems.append(f"search_documents top_k out of range: {top_k!r}")

    for r in tool_results:
        content = r.get("content") or ""
        if r["name"] in case.get("forbidden_tools", []) and not content.startswith("Denied"):
            problems.append(f"forbidden tool {r['name']} executed")
        if content.startswith("Error calling tool"):
            problems.append(f"tool error: {content[:120]}")

    return {"ok": not problems, "problems": problems}


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer or ""))


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 2)


def summarize(records: list[dict]) -> dict:
    """
    One summary over a list of per-case records (one repeat). Metrics that
    need the judge are None when the judge has not run yet.
    """
    src = [r for r in records if r["category"] in CATEGORIES_WITH_SOURCES]
    unans = [r for r in records if r["category"] == "unanswerable"]
    errors = [r for r in records if r.get("error")]
    # records that went through the agent (a retrieval-only run has none)
    ok = [r for r in records if not r.get("error") and "tool_check" in r]
    src_ok = [r for r in src if not r.get("error") and "tool_check" in r]

    out = {
        "cases": len(records),
        "errors": len(errors),
        "tool_call_correctness": _rate([r["tool_check"]["ok"] for r in ok]),
        "retriever_hit_at_k": _rate([r["retriever"]["hit"] for r in src if "retriever" in r]),
        "retriever_doc_hit_at_k": _rate([r["retriever"]["doc_hit"] for r in src if "retriever" in r]),
        "retriever_mrr": round(statistics.mean([r["retriever"]["rr"] for r in src if "retriever" in r]), 4)
        if any("retriever" in r for r in src) else None,
        "agent_hit_rate": _rate([r["agent_retrieval"]["hit"] for r in src if r.get("agent_retrieval")]),
        "agent_searched_rate": _rate([bool(r.get("agent_retrieved")) for r in src_ok]),
        # searches the model asked for itself (forced searches excluded)
        "model_initiated_search_rate": _rate([
            any(c["name"] == "search_documents" and not c.get("forced") for c in r.get("tool_calls", []))
            for r in src_ok
        ]),
        "latency_p50_s": _pct([r["latency_s"] for r in ok if "latency_s" in r], 0.5),
        "latency_p95_s": _pct([r["latency_s"] for r in ok if "latency_s" in r], 0.95),
    }

    judged_src = [r for r in src if r.get("judge")]
    judged_un = [r for r in unans if r.get("judge")]
    out["groundedness"] = _rate([r["judge"]["grounded"] == "supported" for r in judged_src]) if judged_src else None
    out["groundedness_incl_partial"] = (
        _rate([r["judge"]["grounded"] in ("supported", "partial") for r in judged_src]) if judged_src else None
    )
    out["answer_fact_coverage"] = (
        round(statistics.mean([r["judge"]["fact_coverage"] for r in judged_src]), 4) if judged_src else None
    )
    out["refusal_accuracy"] = _rate([r["judge"]["refused"] for r in judged_un]) if judged_un else None
    out["false_refusal_rate"] = _rate([r["judge"]["refused"] for r in judged_src]) if judged_src else None
    out["refusal_accuracy_regex"] = _rate([looks_like_refusal(r["answer"]) for r in unans if "answer" in r])
    return out


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Return a failure message per metric under its minimum (or over its maximum)."""
    failures = []
    for key, rule in thresholds.items():
        value = summary.get(key)
        if value is None:
            continue
        if "min" in rule and value < rule["min"]:
            failures.append(f"{key}={value} below min {rule['min']}")
        if "max" in rule and value > rule["max"]:
            failures.append(f"{key}={value} above max {rule['max']}")
    return failures


def refusal_label_agreement(records: list[dict], labels: list[dict], key: str = "refused",
                            label_key: str = "refused") -> dict:
    """
    How often the judge's refusal call matches a hand label. A label only
    applies to the exact answer it was written for (matched by answer hash),
    so labels from one run are never applied to a different run's answers.
    """
    import hashlib

    by_id = {lab["id"]: lab for lab in labels}
    pairs = []
    for r in records:
        lab = by_id.get(r["id"])
        if not lab or not r.get("judge") or "answer" not in r:
            continue
        if hashlib.sha1(r["answer"].encode()).hexdigest()[:12] != lab["answer_sha"]:
            continue
        if key not in r["judge"]:
            continue
        pairs.append((r["judge"][key], lab[label_key], r["id"]))
    return {
        "compared": len(pairs),
        "agree": sum(j == h for j, h, _ in pairs),
        "agreement": round(sum(j == h for j, h, _ in pairs) / len(pairs), 4) if pairs else None,
        "disagreements": [i for j, h, i in pairs if j != h],
    }


def grounding_label_agreement(records: list[dict], labels: list[dict]) -> dict:
    return refusal_label_agreement(records, labels, key="grounded", label_key="grounded")
