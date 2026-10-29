"""Unit tests for evals/scoring.py - the eval's own rules need tests too."""

from evals.scoring import (
    refusal_label_agreement,
    check_thresholds,
    check_tool_calls,
    looks_like_refusal,
    retrieval_scores,
    retrieved_ids,
    summarize,
)

CASE = {"id": "x", "question": "How do I resume a LangGraph graph after an interrupt?",
        "required_tools": ["search_documents"], "forbidden_tools": []}


def test_retrieved_ids_parses_tool_output():
    text = "[fastapi-cors#3, score=0.7000] text with [brackets]\n[smoke-test, score=0.1] x\nnot a result line"
    assert retrieved_ids(text) == ["fastapi-cors#3", "smoke-test"]
    assert retrieved_ids("No relevant documents found.") == []
    assert retrieved_ids(None) == []


def test_retrieval_scores_rank_and_doc_hit():
    s = retrieval_scores(["a#1", "a#2"], ["b#0", "a#2", "a#1"])
    assert s == {"hit": True, "doc_hit": True, "rr": 0.5}
    s = retrieval_scores(["a#1"], ["a#5", "b#0"])
    assert s == {"hit": False, "doc_hit": True, "rr": 0.0}
    assert retrieval_scores(["a#1"], [])["doc_hit"] is False


def test_tool_check_passes_sane_search():
    calls = [{"name": "search_documents", "args": {"query": "resume langgraph interrupt", "top_k": 3}}]
    assert check_tool_calls(CASE, calls, [{"name": "search_documents", "content": "[a#1, score=0.5] x"}])["ok"]


def test_tool_check_flags_missing_search():
    r = check_tool_calls(CASE, [], [])
    assert not r["ok"] and r["problems"] == ["did not call search_documents"]


def test_tool_check_flags_bad_args():
    calls = [
        {"name": "search_documents", "args": {"query": ""}},
        {"name": "search_documents", "args": {"query": "pizza recipes"}},
        {"name": "search_documents", "args": {"query": "langgraph", "top_k": 500}},
    ]
    problems = check_tool_calls(CASE, calls, [])["problems"]
    assert "search_documents called with empty query" in problems
    assert any("shares no words" in p for p in problems)
    assert any("top_k out of range" in p for p in problems)


def test_tool_check_denied_forbidden_tool_is_fine_but_executed_is_not():
    case = {"question": "ingest this", "required_tools": [], "forbidden_tools": ["ingest_document"]}
    denied = [{"name": "ingest_document", "content": "Denied: role 'reader' is not authorized"}]
    ran = [{"name": "ingest_document", "content": "Ingested document 'x' as id=1."}]
    assert check_tool_calls(case, [], denied)["ok"]
    assert check_tool_calls(case, [], ran)["problems"] == ["forbidden tool ingest_document executed"]


def test_tool_check_flags_tool_errors():
    r = check_tool_calls(CASE, [{"name": "search_documents", "args": {"query": "langgraph resume"}}],
                         [{"name": "search_documents", "content": "Error calling tool 'search_documents': boom"}])
    assert not r["ok"]


def test_refusal_regex():
    assert looks_like_refusal("The provided documents do not contain information about MongoDB.")
    assert looks_like_refusal("I couldn't find anything about that.")
    assert not looks_like_refusal("Use BackgroundTasks and call add_task().")


def _rec(cat, **kw):
    base = {"id": "x", "category": cat, "repeat": 0, "latency_s": 1.0, "answer": "a",
            "tool_check": {"ok": True, "problems": []}}
    base.update(kw)
    return base


def test_summarize_counts_and_rates():
    records = [
        _rec("answerable", retriever={"hit": True, "doc_hit": True, "rr": 1.0},
             agent_retrieved=["a#1"], agent_retrieval={"hit": True, "doc_hit": True, "rr": 1.0},
             judge={"grounded": "supported", "refused": False, "fact_coverage": 1.0}),
        _rec("answerable", retriever={"hit": False, "doc_hit": True, "rr": 0.0}, agent_retrieved=[],
             tool_check={"ok": False, "problems": ["did not call search_documents"]},
             judge={"grounded": "partial", "refused": False, "fact_coverage": 0.5}, latency_s=3.0),
        _rec("unanswerable", answer="The documents do not mention that.",
             judge={"grounded": "no_claims", "refused": True, "fact_coverage": 0.0}),
        {"id": "e", "category": "answerable", "repeat": 0, "error": "boom"},
    ]
    s = summarize(records)
    assert s["cases"] == 4 and s["errors"] == 1
    assert s["tool_call_correctness"] == 0.6667
    assert s["retriever_hit_at_k"] == 0.5 and s["retriever_doc_hit_at_k"] == 1.0 and s["retriever_mrr"] == 0.5
    assert s["agent_hit_rate"] == 1.0
    assert s["agent_searched_rate"] == 0.5
    assert s["groundedness"] == 0.5 and s["groundedness_incl_partial"] == 1.0
    assert s["answer_fact_coverage"] == 0.75
    assert s["refusal_accuracy"] == 1.0 and s["false_refusal_rate"] == 0.0
    assert s["refusal_accuracy_regex"] == 1.0


def test_summarize_without_judge_leaves_judge_metrics_none():
    s = summarize([_rec("answerable", retriever={"hit": True, "doc_hit": True, "rr": 1.0})])
    assert s["groundedness"] is None and s["refusal_accuracy"] is None


def test_check_thresholds():
    summary = {"retriever_hit_at_k": 0.6, "groundedness": None, "false_refusal_rate": 0.3}
    rules = {"retriever_hit_at_k": {"min": 0.7}, "groundedness": {"min": 0.5}, "false_refusal_rate": {"max": 0.2}}
    assert check_thresholds(summary, rules) == [
        "retriever_hit_at_k=0.6 below min 0.7",
        "false_refusal_rate=0.3 above max 0.2",
    ]


def test_summarize_retrieval_only_run():
    records = [{"id": "x", "category": "answerable", "repeat": 0, "retriever": {"hit": True, "doc_hit": True, "rr": 1.0}},
               {"id": "u", "category": "unanswerable", "repeat": 0}]
    s = summarize(records)
    assert s["retriever_hit_at_k"] == 1.0
    assert s["tool_call_correctness"] is None and s["agent_searched_rate"] is None
    assert s["latency_p50_s"] is None and s["refusal_accuracy_regex"] is None


def test_refusal_label_agreement_matches_on_answer_hash():
    import hashlib

    sha = lambda a: hashlib.sha1(a.encode()).hexdigest()[:12]
    records = [
        {"id": "a", "answer": "I don't know.", "judge": {"refused": True}},
        {"id": "b", "answer": "Use add_task().", "judge": {"refused": True}},
        {"id": "c", "answer": "changed answer", "judge": {"refused": False}},
    ]
    labels = [
        {"id": "a", "refused": True, "answer_sha": sha("I don't know.")},
        {"id": "b", "refused": False, "answer_sha": sha("Use add_task().")},
        {"id": "c", "refused": False, "answer_sha": sha("the original answer")},
    ]
    assert refusal_label_agreement(records, labels) == {
        "compared": 2, "agree": 1, "agreement": 0.5, "disagreements": ["b"],
    }
