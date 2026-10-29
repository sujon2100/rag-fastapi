"""
LLM judge for refusals, groundedness and key-fact coverage, run through a
local Ollama model (default gemma3:4b; gpt-oss:20b does not fit next to
anything else in 16GB RAM - see README).

Each question is its own small call. The first version asked one prompt for
refusal + groundedness + facts at once, and checking it by hand against the
baseline showed gemma3:4b mixing the questions up: it marked 6 confident
(wrong) answers as "refused" because they were not grounded, and called
answers "partial"/"supported" when nothing had been retrieved at all.

A second version asked refusal as its own yes/no question. That was worse:
gemma3:4b agreed with hand labels on 19/38, often writing a reason that says
the reply answers the question and then setting declined=true. The plain
refusal regex in scoring.py agreed on 36/38.

So now:
  - refusal: decided by the regex (scoring.looks_like_refusal); the judge's
    yes/no is still asked and stored as refused_judge, so its agreement with
    the hand labels stays visible in every run
  - groundedness: decided by rule when nothing was retrieved (an answer with
    no context cannot be supported by it); asked of the judge only when
    there is retrieved text to check against
  - facts: one question listing the key facts
"""

import json
import os
import re
import time

import requests

from evals.scoring import looks_like_refusal

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_CONTEXT_CHARS = 6000

REFUSAL_PROMPT = """Here is a question and an assistant's reply.

QUESTION:
{question}

REPLY:
{answer}

Does the reply decline to answer? Count it as declining if it says it does not know, says it lacks the
information or access, says the information is not available, or only asks the user for more details
without answering. If the reply gives an actual answer (even a wrong one), it is NOT declining.

Reply with JSON only: {{"declined": true|false, "reason": "one short sentence"}}"""

GROUNDING_PROMPT = """You check whether an answer is supported by source text.

SOURCE TEXT (the only material the assistant was given):
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER stated in the SOURCE TEXT? Do not use your own knowledge;
a claim that is true but not in the SOURCE TEXT is unsupported.
Use "supported" (all claims in the source), "partial" (some are, some are not) or "unsupported"
(the main claims are not in the source).

Reply with JSON only: {{"grounded": "supported|partial|unsupported", "reason": "one short sentence"}}"""

FACTS_PROMPT = """Here is an answer and a numbered list of facts.

ANSWER:
{answer}

FACTS:
{facts}

For each fact, in order, is it stated in the ANSWER (wording may differ)?

Reply with JSON only: {{"covered": [true|false for each fact, in order]}}"""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _chat(model: str, prompt: str, timeout: int) -> tuple[dict, dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    if model.startswith("gpt-oss"):
        body["think"] = "low"
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return _extract_json(data["message"]["content"]), data


def judge_case(record: dict, case: dict, model: str, timeout: int = 1800) -> dict:
    answer = record.get("answer") or "(empty answer)"
    context = "\n\n".join(record.get("agent_retrieved_text") or [])
    facts = case.get("key_facts") or []
    t0 = time.perf_counter()
    tokens = 0

    verdict, data = _chat(model, REFUSAL_PROMPT.format(question=case["question"], answer=answer), timeout)
    tokens += (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
    refused_judge = bool(verdict.get("declined"))
    refused = looks_like_refusal(answer)
    reasons = {"refusal_judge": verdict.get("reason", "")}

    if not context.strip():
        grounded, grounded_source = ("no_claims" if refused else "unsupported"), "rule"
        reasons["grounded"] = "nothing was retrieved"
    elif refused:
        grounded, grounded_source = "no_claims", "rule"
        reasons["grounded"] = "reply declined"
    else:
        verdict, data = _chat(
            model, GROUNDING_PROMPT.format(context=context[:MAX_CONTEXT_CHARS], answer=answer), timeout
        )
        tokens += (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        grounded = str(verdict.get("grounded", "")).lower()
        if grounded not in {"supported", "partial", "unsupported"}:
            grounded = "invalid"
        grounded_source = "judge"
        reasons["grounded"] = verdict.get("reason", "")

    covered = []
    if facts:
        verdict, data = _chat(
            model,
            FACTS_PROMPT.format(answer=answer, facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))),
            timeout,
        )
        tokens += (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        covered = [bool(x) for x in (verdict.get("covered") or [])][: len(facts)]
        covered += [False] * (len(facts) - len(covered))

    return {
        "model": model,
        "refused": refused,
        "refused_source": "regex",
        "refused_judge": refused_judge,
        "grounded": grounded,
        "grounded_source": grounded_source,
        "facts_covered": covered,
        "fact_coverage": round(sum(covered) / len(facts), 4) if facts else 0.0,
        "reasons": reasons,
        "latency_s": round(time.perf_counter() - t0, 2),
        "tokens": tokens,
    }


def unload(model: str):
    """Ask Ollama to drop a model from memory (keep_alive=0)."""
    try:
        requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass
