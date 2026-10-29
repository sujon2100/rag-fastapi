"""
Checks evals/golden.jsonl against the chunk manifest, so a corpus or
chunking change that orphans an expected chunk id fails here, offline,
instead of silently dropping the hit rate.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = [json.loads(line) for line in (ROOT / "evals/golden.jsonl").read_text().splitlines() if line.strip()]
MANIFEST = json.loads((ROOT / "evals/corpus/manifest.json").read_text())
CHUNK_IDS = {c["chunk_id"] for c in MANIFEST["chunks"]}
REQUIRED = {"id", "category", "role", "question", "expected_chunks", "required_tools", "key_facts"}


def test_size_and_mix():
    assert len(CASES) == 40
    assert Counter(c["category"] for c in CASES) == {"answerable": 26, "unanswerable": 8, "multi_doc": 4, "authz": 2}


def test_schema_and_unique_ids():
    for c in CASES:
        assert REQUIRED <= set(c), c["id"]
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_expected_chunks_exist_in_corpus():
    missing = [(c["id"], ch) for c in CASES for ch in c["expected_chunks"] if ch not in CHUNK_IDS]
    assert missing == []


def test_category_rules():
    for c in CASES:
        if c["category"] in ("answerable", "multi_doc"):
            assert c["expected_chunks"] and c["key_facts"], c["id"]
        else:
            assert c["expected_chunks"] == [], c["id"]
        if c["category"] == "multi_doc":
            assert len({ch.split("#")[0] for ch in c["expected_chunks"]}) >= 2, c["id"]
        if c["category"] == "authz":
            assert c["role"] == "reader" and "ingest_document" in c["forbidden_tools"]


HOLDOUT = [json.loads(line) for line in (ROOT / "evals/holdout.jsonl").read_text().splitlines() if line.strip()]


def test_holdout_is_valid_and_disjoint_from_golden():
    assert len(HOLDOUT) == 12
    assert [c for c in HOLDOUT for ch in c["expected_chunks"] if ch not in CHUNK_IDS] == []
    assert not {c["id"] for c in HOLDOUT} & {c["id"] for c in CASES}
    golden_chunks = {ch for c in CASES for ch in c["expected_chunks"]}
    assert not {ch for c in HOLDOUT for ch in c["expected_chunks"]} & golden_chunks
    for c in HOLDOUT:
        assert REQUIRED <= set(c), c["id"]
