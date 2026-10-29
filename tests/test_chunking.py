"""Unit tests for app/services/chunking.py. Pure text in, chunks out."""

import pytest

from app.services.chunking import chunk_document, clean_markdown

MDX = """---
title: Persistence
description: something
---

{/* anchor stubs */}
<a id="checkpoints" />
import Foo from '/snippets/foo.mdx';

Intro paragraph about [checkpointers](/oss/checkpointers) and @[`interrupt`].

## Quickstart

:::python
Python prose stays.
```python
graph = builder.compile()
```
:::

:::js
JS prose goes.
:::

<Note>
Inside a note.
</Note>

## Second { #second }

{* ../../docs_src/x.py hl[1] *}

Last section text.
"""


def test_clean_markdown_strips_markup_but_keeps_prose():
    title, text = clean_markdown(MDX)
    assert title == "Persistence"
    for gone in ("```", "builder.compile", ":::", "JS prose goes", "<Note>", "{/*", "import Foo",
                 "{*", "{ #second }", "/oss/checkpointers", "@["):
        assert gone not in text, gone
    for kept in ("Intro paragraph about checkpointers and `interrupt`.", "Python prose stays.",
                 "Inside a note.", "## Second", "Last section text."):
        assert kept in text, kept


def test_title_falls_back_to_first_h1():
    title, _ = clean_markdown("# Background Tasks { #background-tasks }\n\nBody.")
    assert title == "Background Tasks"


def test_chunks_carry_title_and_section_header():
    chunks = chunk_document("doc", MDX)
    assert [c.chunk_id for c in chunks] == [f"doc#{i}" for i in range(len(chunks))]
    sections = [c.section for c in chunks]
    assert sections == ["Persistence", "Quickstart", "Second"]
    assert chunks[1].text.startswith("Persistence > Quickstart\n\n")
    assert chunks[0].text.startswith("Persistence\n\n")


def test_respects_max_chars_and_loses_no_paragraphs():
    paras = [f"Paragraph number {i} " + "word " * 30 for i in range(20)]
    raw = "# T\n\n" + "\n\n".join(paras)
    chunks = chunk_document("d", raw, max_chars=400, overlap_paragraphs=0)
    assert len(chunks) > 1
    for c in chunks:
        body = c.text.split("\n\n", 1)[1]
        assert len(body) <= 400
    joined = "\n\n".join(c.text for c in chunks)
    for i in range(20):
        assert f"Paragraph number {i} " in joined


def test_overlap_repeats_last_paragraph():
    paras = [f"P{i} " + "x" * 150 for i in range(6)]
    chunks = chunk_document("d", "# T\n\n" + "\n\n".join(paras), max_chars=400, overlap_paragraphs=1)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_para = prev.text.split("\n\n")[-1]
        assert nxt.text.split("\n\n")[1] == last_para


def test_oversized_paragraph_is_split():
    long_para = " ".join(f"Sentence {i} is here." for i in range(200))
    chunks = chunk_document("d", "# T\n\n" + long_para, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text.split("\n\n", 1)[1]) <= 300 for c in chunks)


def test_single_word_longer_than_max_is_hard_cut():
    chunks = chunk_document("d", "# T\n\n" + "a" * 1000, max_chars=300, overlap_paragraphs=0)
    assert "".join(c.text.split("\n\n", 1)[1] for c in chunks) == "a" * 1000


def test_empty_and_markup_only_documents_give_no_chunks():
    assert chunk_document("d", "") == []
    assert chunk_document("d", "```python\nx = 1\n```\n") == []


def test_unicode_survives():
    chunks = chunk_document("d", "# Título\n\nCafé ✅ 🔴 naïve résumé")
    assert "Café ✅ 🔴 naïve résumé" in chunks[0].text


def test_rejects_non_positive_max_chars():
    with pytest.raises(ValueError):
        chunk_document("d", "# T\n\nx", max_chars=0)
