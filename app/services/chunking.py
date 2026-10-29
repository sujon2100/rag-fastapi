"""
Markdown/MDX cleaning and heading-aware chunking.

Before this, ingest_document embedded a whole document as one vector. That
works for one-line canaries but not for real pages: all-MiniLM-L6-v2 only
reads the first 256 word pieces (roughly 800-1000 characters of English) and
silently drops the rest.

Chunks are built per section (split on markdown headings), then paragraphs
are packed up to max_chars. Each chunk starts with "Title > Section" so a
chunk that only says "Use it like this:" still embeds near its topic.
"""

import re
from dataclasses import dataclass

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_FENCE = re.compile(r"^ *```.*?^ *```[^\n]*\n?", re.S | re.M)
_JS_BLOCK = re.compile(r"^[ \t]*:::js[ \t]*\n.*?^[ \t]*:::[ \t]*$\n?", re.S | re.M)
_LANG_MARKER = re.compile(r"^[ \t]*:::\w*[ \t]*$\n?", re.M)
_REF = re.compile(r"@\[(`?[^\]]+`?)\]")  # LangGraph docs API refs: @[`interrupt`]
_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_INCLUDE = re.compile(r"^\{\*.*?\*\}\n?", re.M)
_IMPORT = re.compile(r"^import .*\n?", re.M)
_JSX_TAG = re.compile(r"</?[A-Za-z][\w.]*(?:\s[^<>]*?)?/?>")
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str


def clean_markdown(raw: str) -> tuple[str, str]:
    """Return (title, cleaned_text). Title comes from frontmatter or first H1."""
    title = ""
    m = _FRONTMATTER.match(raw)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        if t:
            title = t.group(1).strip().strip("\"'")
        raw = raw[m.end():]

    text = _JS_BLOCK.sub("", raw)
    text = _FENCE.sub("", text)
    text = _LANG_MARKER.sub("", text)
    text = _MDX_COMMENT.sub("", text)
    text = _INCLUDE.sub("", text)
    text = _IMPORT.sub("", text)
    text = _JSX_TAG.sub("", text)
    text = _HEADING_ANCHOR.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _REF.sub(r"\1", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not title:
        h1 = re.search(r"^#\s+(.*)$", text, re.M)
        if h1:
            title = h1.group(1).strip()
    return title, text


def _sections(text: str, title: str) -> list[tuple[str, str]]:
    """Split on headings. Returns (section_name, body) pairs, skipping empty bodies."""
    out = []
    current = title
    last = 0
    for m in _HEADING.finditer(text):
        body = text[last:m.start()].strip()
        if body:
            out.append((current, body))
        level, name = len(m.group(1)), m.group(2).strip()
        current = title if level == 1 else name
        last = m.end()
    body = text[last:].strip()
    if body:
        out.append((current, body))
    return out


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph on sentence ends, then hard-cut if needed."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, buf = [], ""
    for s in sentences:
        while len(s) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(s[:max_chars])
            s = s[max_chars:]
        if buf and len(buf) + 1 + len(s) > max_chars:
            pieces.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def chunk_document(
    doc_id: str,
    raw: str,
    max_chars: int = 800,
    overlap_paragraphs: int = 1,
) -> list[Chunk]:
    """
    Clean a markdown/MDX document and cut it into chunks of at most max_chars
    body characters (the "Title > Section" header is added on top). The last
    `overlap_paragraphs` paragraphs of a chunk are repeated at the start of
    the next chunk in the same section, if they fit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    title, text = clean_markdown(raw)
    title = title or doc_id
    chunks: list[Chunk] = []

    for section, body in _sections(text, title):
        paragraphs = []
        for p in re.split(r"\n\s*\n", body):
            p = p.strip()
            if p:
                paragraphs.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])

        current: list[str] = []
        for p in paragraphs:
            if current and len("\n\n".join(current + [p])) > max_chars:
                chunks.append(_make(doc_id, len(chunks), title, section, current))
                keep = current[-overlap_paragraphs:] if overlap_paragraphs else []
                current = keep if len("\n\n".join(keep + [p])) <= max_chars else []
            current.append(p)
        if current:
            chunks.append(_make(doc_id, len(chunks), title, section, current))

    return chunks


def _make(doc_id: str, n: int, title: str, section: str, paragraphs: list[str]) -> Chunk:
    header = title if section == title else f"{title} > {section}"
    return Chunk(
        chunk_id=f"{doc_id}#{n}",
        doc_id=doc_id,
        section=section,
        text=f"{header}\n\n" + "\n\n".join(paragraphs),
    )
