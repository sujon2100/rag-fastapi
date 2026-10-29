"""
Chunk the eval corpus and upsert it into a dedicated Pinecone namespace.

  python -m scripts.ingest_corpus                    # chunk + upsert into eval-v1
  python -m scripts.ingest_corpus --dry-run          # chunk + write manifest only
  python -m scripts.ingest_corpus --reset            # clear the namespace first

Vector ids are the chunk ids ("fastapi-cors#3"), so re-running is idempotent.
--reset exists for when chunking changes and old ids would otherwise linger.

Always writes evals/corpus/manifest.json (chunk ids, sections, text hashes),
so the golden set can be checked against the corpus without touching Pinecone.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from app.services.chunking import chunk_document

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "evals" / "corpus" / "raw"
MANIFEST = REPO_ROOT / "evals" / "corpus" / "manifest.json"
BATCH = 50


def build_chunks(max_chars: int):
    chunks = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix in {".md", ".mdx"}:
            chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), max_chars=max_chars))
    return chunks


def write_manifest(chunks, namespace: str, max_chars: int):
    MANIFEST.write_text(
        json.dumps(
            {
                "namespace": namespace,
                "max_chars": max_chars,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "section": c.section,
                        "chars": len(c.text),
                        "sha1": hashlib.sha1(c.text.encode()).hexdigest()[:12],
                    }
                    for c in chunks
                ],
            },
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="eval-v1")
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args(argv)

    if not args.namespace:
        sys.exit("refusing to write eval data into the default namespace")

    chunks = build_chunks(args.max_chars)
    write_manifest(chunks, args.namespace, args.max_chars)
    docs = sorted({c.doc_id for c in chunks})
    print(f"{len(chunks)} chunks from {len(docs)} documents, manifest at {MANIFEST.relative_to(REPO_ROOT)}")
    if args.dry_run:
        return

    from app.services.embeddings import get_embedder
    from app.services.vectorstore import get_index

    index = get_index()
    if args.reset:
        stats = index.describe_index_stats()
        if args.namespace in (stats.namespaces or {}):
            index.delete(delete_all=True, namespace=args.namespace)
            print(f"cleared namespace {args.namespace}")

    t0 = time.perf_counter()
    vectors = get_embedder().encode([c.text for c in chunks], batch_size=32).tolist()
    print(f"embedded in {time.perf_counter() - t0:.1f}s")

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        index.upsert(
            vectors=[
                {
                    "id": c.chunk_id,
                    "values": v,
                    "metadata": {"text": c.text, "source": c.doc_id, "chunk_id": c.chunk_id, "section": c.section},
                }
                for c, v in zip(batch, vectors[start:start + BATCH])
            ],
            namespace=args.namespace,
        )
    print(f"upserted {len(chunks)} vectors into namespace {args.namespace}")


if __name__ == "__main__":
    main()
