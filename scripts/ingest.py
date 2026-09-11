"""Ingest: chunk -> embed -> pgvector.

The chunking strategy comes from PUB17_CHUNKER (see pub17/chunking.py), so the
same script loads the naive baseline or the section-aware chunks.

Re-ingesting a publication deletes its old chunks and inserts the new ones in a
single transaction. That keeps it idempotent across chunkers too: switching
strategy changes the chunk count, and an upsert keyed on chunk_index would
leave the old strategy's tail behind.

Usage:  PUB17_CHUNKER=section python3 scripts/ingest.py [--reset] [pdf_stem ...]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import config
from pub17.chunking import chunk_pdf
from pub17.store import SCHEMA, connect, embedder


def ingest_one(conn, model, stem, label):
    chunks = chunk_pdf(stem)

    # bge embeds documents bare; only queries get the instruction prefix.
    vecs = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    source = f"{stem}.pdf"
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_file = %s", (source,))
        cur.executemany(
            """
            INSERT INTO chunks
                (publication, source_file, page_start, page_end, section,
                 chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (label, source, c["page_start"], c["page_end"], c["section"], i, c["text"], vec)
                for i, (c, vec) in enumerate(zip(chunks, vecs))
            ],
        )
    return sum(len(c["text"].split()) for c in chunks), len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="pdf stems to ingest (default: all)")
    ap.add_argument("--reset", action="store_true", help="drop the chunks table first")
    args = ap.parse_args()

    targets = args.stems or list(config.PUBLICATIONS)
    unknown = [s for s in targets if s not in config.PUBLICATIONS]
    if unknown:
        sys.exit(f"unknown publication stem(s): {', '.join(unknown)}")

    print(f"chunker={config.CHUNKER}, loading {config.EMBED_MODEL} ...")
    model = embedder()
    print(f"  device: {model.device}")

    with connect() as conn:
        if args.reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(SCHEMA)

        total = 0
        started = time.perf_counter()
        for stem in targets:
            t0 = time.perf_counter()
            words, n = ingest_one(conn, model, stem, config.PUBLICATIONS[stem])
            total += n
            print(f"  {stem + '.pdf':14s} {words:>8,} words -> {n:>5,} chunks"
                  f"  ({time.perf_counter() - t0:5.1f}s)")

        # HNSW build is expensive, so only pay for it when the index is missing.
        exists = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'chunks' AND indexname = 'chunks_embedding_hnsw'"
        ).fetchone()
        if not exists:
            print("building HNSW index (cosine) ...")
            conn.execute(
                "CREATE INDEX chunks_embedding_hnsw ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )

        indexed = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    print(f"done. {total:,} chunks written, {indexed:,} in table "
          f"({time.perf_counter() - started:.1f}s total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
