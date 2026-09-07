"""Weekend 1 ingest: parse -> naive fixed-window chunk -> embed -> pgvector.

The chunking here is deliberately unstructured: fixed word windows that run
straight through headings and tables. It is the baseline the Weekend 2
section-aware chunker has to beat, so resist improving it.

Upserts are idempotent on (source_file, chunk_index), so a re-run replaces a
publication in place rather than duplicating it.

Usage:  python3 scripts/ingest.py [--reset] [pdf_stem ...]
"""
import argparse
import sys
import time
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import config
from pub17.store import SCHEMA, connect, embedder


def read_pdf(path):
    """Return [(word, page_number), ...] so a chunk knows which pages it spans."""
    tokens = []
    with fitz.open(path) as doc:
        for pageno, page in enumerate(doc, start=1):
            for word in page.get_text("text").split():
                tokens.append((word, pageno))
    return tokens


def chunk_document(tokens):
    """Fixed word windows with overlap. No structure awareness -- that is the point.

    Yields (text, page_start, page_end).
    """
    step = config.WORDS_PER_CHUNK - config.OVERLAP
    for i in range(0, len(tokens), step):
        window = tokens[i:i + config.WORDS_PER_CHUNK]
        if len(window) < 20:
            break
        yield " ".join(w for w, _ in window), window[0][1], window[-1][1]


def ingest_one(conn, model, stem, label):
    path = config.RAW_DIR / f"{stem}.pdf"
    tokens = read_pdf(path)
    chunks = list(chunk_document(tokens))

    # bge embeds documents bare; only queries get the instruction prefix.
    vecs = model.encode(
        [c[0] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (publication, source_file, page_start, page_end, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file, chunk_index) DO UPDATE SET
                publication = EXCLUDED.publication,
                page_start  = EXCLUDED.page_start,
                page_end    = EXCLUDED.page_end,
                text        = EXCLUDED.text,
                embedding   = EXCLUDED.embedding
            """,
            [
                (label, f"{stem}.pdf", start, end, i, text, vec)
                for i, ((text, start, end), vec) in enumerate(zip(chunks, vecs))
            ],
        )
    return len(tokens), len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="pdf stems to ingest (default: all)")
    ap.add_argument("--reset", action="store_true", help="drop the chunks table first")
    args = ap.parse_args()

    targets = args.stems or list(config.PUBLICATIONS)
    unknown = [s for s in targets if s not in config.PUBLICATIONS]
    if unknown:
        sys.exit(f"unknown publication stem(s): {', '.join(unknown)}")

    print(f"loading {config.EMBED_MODEL} ...")
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
