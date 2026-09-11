"""pgvector-backed chunk store and dense retrieval.

Weekend 1 is dense cosine search only. Weekend 2 adds tsvector, RRF, and a
cross-encoder alongside `search_dense`; the signature it returns is meant to
stay the same so the eval harness doesn't have to change.
"""
import functools

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from . import config

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    publication  TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    page_start   INT  NOT NULL,
    page_end     INT  NOT NULL,
    section      TEXT NOT NULL DEFAULT '',
    chunk_index  INT  NOT NULL,
    text         TEXT NOT NULL,
    embedding    vector({config.EMBED_DIM}),
    UNIQUE (source_file, chunk_index)
);

CREATE TABLE IF NOT EXISTS query_log (
    id             BIGSERIAL PRIMARY KEY,
    asked_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    config_name    TEXT NOT NULL,
    question       TEXT NOT NULL,
    answer         TEXT,
    chunk_ids      BIGINT[],
    top_similarity REAL,
    input_tokens   INT,
    output_tokens  INT,
    cost_usd       NUMERIC(10, 6),
    retrieve_ms    INT,
    generate_ms    INT,
    refused        BOOLEAN NOT NULL DEFAULT FALSE
);
"""


def connect(autocommit=True):
    conn = psycopg.connect(config.DSN, autocommit=autocommit)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as exc:
        conn.close()
        raise SystemExit(
            f"{exc}\n\n"
            "The pub17 database has no pgvector. Install it and create the extension:\n"
            "    sudo apt-get install -y postgresql-18-pgvector\n"
            "    ./scripts/setup_db.sh"
        ) from None
    return conn


@functools.lru_cache(maxsize=1)
def embedder():
    """Loading the model takes seconds, so do it once per process."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed_query(question):
    return embedder().encode(
        config.QUERY_PREFIX + question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def search_dense(conn, question, k=None):
    """Top-k chunks by cosine similarity. Returns a list of dicts."""
    k = k or config.TOP_K
    qvec = embed_query(question)
    rows = conn.execute(
        """
        SELECT id, publication, source_file, page_start, page_end, text,
               1 - (embedding <=> %s) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, k),
    ).fetchall()
    return [
        {
            "id": r[0],
            "publication": r[1],
            "source_file": r[2],
            "page_start": r[3],
            "page_end": r[4],
            "text": r[5],
            "similarity": float(r[6]),
        }
        for r in rows
    ]


def cite(chunk):
    """Short human-readable citation, e.g. 'Pub 502, p.13'."""
    pages = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    return f"{chunk['publication']}, {pages}"
