"""pgvector-backed chunk store and dense retrieval.

`retrieve` is the entry point callers use: dense cosine search, fused with
Postgres full-text search when config.HYBRID is on, then cross-encoder
reranking when config.RERANK is on. Every stage returns the same
list of chunk dicts, so the eval harness doesn't change as stages are added.
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

-- Full-text side of hybrid retrieval. Generated, so existing rows are backfilled
-- when the column is added and every insert keeps it current.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Generated section context, embedded ahead of the text but stored apart from it.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT NOT NULL DEFAULT '';

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
    try:
        conn = psycopg.connect(config.DSN, autocommit=autocommit)
    except psycopg.OperationalError as exc:
        raise SystemExit(
            f"{exc}\n\nCan't reach Postgres. Start the bundled one with:\n"
            "    docker compose up -d"
        ) from None
    try:
        # The vector type must exist before register_vector can look it up, so
        # on a fresh database the extension has to be created first.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
    except psycopg.Error as exc:
        conn.close()
        raise SystemExit(
            f"{exc}\n\nThis server has no pgvector. Use the bundled container "
            "(docker compose up -d), or install postgresql-18-pgvector on a "
            "native server and run ./scripts/setup_db.sh."
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
        f"""
        SELECT {_CHUNK_COLS}, 1 - (embedding <=> %s) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, k),
    ).fetchall()
    return _as_dicts(rows, "similarity")


_CHUNK_COLS = "id, publication, source_file, page_start, page_end, text, context"


def _as_dicts(rows, score_name):
    # Every chunk carries a `similarity` key; it is None for a chunk only the
    # full-text side found, since there is no cosine score to report for it.
    return [
        {
            "id": r[0], "publication": r[1], "source_file": r[2],
            "page_start": r[3], "page_end": r[4], "text": r[5], "context": r[6],
            "similarity": None,
            score_name: float(r[7]),
        }
        for r in rows
    ]


def search_lexical(conn, question, k):
    """Top-k chunks by Postgres full-text rank.

    plainto_tsquery ANDs every word, and a full-sentence question almost never
    has all its words in one chunk, so the terms are OR-ed instead and ts_rank_cd
    does the ranking. Stop words are already gone by then.
    """
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS q
        )
        SELECT {_CHUNK_COLS}, ts_rank_cd(tsv, q.q) AS rank
        FROM chunks, q
        WHERE tsv @@ q.q
        ORDER BY rank DESC
        LIMIT %s
        """,
        (question, k),
    ).fetchall()
    return _as_dicts(rows, "lexical_rank")


def search_hybrid(conn, question, k):
    """Reciprocal Rank Fusion of dense and full-text results.

    RRF scores by rank position alone, 1 / (RRF_K + rank), so cosine similarity
    and ts_rank never have to be put on one scale.
    """
    depth = config.HYBRID_DEPTH
    fused = {}
    for results in (search_dense(conn, question, k=depth), search_lexical(conn, question, depth)):
        for rank, chunk in enumerate(results, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "rrf": 0.0})
            entry["rrf"] += 1.0 / (config.RRF_K + rank)
    return sorted(fused.values(), key=lambda c: -c["rrf"])[:k]


def retrieve(conn, question, k=None):
    """Dense or hybrid candidates, then cross-encoder reranking when on."""
    k = k or config.TOP_K
    pool = max(k, config.RERANK_CANDIDATES) if config.RERANK else k
    search = search_hybrid if config.HYBRID else search_dense
    candidates = search(conn, question, k=pool)
    if not config.RERANK:
        return candidates
    # Imported lazily so the dense-only path never loads the cross-encoder.
    from .rerank import rerank
    return rerank(question, candidates, k=k)


def cite(chunk):
    """Short human-readable citation, e.g. 'Pub 502, p.13'."""
    pages = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    return f"{chunk['publication']}, {pages}"
