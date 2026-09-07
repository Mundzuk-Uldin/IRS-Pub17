"""Weekend 1 retrieval: embed the question, cosine top-k out of pgvector.

Usage:  python3 query.py "what is the standard deduction for a single filer?"
"""
import sys
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:pub17@localhost:5433/pub17"
MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5

# bge is trained asymmetrically: queries take this prefix, documents do not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(question, k=TOP_K):
    model = SentenceTransformer(MODEL)
    qvec = model.encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT publication, page, text, 1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, qvec, k),
        ).fetchall()
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    question = " ".join(sys.argv[1:])
    print(f"\nQ: {question}\n" + "=" * 78)
    for rank, (pub, page, text, sim) in enumerate(search(question), start=1):
        snippet = " ".join(text.split())[:400]
        print(f"\n[{rank}] sim={sim:.3f}  {pub}, p.{page}")
        print(f"    {snippet}...")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
