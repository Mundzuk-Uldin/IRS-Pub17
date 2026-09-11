"""Score retrieval with no Postgres: same chunker, same gold logic, numpy cosine.

`run_eval.py` is the real harness -- it measures the pipeline that actually
serves queries, including the HNSW index's approximation error. This script
exists for two narrower reasons:

  * it reproduces the retrieval baseline on a clean checkout with no services
    running, which is what most people reviewing this repo will do; and
  * exhaustive cosine over every chunk is the exact upper bound on what the
    approximate index can return, so a gap between the two numbers is HNSW
    recall loss rather than a retrieval-quality change.

It imports the chunker and the hit logic from the real pipeline rather than
reimplementing them, so the only thing that differs is the vector search.

Usage:  python3 eval/baseline_numpy.py [--csv]
"""
import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pub17 import config
from pub17.chunking import chunk_pdf
from pub17.store import embedder
from run_eval import RUNS_CSV, RUN_FIELDS, first_hit_rank, load_questions, loose_hit


def build_index(model):
    rows = []
    for stem, label in config.PUBLICATIONS.items():
        for i, chunk in enumerate(chunk_pdf(stem)):
            rows.append(dict(chunk, publication=label, source_file=f"{stem}.pdf", chunk_index=i))
    matrix = model.encode(
        [r["text"] for r in rows], batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return rows, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="append the run to results/runs.csv")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    model = embedder()
    print(f"chunker={config.CHUNKER}, embedding corpus on {model.device} ...")
    rows, matrix = build_index(model)
    print(f"  {len(rows):,} chunks")

    questions = load_questions()
    qmat = model.encode(
        [config.QUERY_PREFIX + q["question"] for q in questions],
        batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
    )
    sims = qmat @ matrix.T

    ranks, loose_ranks = [], []
    by_topic, hit_topic = Counter(), Counter()
    misses = []
    for qi, q in enumerate(questions):
        top = np.argsort(-sims[qi])[:args.k]
        chunks = [dict(rows[j], similarity=float(sims[qi][j])) for j in top]
        rank = first_hit_rank(chunks, q)
        ranks.append(rank)
        loose_ranks.append(first_hit_rank(chunks, q, hit=loose_hit))
        by_topic[q["topic"]] += 1
        if rank:
            hit_topic[q["topic"]] += 1
        else:
            misses.append((q, chunks))

    def pct(pred):
        return 100.0 * sum(1 for r in ranks if pred(r)) / len(ranks)

    r1 = pct(lambda r: r == 1)
    r3 = pct(lambda r: r is not None and r <= 3)
    r5 = pct(lambda r: r is not None and r <= 5)
    mrr = float(np.mean([1 / r if r else 0.0 for r in ranks]))
    loose5 = 100.0 * sum(1 for r in loose_ranks if r is not None and r <= 5) / len(loose_ranks)

    print(f"\n{config.CONFIG_NAME} (numpy exhaustive, n={len(questions)})")
    print(f"  recall@1 {r1:.1f}%   recall@3 {r3:.1f}%   recall@5 {r5:.1f}%   MRR@5 {mrr:.3f}")
    print(f"  loose recall@5 {loose5:.1f}%  (page overlap only)")
    print("\n  per topic (hit/total):")
    for topic in sorted(by_topic):
        print(f"    {topic:22s} {hit_topic[topic]}/{by_topic[topic]}")
    print(f"\n  {len(misses)} misses:")
    for q, chunks in misses:
        got = ", ".join(f"{c['source_file']} p{c['page_start']}" for c in chunks)
        print(f"    {q['id']} [{q['topic']}] {q['question'][:66]}")
        print(f"        got: {got}")

    if args.csv:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        row = {f: "" for f in RUN_FIELDS}
        row.update(
            run_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            config=f"{config.CONFIG_NAME}-numpy", chunker=config.CHUNKER,
            n=len(questions), embed_model=config.EMBED_MODEL, top_k=args.k,
            words_per_chunk=config.WORDS_PER_CHUNK, overlap=config.OVERLAP,
            chunks_in_store=len(rows),
            recall_at_1=round(r1, 1), recall_at_3=round(r3, 1),
            recall_at_5=round(r5, 1), mrr_at_5=round(mrr, 3),
            loose_recall_at_5=round(loose5, 1),
        )
        write_header = not RUNS_CSV.exists()
        with RUNS_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"\nappended to {RUNS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
