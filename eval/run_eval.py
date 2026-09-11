"""Score a retrieval+generation configuration against the verified eval set.

Gold is anchored on (source_file, page) plus a verbatim answer-bearing span,
never on chunk ids. Chunk ids do not survive a re-chunk, so a chunk-id gold
standard would silently invalidate every comparison the moment Weekend 2
changes the chunker -- which is the one comparison this whole file exists to
make.

A retrieved chunk counts as a hit only if it actually contains the answer:
either the verbatim gold span, or every answer key while sitting on a gold
page. The Weekend 1 rule -- page overlap alone -- is still reported as
`loose_recall_at_5`, because it overstated every configuration: a chunk on the
right page is not necessarily a chunk a model can answer from, and
section-aware chunking produces heading-sized fragments that make the gap
worse.

Usage:
  python3 eval/run_eval.py --retrieval-only          # free, no model calls
  python3 eval/run_eval.py                           # full run, costs money
  python3 eval/run_eval.py --limit 5                 # smoke test
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import config
from pub17.store import connect, retrieve

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RUNS_CSV = config.RESULTS_DIR / "runs.csv"

RUN_FIELDS = [
    "run_at", "config", "chunker", "n", "embed_model", "gen_model", "top_k",
    "words_per_chunk", "overlap", "chunks_in_store",
    "recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5", "loose_recall_at_5",
    "answer_accuracy", "citation_rate",
    "retrieve_p50_ms", "retrieve_p95_ms", "generate_p50_ms", "generate_p95_ms",
    "total_cost_usd",
]


def norm(s):
    """Strip to letters and digits, absorbing pymupdf's line-wrap hyphenation."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_questions():
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def on_gold_page(chunk, question):
    return any(
        chunk["source_file"] == fname and chunk["page_start"] <= pageno <= chunk["page_end"]
        for fname, pageno in question["gold"]
    )


def is_hit(chunk, question):
    """Does this chunk actually contain the answer?"""
    text = norm(chunk["text"])
    if norm(question["gold_span"]) in text:
        return True
    return on_gold_page(chunk, question) and all(norm(k) in text for k in question["answer_keys"])


def loose_hit(chunk, question):
    """The Weekend 1 rule: page overlap or verbatim span. Reported for comparison."""
    return on_gold_page(chunk, question) or norm(question["gold_span"]) in norm(chunk["text"])


def first_hit_rank(chunks, question, hit=is_hit):
    """1-indexed rank of the first hit, or None."""
    for rank, c in enumerate(chunks, start=1):
        if hit(c, question):
            return rank
    return None


def answer_is_correct(text, question):
    """Every required key must appear. Deterministic, no LLM judge.

    A judge would be more forgiving of paraphrase, but it would also make the
    metric depend on a second model's mood. These answers are dollar amounts,
    rates, and yes/no -- exact-key matching is the right tool.

    `answer_keys` are evidence: the verifier requires them on every gold page,
    and a strict retrieval hit must contain them. Usually the answer repeats
    the evidence, so they grade the answer too. When it needn't -- a yes/no
    question whose evidence is a dollar figure -- `grade_keys` override them
    for grading only.
    """
    body = norm(text)
    keys = question.get("grade_keys", question["answer_keys"])
    return all(norm(k) in body for k in keys)


def cites_a_source(text):
    return bool(re.search(r"\[\d+\]", text))


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


def p(values, q):
    if not values:
        return 0
    values = sorted(values)
    idx = min(int(round((q / 100) * (len(values) - 1))), len(values) - 1)
    return int(values[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="score retrieval only; makes no model calls and costs nothing")
    ap.add_argument("--limit", type=int, help="only run the first N questions")
    ap.add_argument("--config-name", default=config.CONFIG_NAME)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    generate = None
    if not args.retrieval_only:
        from pub17.generate import answer as generate

    config.RESULTS_DIR.mkdir(exist_ok=True)
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    detail_path = config.RESULTS_DIR / f"{args.config_name}-{run_at.replace(':', '')}.jsonl"

    ranks, loose_ranks, correct, cited = [], [], [], []
    retrieve_ms, generate_ms, costs = [], [], []

    with connect() as conn, detail_path.open("w") as detail:
        chunks_in_store = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if not chunks_in_store:
            sys.exit("chunks table is empty -- run scripts/ingest.py first")

        for i, q in enumerate(questions, start=1):
            t0 = time.perf_counter()
            chunks = retrieve(conn, q["question"], k=args.k)
            retrieve_ms.append((time.perf_counter() - t0) * 1000)

            rank = first_hit_rank(chunks, q)
            ranks.append(rank)
            loose_ranks.append(first_hit_rank(chunks, q, hit=loose_hit))

            row = {
                "id": q["id"],
                "topic": q["topic"],
                "kind": q["kind"],
                "question": q["question"],
                "expected": q["answer"],
                "first_hit_rank": rank,
                "retrieved": [
                    {"cite": f"{c['source_file']} p{c['page_start']}-{c['page_end']}",
                     "similarity": None if c["similarity"] is None else round(c["similarity"], 4),
                     "hit": is_hit(c, q)}
                    for c in chunks
                ],
            }

            if generate:
                result = generate(q["question"], chunks)
                ok = answer_is_correct(result["text"], q)
                correct.append(ok)
                cited.append(cites_a_source(result["text"]))
                generate_ms.append(result["generate_ms"])
                costs.append(result["cost_usd"])
                row.update(answer=result["text"], correct=ok, refused=result["refused"],
                           cost_usd=round(result["cost_usd"], 6))
                mark = "OK " if ok else "BAD"
                print(f"  [{i:>2}/{len(questions)}] {mark} rank={rank or '-':<3} {q['id']} {q['topic']}")
            else:
                print(f"  [{i:>2}/{len(questions)}] rank={rank or '-':<3} {q['id']} {q['topic']}")

            detail.write(json.dumps(row) + "\n")

    n = len(questions)
    summary = {
        "run_at": run_at,
        "config": args.config_name,
        "chunker": config.CHUNKER,
        "n": n,
        "embed_model": config.EMBED_MODEL,
        "gen_model": config.GEN_MODEL if generate else "",
        "top_k": args.k,
        "words_per_chunk": config.WORDS_PER_CHUNK,
        "overlap": config.OVERLAP,
        "chunks_in_store": chunks_in_store,
        "recall_at_1": round(pct([r == 1 for r in ranks]), 1),
        "recall_at_3": round(pct([r is not None and r <= 3 for r in ranks]), 1),
        "recall_at_5": round(pct([r is not None and r <= 5 for r in ranks]), 1),
        "mrr_at_5": round(statistics.fmean([1 / r if r else 0.0 for r in ranks]), 3),
        "loose_recall_at_5": round(pct([r is not None and r <= 5 for r in loose_ranks]), 1),
        "answer_accuracy": round(pct(correct), 1) if correct else "",
        "citation_rate": round(pct(cited), 1) if cited else "",
        "retrieve_p50_ms": p(retrieve_ms, 50),
        "retrieve_p95_ms": p(retrieve_ms, 95),
        "generate_p50_ms": p(generate_ms, 50) if generate_ms else "",
        "generate_p95_ms": p(generate_ms, 95) if generate_ms else "",
        "total_cost_usd": round(sum(costs), 4) if costs else "",
    }

    write_header = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print(f"\n{'=' * 60}\n{args.config_name}  (n={n})")
    print(f"  recall@1        {summary['recall_at_1']:>6}%")
    print(f"  recall@3        {summary['recall_at_3']:>6}%")
    print(f"  recall@5        {summary['recall_at_5']:>6}%")
    print(f"  MRR@5           {summary['mrr_at_5']:>6}")
    print(f"  loose recall@5  {summary['loose_recall_at_5']:>6}%  (page overlap only)")
    if correct:
        print(f"  answer accuracy {summary['answer_accuracy']:>6}%")
        print(f"  citation rate   {summary['citation_rate']:>6}%")
        print(f"  cost            ${summary['total_cost_usd']}")
    print(f"  retrieve p50/p95  {summary['retrieve_p50_ms']} / {summary['retrieve_p95_ms']} ms")
    if generate_ms:
        print(f"  generate p50/p95  {summary['generate_p50_ms']} / {summary['generate_p95_ms']} ms")
    print(f"\nappended to {RUNS_CSV}")
    print(f"per-question detail in {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
