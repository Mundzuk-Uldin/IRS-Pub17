"""Pick the refusal threshold from data, retrieval-only and free.

The service should decline rather than guess when retrieval found nothing
relevant. The signal is the reranker's score for the best candidate. Choosing
where to cut needs questions the corpus can't answer, and the main eval has
none, so eval/unanswerable.jsonl holds 14. Two are hard negatives whose topic
the corpus does discuss without giving the answer.

Each negative is checked first, the way the main eval is: the terms that would
carry its real answer must not appear on any page, matched as whole words. A
number that appears only on a tax-table page doesn't count -- the 1040 tables
list every income bracket in $50 steps, so almost any round figure is in them
without anything being stated.

Usage:  python3 eval/calibrate_refusal.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from pub17 import config
from pub17.rerank import rerank
from pub17.store import embedder
from baseline_numpy import build_index
from run_eval import load_questions

NEGATIVES = ROOT / "eval" / "unanswerable.jsonl"


def is_table_page(text):
    tokens = text.split()
    numeric = sum(bool(re.fullmatch(r"[\d,.$%-]+", t)) for t in tokens)
    return bool(tokens) and numeric / len(tokens) > 0.6


def stated_in(pages, term):
    """Pages where `term` appears as whole words, outside tax tables if numeric."""
    pattern = re.compile(r"(?<![\w,])" + re.escape(term) + r"(?![\w,])", re.I)
    numeric = bool(re.search(r"\d", term))
    return [(f, n) for f, n, text in pages
            if pattern.search(text) and not (numeric and is_table_page(text))]


def main():
    pages = []
    for pdf in sorted(config.RAW_DIR.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            pages += [(pdf.name, n, " ".join(p.get_text().split())) for n, p in enumerate(doc, start=1)]
    negatives = [json.loads(l) for l in NEGATIVES.read_text().splitlines() if l.strip()]
    leaks = [(n["id"], t, stated_in(pages, t)[:3]) for n in negatives for t in n["absent_terms"]
             if stated_in(pages, t)]
    if leaks:
        sys.exit("not unanswerable after all -- stated in the corpus:\n" +
                 "\n".join(f"  {i} {t!r}: {where}" for i, t, where in leaks))
    print(f"{len(negatives)} unanswerable questions verified: no answer term is stated in the corpus")

    model = embedder()
    rows, matrix = build_index(model)

    def top_score(question):
        v = model.encode(config.QUERY_PREFIX + question, normalize_embeddings=True, convert_to_numpy=True)
        pool = [rows[j] for j in np.argsort(-(matrix @ v))[:config.RERANK_CANDIDATES]]
        return rerank(question, pool, k=1)[0]["rerank_score"]

    pos = {q["id"]: top_score(q["question"]) for q in load_questions()}
    neg = {n["id"]: top_score(n["question"]) for n in negatives}

    print(f"\nanswerable   (n={len(pos)}): min {min(pos.values()):.4f}  "
          f"p5 {np.percentile(list(pos.values()), 5):.4f}  median {np.median(list(pos.values())):.4f}")
    print(f"unanswerable (n={len(neg)}): max {max(neg.values()):.4f}  median {np.median(list(neg.values())):.4f}")
    print("\nlowest answerable:", ", ".join(f"{k} {v:.4f}" for k, v in sorted(pos.items(), key=lambda x: x[1])[:6]))
    print("unanswerable:     ", ", ".join(f"{k} {v:.4f}" for k, v in sorted(neg.items(), key=lambda x: -x[1])))
    print("\n  threshold   answerable refused   unanswerable refused")
    for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  {t:9.3f}   {sum(v < t for v in pos.values()):>6} / {len(pos)}"
              f"        {sum(v < t for v in neg.values()):>6} / {len(neg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
