"""Do unanswerable questions get refused? Makes model calls, ~$0.01 each on Sonnet.

The reranker gate (config.MIN_RERANK_SCORE) only catches clearly off-topic
questions, 3 of the 14 in eval/unanswerable.jsonl. This sends all 14 through
the same path /ask uses -- retrieve, rerank, gate, then generate whatever the
gate lets through -- and records every answer.

A decline is flagged by phrasing, since the system prompt tells the model to
say plainly when the excerpts don't contain the answer. Phrase matching isn't
trusted alone: every generated answer is printed so each call can be read.

Usage:  python3 eval/refusal_check.py
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pub17 import config
from pub17.generate import answer
from pub17.rerank import rerank
from pub17.store import connect, search_dense

DECLINE = re.compile(
    r"\b(excerpts?|material|passages?|sources?|provided)\b[^.]{0,90}"
    r"\b(don.?t|do not|doesn.?t|does not|never|no|nothing)\b"
    r"|\b(can.?t|cannot|unable to) (answer|determine|find|say)"
    r"|\bnot (covered|addressed|mentioned|included|stated|provided|given)\b",
    re.I,
)


def main():
    negatives = [json.loads(l) for l in (ROOT / "eval" / "unanswerable.jsonl").read_text().splitlines() if l.strip()]
    rows, spent = [], 0.0
    with connect() as conn:
        for n in negatives:
            chunks = rerank(n["question"], search_dense(conn, n["question"], k=config.RERANK_CANDIDATES))
            top = chunks[0]["rerank_score"]
            row = {"id": n["id"], "question": n["question"], "top_rerank_score": round(top, 4)}
            if top < config.MIN_RERANK_SCORE:
                rows.append({**row, "outcome": "gate", "answer": None})
                continue
            result = answer(n["question"], chunks)
            spent += result["cost_usd"]
            outcome = ("model_refusal" if result["refused"]
                       else "declined" if DECLINE.search(result["text"]) else "ANSWERED")
            rows.append({**row, "outcome": outcome, "answer": result["text"],
                         "cost_usd": round(result["cost_usd"], 6)})

    for r in rows:
        print(f"\n[{r['id']}] {r['outcome']:<13} top={r['top_rerank_score']:.4f}  {r['question']}")
        if r["answer"]:
            print("   " + " ".join(r["answer"].split())[:420])

    counts = {k: sum(r["outcome"] == k for r in rows) for k in ("gate", "declined", "model_refusal", "ANSWERED")}
    print(f"\n{len(rows)} unanswerable: {counts['gate']} stopped by the gate, "
          f"{counts['declined']} declined by the model, {counts['model_refusal']} model refusals, "
          f"{counts['ANSWERED']} answered anyway   (${spent:.4f}, {config.GEN_MODEL})")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = config.RESULTS_DIR / f"refusal-check-{stamp}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"answers saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
