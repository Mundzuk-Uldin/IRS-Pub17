"""Weekend 1 end to end: type a tax question, get a cited answer.

Usage:  python3 scripts/ask.py "what is the standard deduction for a single filer?"
        python3 scripts/ask.py --show-chunks "how much of my medical expenses can I deduct?"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import config
from pub17.generate import answer
from pub17.store import cite, connect, retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--show-chunks", action="store_true", help="print the retrieved text")
    ap.add_argument("--retrieve-only", action="store_true", help="skip the model call")
    args = ap.parse_args()

    question = " ".join(args.question)

    t0 = time.perf_counter()
    with connect() as conn:
        chunks = retrieve(conn, question, k=args.k)
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    print(f"\nQ: {question}")
    print("=" * 78)

    if not chunks:
        print("\nNothing retrieved -- has the corpus been ingested?")
        return 1

    print(f"\nRetrieved {len(chunks)} chunks in {retrieve_ms} ms:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] sim={c['similarity']:.3f}  {cite(c)}")
        if args.show_chunks:
            print(f"      {' '.join(c['text'].split())[:400]}...")

    if args.retrieve_only:
        return 0

    result = answer(question, chunks)
    print("\n" + "-" * 78)
    print(result["text"])
    print("-" * 78)
    print(
        f"{result['input_tokens']:,} in / {result['output_tokens']:,} out"
        f"  ${result['cost_usd']:.4f}"
        f"  retrieve {retrieve_ms} ms, generate {result['generate_ms']} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
