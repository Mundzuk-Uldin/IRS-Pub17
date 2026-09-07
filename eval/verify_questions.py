"""Verify every eval question is actually grounded in the corpus.

A wrong eval set is worse than no eval set, so each question is checked against
the PDFs themselves before it is allowed to score anything:

  1. every gold page exists in the named PDF;
  2. `gold_span` appears verbatim on at least one gold page;
  3. every gold page independently contains every string in `answer_keys`.

Check 3 is the strict one, and it encodes what "gold" means here: a page earns
its place only if a careful reader could answer the question from that page
alone. The IRS corpus duplicates heavily -- Pub 17 chapter 10 restates Pub
501's standard deduction charts and the 1040 instructions restate both -- so
most questions have several genuinely correct source pages. Listing only one
would score a retriever wrong for finding a different correct answer, which
would quietly corrupt every configuration comparison.

Comparison strips everything but letters and digits, which absorbs the
line-wrap hyphenation and column whitespace that pymupdf leaves in IRS text
("informa-\\ntion" and "information" compare equal).

Usage:  python3 eval/verify_questions.py
Exit code is nonzero if any question fails, so CI can gate on it.
"""
import json
import re
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_pages():
    """{filename: {page_number: normalized_text}} for every PDF in the corpus."""
    pages = {}
    for pdf in sorted(RAW.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            pages[pdf.name] = {
                i: norm(page.get_text("text")) for i, page in enumerate(doc, start=1)
            }
    return pages


def check(q, pages):
    """Return a list of human-readable problems with this question."""
    problems = []
    golds = []
    for fname, pageno in q["gold"]:
        if fname not in pages:
            problems.append(f"no such file {fname}")
        elif pageno not in pages[fname]:
            problems.append(f"{fname} has no page {pageno}")
        else:
            golds.append(pages[fname][pageno])

    if not golds:
        return problems

    if not any(norm(q["gold_span"]) in g for g in golds):
        problems.append(f"gold_span not found on any gold page: {q['gold_span']!r}")

    for fname, pageno in q["gold"]:
        page = pages.get(fname, {}).get(pageno)
        if page is None:
            continue
        missing = [k for k in q["answer_keys"] if norm(k) not in page]
        if missing:
            problems.append(
                f"{fname} p{pageno} does not contain {missing} -- "
                "a gold page must independently answer the question"
            )

    return problems


def main():
    pages = load_pages()
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]

    ids = [q["id"] for q in questions]
    failed = 0
    if len(set(ids)) != len(ids):
        print("FAIL duplicate question ids")
        failed += 1

    for q in questions:
        problems = check(q, pages)
        if problems:
            failed += 1
            print(f"FAIL {q['id']}  {q['question'][:60]}")
            for p in problems:
                print(f"       {p}")

    topics = {}
    for q in questions:
        topics[q["topic"]] = topics.get(q["topic"], 0) + 1
    kinds = {}
    for q in questions:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1

    print(f"\n{len(questions)} questions, {failed} failing")
    print("topics:", ", ".join(f"{k}={v}" for k, v in sorted(topics.items())))
    print("kinds: ", ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
