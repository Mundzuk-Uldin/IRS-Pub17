"""Generate one short retrieval context per document section.

Weekend 2 change 2, contextual enrichment. A chunk cut from the middle of a
section often never names what it is about -- the premium tax credit examples
in Pub 502 are arithmetic with no subject. Before embedding, each chunk gets
the generated context of its parent section prepended (PUB17_CHUNKER=
section-context), so the embedding carries the topic the chunk text leaves out.

Contexts are generated once per section, not per chunk: 675 calls instead of
1,508, and every chunk of a section shares one context. Results are cached in
data/section_summaries.jsonl, keyed by publication, heading path, and a hash
of the section's text, so a re-run only pays for sections that are new or
changed, and an interrupted run resumes where it stopped.

The prompt is the generic contextual-retrieval instruction. It deliberately
doesn't mention abbreviations or any specific failure in the eval set, since
that would be tuning the enrichment to the questions it is scored on.

Usage:
  python3 scripts/summarize_sections.py --dry-run     # count and cost, no calls
  python3 scripts/summarize_sections.py [--limit N] [--workers 8]
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import config
from pub17.chunking import _sections
from pub17.generate import PRICES, cost_usd, create_message

OUT = config.ROOT / "data" / "section_summaries.jsonl"

PROMPT = """<section>
{text}
</section>

This section is from {label}, tax year {year}, under the heading path:
{path}

Write a short context, at most 60 words, that situates this section within the publication, for the purpose of improving search retrieval of passages taken from it. Answer only with the context."""


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_sections():
    """[{stem, section, text}], one per (publication, heading path).

    A heading path that occurs twice in a publication is treated as one
    section, since the chunks only carry the path and couldn't tell them apart.
    """
    merged = {}
    for stem, label in config.PUBLICATIONS.items():
        with fitz.open(config.RAW_DIR / f"{stem}.pdf") as doc:
            for path, blocks in _sections(doc, label):
                merged.setdefault((stem, path), []).extend(t for _, t in blocks)
    return [{"stem": s, "section": p, "text": " ".join(ts)} for (s, p), ts in merged.items()]


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["stem"], r["section"], r["digest"]))
    return done


def summarize(client, sec):
    response = create_message(
        client,
        model=config.GEN_MODEL,
        # Room for adaptive thinking on top of the ~80-token answer.
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROMPT.format(
            text=sec["text"], label=config.PUBLICATIONS[sec["stem"]],
            year=config.TAX_YEAR, path=sec["section"],
        )}],
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("empty context")
    return text, response.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count sections and estimate cost only")
    ap.add_argument("--limit", type=int, help="only generate the first N outstanding sections")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sections = load_sections()
    done = load_done()
    todo = [s for s in sections if (s["stem"], s["section"], digest(s["text"])) not in done]
    if args.limit:
        todo = todo[:args.limit]

    words = sum(len(s["text"].split()) for s in todo)
    # Rates measured on the first full run: 821k input tokens for 338k words,
    # 118 output tokens per call. Tax text, with its figures and form numbers,
    # tokenizes at ~2.1 tokens a word; an earlier 1.35 guess came in 47% low.
    price_in, price_out = PRICES[config.GEN_MODEL]
    estimate = ((words * 2.13 + len(todo) * 150) * price_in + len(todo) * 120 * price_out) / 1_000_000
    print(f"{len(sections)} sections, {len(sections) - len(todo) if not args.limit else len(done)} cached, "
          f"{len(todo)} to generate ({words:,} words), estimated ${estimate:.2f} on {config.GEN_MODEL}")
    if args.dry_run or not todo:
        return 0

    import anthropic
    client = anthropic.Anthropic(max_retries=5)

    spent, written, failed = 0.0, 0, []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Only this loop writes the file, so worker threads never contend for it.
    with OUT.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(summarize, client, s): s for s in todo}
        for future in as_completed(futures):
            sec = futures[future]
            try:
                context, usage = future.result()
            except Exception as exc:
                failed.append((sec["stem"], sec["section"], f"{type(exc).__name__}: {str(exc)[:160]}"))
                continue
            spent += cost_usd(usage)
            written += 1
            out.write(json.dumps({
                "stem": sec["stem"], "section": sec["section"], "digest": digest(sec["text"]),
                "context": context,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            }) + "\n")
            out.flush()
            if written % 50 == 0:
                print(f"  {written}/{len(todo)}  ${spent:.2f}")

    print(f"done: {written} generated, {len(failed)} failed, ${spent:.2f} spent")
    for stem, section, err in failed:
        print(f"  FAILED {stem} {section[-60:]}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
