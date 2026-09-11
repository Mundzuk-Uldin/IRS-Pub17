"""Turn a PDF into chunks. The strategy is chosen by `config.CHUNKER`.

naive            Fixed word windows with overlap that run straight through
                 headings and tables. The Weekend 1 baseline.
section          Split on the PDF's own outline (the IRS publications ship
                 with complete bookmarks), pack whole text blocks, and never
                 let a chunk cross a heading. Tables stay intact unless the
                 section they sit in is itself longer than a chunk.
section-heading  As `section`, with the heading path prepended to the text
                 that gets embedded -- e.g. "Pub 501 > Standard Deduction >
                 Standard Deduction Chart". Deterministic, so it is measured
                 separately from the LLM-written contextual enrichment.
section-context  As `section`, with the generated context of the chunk's
                 parent section (scripts/summarize_sections.py) attached.
                 Weekend 2 change 2, contextual enrichment.

The 1040 instructions bookmark every form line ("Line 1a", "Line 1b", ...), so
a pure split-on-outline leaves heading-only fragments with nothing to answer
from. Sections shorter than MIN_SECTION_WORDS are folded into the section that
follows them.

Every strategy returns the same shape, so ingest and eval don't care which ran:
    [{"text", "page_start", "page_end", "section", "context"}, ...]

`context` is kept apart from `text` on purpose. It is prepended for embedding
and reranking (see `embed_text`), but a strict eval hit and the generation
prompt still see only the chunk's own text. Otherwise a context that happens
to mention an answer figure would score as retrieving it.
"""
import json
import re

import pymupdf as fitz

from . import config

# Outline titles are compared on their first 40 normalized characters: long
# enough to be unambiguous, short enough to survive a heading that wraps.
TITLE_PREFIX = 40


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def chunk_naive(path):
    words = []
    with fitz.open(path) as doc:
        for pageno, page in enumerate(doc, start=1):
            for word in page.get_text("text").split():
                words.append((word, pageno))

    step = config.WORDS_PER_CHUNK - config.OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + config.WORDS_PER_CHUNK]
        if len(window) < 20:
            break
        chunks.append({
            "text": " ".join(w for w, _ in window),
            "page_start": window[0][1],
            "page_end": window[-1][1],
            "section": "",
            "context": "",
        })
    return chunks


def _page_blocks(page):
    """Text blocks in content-stream order, which follows the columns on these PDFs."""
    return [
        " ".join(b[4].split())
        for b in page.get_text("blocks")
        if b[6] == 0 and b[4].strip()
    ]


def _sections(doc, root):
    """Walk the document once and return [(heading_path, [(page, block_text), ...])].

    A heading takes effect at the text block that matches its outline title. An
    outline entry with no matching block (usually a Part or Chapter banner drawn
    as artwork) takes effect at the top of its page instead; the child heading
    that follows it is almost always matched.
    """
    outline = {}
    for level, title, pageno in doc.get_toc():
        if pageno >= 1 and _norm(title):
            outline.setdefault(pageno, []).append((level, title.strip()))

    stack = []
    sections = []
    current = []

    def enter(level, title):
        if current:
            sections.append((" > ".join([root] + [t for _, t in stack]), list(current)))
            current.clear()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    for pageno, page in enumerate(doc, start=1):
        blocks = _page_blocks(page)
        normed = [_norm(b) for b in blocks]
        pending = list(outline.get(pageno, []))

        # Entries no block on this page matches -- enter them before reading it.
        for entry in list(pending):
            key = _norm(entry[1])[:TITLE_PREFIX]
            if not any(n.startswith(key) for n in normed):
                enter(*entry)
                pending.remove(entry)

        for text, n in zip(blocks, normed):
            for entry in pending:
                if n.startswith(_norm(entry[1])[:TITLE_PREFIX]):
                    enter(*entry)
                    pending.remove(entry)
                    break
            current.append((pageno, text))

    if current:
        sections.append((" > ".join([root] + [t for _, t in stack]), current))
    return sections


def _pack(path_label, blocks, limit):
    """Pack whole blocks into chunks of at most `limit` words.

    A block is only ever split when it alone is longer than a chunk.
    """
    chunks = []
    buf = []

    def emit(items):
        if items:
            chunks.append({
                "text": " ".join(t for _, t in items),
                "page_start": items[0][0],
                "page_end": items[-1][0],
                "section": path_label,
                "context": "",
            })

    words = 0
    for pageno, text in blocks:
        n = len(text.split())
        if n > limit:
            emit(buf)
            buf, words = [], 0
            tokens = text.split()
            for i in range(0, len(tokens), limit):
                emit([(pageno, " ".join(tokens[i:i + limit]))])
        elif buf and words + n > limit:
            emit(buf)
            buf, words = [(pageno, text)], n
        else:
            buf.append((pageno, text))
            words += n
    emit(buf)
    return chunks


def embed_text(chunk):
    """What gets embedded and reranked: the chunk, behind its section context if any."""
    context = chunk.get("context") or ""
    return f"{context}\n\n{chunk['text']}" if context else chunk["text"]


def _section_contexts(stem):
    """{heading_path: context} for one publication. Later lines win, so a re-run
    that regenerates a section supersedes the older context."""
    path = config.ROOT / "data" / "section_summaries.jsonl"
    if not path.exists():
        raise RuntimeError("no section contexts yet -- run scripts/summarize_sections.py")
    contexts = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["stem"] == stem:
                contexts[r["section"]] = r["context"]
    return contexts


def chunk_sections(path, label, with_heading, contexts=None):
    with fitz.open(path) as doc:
        sections = _sections(doc, label)

    chunks = []
    carry = []
    for path_label, blocks in sections:
        blocks = carry + blocks
        if sum(len(t.split()) for _, t in blocks) < config.MIN_SECTION_WORDS:
            carry = blocks
            continue
        carry = []
        chunks.extend(_pack(path_label, blocks, config.WORDS_PER_CHUNK))
    if carry:
        chunks.extend(_pack(sections[-1][0], carry, config.WORDS_PER_CHUNK))

    if with_heading:
        for c in chunks:
            c["text"] = f"{c['section']}\n\n{c['text']}"

    if contexts is not None:
        missing = sorted({c["section"] for c in chunks if c["section"] not in contexts})
        if missing:
            raise RuntimeError(
                f"{len(missing)} sections of {label} have no generated context "
                "-- run scripts/summarize_sections.py to fill them in"
            )
        for c in chunks:
            c["context"] = contexts[c["section"]]
    return chunks


def chunk_pdf(stem):
    path = config.RAW_DIR / f"{stem}.pdf"
    label = config.PUBLICATIONS[stem]
    if config.CHUNKER == "naive":
        return chunk_naive(path)
    if config.CHUNKER == "section":
        return chunk_sections(path, label, with_heading=False)
    if config.CHUNKER == "section-heading":
        return chunk_sections(path, label, with_heading=True)
    if config.CHUNKER == "section-context":
        return chunk_sections(path, label, with_heading=False, contexts=_section_contexts(stem))
    raise ValueError(f"unknown PUB17_CHUNKER {config.CHUNKER!r}")
