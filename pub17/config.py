"""Settings shared by ingest, retrieval, generation, and the eval harness.

Everything is env-overridable so the eval can sweep configurations without
editing source.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

# The pgvector container from docker-compose.yml. For a native server with
# pgvector installed, point PUB17_DSN at it instead.
DSN = os.environ.get("PUB17_DSN", "postgresql://postgres:pub17@localhost:5434/pub17")

EMBED_MODEL = os.environ.get("PUB17_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = 768

# bge is trained asymmetrically: queries take this prefix, documents do not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

GEN_MODEL = os.environ.get("PUB17_GEN_MODEL", "claude-opus-5")
# These are short factual extractions over supplied context, not open reasoning,
# so the low effort level is enough and keeps a 60-question eval run cheap.
GEN_EFFORT = os.environ.get("PUB17_GEN_EFFORT", "low")

TOP_K = int(os.environ.get("PUB17_TOP_K", "5"))

# Cross-encoder reranking: over-fetch RERANK_CANDIDATES dense results, rescore
# each (question, chunk) pair jointly, keep TOP_K. See pub17/rerank.py.
RERANK = os.environ.get("PUB17_RERANK", "1") == "1"
RERANK_MODEL = os.environ.get("PUB17_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.environ.get("PUB17_RERANK_CANDIDATES", "20"))

# naive | section | section-heading -- see pub17/chunking.py. Defaults are the
# best measured pipeline; PUB17_CHUNKER=naive PUB17_RERANK=0 is the baseline.
CHUNKER = os.environ.get("PUB17_CHUNKER", "section")

# Upper bound on chunk size for every strategy. OVERLAP applies to naive only;
# section chunks start at headings and don't need it.
WORDS_PER_CHUNK = int(os.environ.get("PUB17_WORDS_PER_CHUNK", "350"))
OVERLAP = int(os.environ.get("PUB17_OVERLAP", "50"))

# Section chunkers can fold sections shorter than this into the next one.
# Measured and rejected: at 40 words it cost a question at recall@1 and @5 and
# gained nothing (results/runs.csv, section-fold40 vs section), so it is off.
MIN_SECTION_WORDS = int(os.environ.get("PUB17_MIN_SECTION_WORDS", "0"))

# Named so the eval CSV records which pipeline produced a row.
CONFIG_NAME = os.environ.get(
    "PUB17_CONFIG",
    ("baseline-naive-dense" if CHUNKER == "naive" else f"{CHUNKER}-dense")
    + ("-rerank" if RERANK else ""),
)

PUBLICATIONS = {
    "p17": "Publication 17 (Your Federal Income Tax)",
    "p501": "Publication 501 (Dependents, Standard Deduction, and Filing Information)",
    "p502": "Publication 502 (Medical and Dental Expenses)",
    "i1040gi": "Form 1040 Instructions",
    "f1040s1": "Schedule 1 (Additional Income and Adjustments to Income)",
    "f1040s1a": "Schedule 1-A (Additional Deductions)",
    "f1040s2": "Schedule 2 (Additional Taxes)",
    "f1040s3": "Schedule 3 (Additional Credits and Payments)",
}

TAX_YEAR = 2025
