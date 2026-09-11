# pub17

Retrieval-augmented question answering over ~2,000 pages of IRS tax year 2025
publications, built to measure what each retrieval technique is actually worth
rather than to assemble a stack.

**Status:** Weekend 1 and the evaluation set are done. Two of the four
Weekend 2 changes are measured: section-aware chunking and cross-encoder
reranking. The pipeline runs end to end against pgvector in Docker.
Contextual enrichment and Postgres full-text hybrid search are next.

## Results

| Configuration | recall@1 | recall@3 | recall@5 | MRR@5 | loose recall@5 | answer acc. |
|---|---|---|---|---|---|---|
| Naive 350-word windows, dense only (baseline) | 51.7% | 78.3% | 83.3% | 0.642 | 86.7% | pending |
| Section-aware chunking | 63.3% | 86.7% | 91.7% | 0.753 | 95.0% | pending |
| **+ cross-encoder reranking**, 20 → 5 | **90.0%** | **98.3%** | **98.3%** | **0.939** | 98.3% | pending |
| *rejected:* + heading path in the embedded text | 55.0% | 85.0% | 93.3% | 0.696 | 96.7% | — |
| *rejected:* + fold sections under 40 words | 61.7% | 83.3% | 90.0% | 0.734 | 93.3% | — |
| *ablation:* naive chunks + reranking | 90.0% | 93.3% | 96.7% | 0.924 | 98.3% | — |

n=60 questions, `bge-base-en-v1.5` retrieval, `bge-reranker-base` reranking,
top-k=5, 350-word chunk ceiling. Measured
by `eval/baseline_numpy.py` (exhaustive search, no services) and confirmed
through pgvector by `eval/run_eval.py`; every run is appended to
`results/runs.csv`.

**One question is 1.7 points.** With n=60, only naive → section clears that
margin comfortably: +11.6 points at recall@1 and +8.4 at recall@5, seven and
five questions. The two rejected variants sit within a question or two of the
row above them, so they are recorded as measured, not as findings.

Section chunking fixed 8 of the baseline's 10 misses (the married-filing-
separately `$5` threshold, both premium tax credit examples, the 401(k) and
adoption limits, the overtime cap, the car-loan income limit, the MFS capital
loss limit)
and introduced 3 new ones (q012, q024, q029).

Putting the heading path into the embedded text bought one question at
recall@5 and cost five at recall@1. It was not adopted.

**Reranking did most of the work.** Over-fetching 20 dense candidates and
rescoring them with a cross-encoder took recall@1 from 63.3% to 90.0% and
recall@5 from 91.7% to 98.3%. That is the ceiling: dense recall@20 is also
98.3%, so the reranker ordered the pool as well as the pool allowed. The
ablation matters more than the headline, though. Naive chunks with reranking
reach 90.0% at recall@1 and 96.7% at recall@5, so once the reranker is in
place, section chunking is worth one question at recall@5, which is within
noise. Most of what section chunking gained on its own, the reranker recovers
anyway. Section chunking stays because it costs nothing at query time and
never measured worse, but the claim it supports is small.

**The one remaining miss is out of reach, not misranked.** q028 asks how much
of the child tax credit is claimable as the additional child tax credit. Its
answer chunk sits at dense rank 49, outside the 20-candidate pool, because it
says "ACTC" where the question spells the name out. Widening the pool to 50
would fix it, and it wasn't done: that would be tuning the pipeline to one
question it is scored on. Lexical hybrid search is the change that should be
tested against it.

**Through pgvector the numbers hold.** Section chunks score identically
through the HNSW index and through exhaustive search, both dense-only and
reranked, and so do naive chunks dense-only. The one difference was naive
chunks with reranking: 88.3% at recall@1 in one run against 90.0% exhaustive.
That was q013, where that run's 20-candidate pool from the index differed from
the exhaustive one. pgvector builds its HNSW graph randomly at each ingest, and
the next build matched exhaustive search exactly, at both `ef_search` 40 and
200. So an approximate index can move a question when you over-fetch 20
candidates, and it is a one-question effect here.

**Latency** through Postgres, per query on the development GPU: dense
retrieval 9 ms p50 / 10 ms p95, dense plus reranking 166 ms / 174 ms. The
reranker runs in unoptimized fp32 and is almost all of it.

## Why the numbers are trustworthy

The evaluation set is the point of this project, so it is built to be
falsifiable rather than plausible.

**Gold is anchored on pages and verbatim spans, not chunk ids.** Chunk ids do
not survive a re-chunk. Anchoring gold to them would mean Weekend 2's new
chunker silently invalidates the baseline it is supposed to be measured
against — the one comparison the project exists to make.

**A hit means the chunk contains the answer.** A retrieved chunk counts only if
it holds the verbatim gold span, or sits on a gold page and contains every
answer key. The first version of the harness accepted page overlap alone, and
that overstated every configuration, the baseline included (86.7% loose against
83.3% strict). Section chunking makes the gap matter: the 1040 instructions
bookmark every form line, so a pure split-on-outline yields hundreds of
heading-sized fragments like `Line 1a` that sit on the right page and hold
nothing. The loose number is still reported, and the Weekend 1 rows under the
old rule are kept in `results/runs_v1_loose.csv`.

**Every question is checked against the PDFs.** `eval/verify_questions.py`
confirms that each gold page exists, that the answer-bearing span appears
verbatim on it, and — the strict check — that *every* gold page independently
contains the answer. It exits nonzero on any failure, so CI can gate on it.

**Gold lists every page that genuinely answers.** The IRS corpus duplicates
heavily: Pub 17 chapter 10 restates Pub 501's standard deduction charts, and
the 1040 instructions restate both. The first draft of the eval listed one
source page per question and scored 75.0% recall@5. Scanning the whole corpus
for answer-bearing pages and reading each candidate raised the gold set from 85
pages to 188 and the measured baseline to 86.7% — the retrieval had not
changed, the ruler had. Coincidental number matches were reviewed and rejected
(EIC lookup tables, the $10,000 SALT floor against the $10,000 car-loan cap,
student-loan MAGI limits that share the car-loan thresholds).

**Answers are graded on exact keys, not by a model.** These questions have
dollar amounts, rates, and yes/no answers. An LLM judge would be more forgiving
of paraphrase and would also make the metric depend on a second model's mood.

**The corpus postdates model training.** Tax year 2025 includes the Schedule
1-A deductions, a $2,200 child tax credit, and a $40,000 SALT cap. A model
answering from memory gets these confidently wrong, so the eval measures
retrieval rather than recall of pretraining.

## The eval set

60 questions across 15 topics — 30 direct lookups, 17 rule questions, and 13
worked examples taken from the publications' own examples with their stated
answers.

```
standard_deduction 9   new_deductions_2025 11   medical 8   filing_requirement 6
dependents 5   mileage 4   adoption 3   child_tax_credit 3   retirement 3
salt 2   filing_status 2   capital_gains 1   deadlines 1   digital_assets 1
itemized 1
```

## Corpus

Pub 17, Pub 501, Pub 502, the Form 1040 instructions, and Schedules 1, 1-A, 2,
and 3 — 338,000 words, 1,130 chunks at the baseline chunk size.

## Setup

```bash
pip install -r requirements.txt
./scripts/download_corpus.sh
```

Retrieval scoring needs nothing else:

```bash
python3 eval/verify_questions.py      # gate: is the eval set still grounded?
python3 eval/baseline_numpy.py --csv  # score the default pipeline, no services required
PUB17_CHUNKER=naive PUB17_RERANK=0 python3 eval/baseline_numpy.py   # the Weekend 1 baseline
```

For the full pipeline (pgvector storage, generation, latency and cost metrics):

```bash
docker compose up -d --wait     # Postgres 18 + pgvector on localhost:5434
python3 scripts/ingest.py
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/ask.py "what is the standard deduction for a single filer?"
python3 eval/run_eval.py --retrieval-only        # free
python3 eval/run_eval.py                         # full run, makes model calls
```

To use a native Postgres instead, install pgvector, run
`./scripts/setup_db.sh`, and set `PUB17_DSN`.

## Layout

```
docker-compose.yml         Postgres 18 + pgvector on localhost:5434
pub17/config.py            settings, all env-overridable
pub17/chunking.py          naive and section-aware chunkers (PUB17_CHUNKER)
pub17/rerank.py            cross-encoder reranking (PUB17_RERANK)
pub17/store.py             pgvector schema, retrieve() = dense search + rerank
pub17/generate.py          cited answer generation
scripts/ingest.py          parse -> chunk -> embed -> idempotent upsert
scripts/ask.py             ask a question, get a cited answer
eval/questions.jsonl       60 verified questions
eval/verify_questions.py   grounding check, exits nonzero on failure
eval/run_eval.py           scores the pgvector pipeline, appends to runs.csv
eval/baseline_numpy.py     scores retrieval with no services
results/runs.csv           every configuration measured under the strict hit rule
results/runs_v1_loose.csv  Weekend 1 rows under the retired page-overlap rule
```

## What the baseline gets wrong

Eight of sixty questions miss at k=5, and the failures are consistent rather
than random:

- **Tables shredded mid-row.** Fixed 350-word windows cut the standard deduction
  and filing threshold charts apart, so the row and its header land in different
  chunks. This is what breaks the married-filing-separately `$5` threshold.
- **Context stripped from worked examples.** The Pub 502 premium tax credit
  examples retrieve their neighbours instead of themselves, because the chunk
  holding the arithmetic never says which deduction it belongs to.
- **"What's New" summaries lose their subject.** The 401(k) limits and the
  additional child tax credit amount sit in dense bulleted summaries whose
  topic is stated only in a heading the chunker discarded.

Each maps onto a planned Weekend 2 change — section-aware chunking that keeps
tables whole, and contextual enrichment that prepends the parent section — which
is the point of measuring before changing anything.

## Known weak

- One embedding model and one chunk size; neither has been swept.
- n=60, and every configuration is chosen on the same 60 questions it is scored
  on. There is no held-out split, so small wins are indistinguishable from
  fitting the eval.
- No lexical retrieval yet, so an abbreviation mismatch like "ACTC" against
  "additional child tax credit" (q028) never reaches the reranker.
- The HNSW graph is rebuilt randomly on every ingest, so with a 20-candidate
  pool a rebuild can move a question (q013 once). Runs aren't pinned to one
  index build.
- Answer accuracy is unmeasured pending the generation run.
- Exact-key grading cannot tell a correct answer from one that states the right
  number for the wrong reason.
- `page_start`/`page_end` come from the words in a chunk, so a chunk spanning a
  page boundary is credited with both pages.
