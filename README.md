# pub17

Retrieval-augmented question answering over ~2,000 pages of IRS tax year 2025
publications, built to measure what each retrieval technique is actually worth
rather than to assemble a stack.

**Status:** Weekend 1 and the evaluation set are done. Of the four Weekend 2
changes, section-aware chunking is measured. Reranking is next. Contextual
enrichment is waiting on an Anthropic API key, and Postgres full-text hybrid
search is waiting on pgvector.

## Results

| Configuration | recall@1 | recall@3 | recall@5 | MRR@5 | loose recall@5 | answer acc. |
|---|---|---|---|---|---|---|
| Naive 350-word windows, dense only (baseline) | 51.7% | 78.3% | 83.3% | 0.642 | 86.7% | pending |
| **Section-aware chunking** | **63.3%** | **86.7%** | **91.7%** | **0.753** | 95.0% | pending |
| *rejected:* + heading path in the embedded text | 55.0% | 85.0% | 93.3% | 0.696 | 96.7% | — |
| *rejected:* + fold sections under 40 words | 61.7% | 83.3% | 90.0% | 0.734 | 93.3% | — |

n=60 questions, `bge-base-en-v1.5`, top-k=5, 350-word chunk ceiling. Measured
by `eval/baseline_numpy.py`; every run is appended to `results/runs.csv`.

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
recall@5 and cost five at recall@1. It was not adopted. The reranker reorders
the top candidates anyway, so rank-1 precision is the better signal to keep.

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
python3 eval/baseline_numpy.py --csv  # score retrieval, no services required
```

For the full pipeline (pgvector storage, generation, latency and cost metrics):

```bash
sudo apt-get install -y postgresql-18-pgvector   # Ubuntu ships pgvector for PG18 only
./scripts/setup_db.sh
python3 scripts/ingest.py
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/ask.py "what is the standard deduction for a single filer?"
python3 eval/run_eval.py --retrieval-only        # free
python3 eval/run_eval.py                         # full run, makes model calls
```

Docker Desktop's WSL integration was off on the development machine, so this
targets a native Postgres rather than a container.

## Layout

```
pub17/config.py            settings, all env-overridable
pub17/chunking.py          naive and section-aware chunkers (PUB17_CHUNKER)
pub17/store.py             pgvector schema, dense retrieval
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
- Retrieval is dense-only, so exact-phrase questions have no lexical fallback.
- Answer accuracy is unmeasured pending the generation run.
- Exact-key grading cannot tell a correct answer from one that states the right
  number for the wrong reason.
- `page_start`/`page_end` come from the words in a chunk, so a chunk spanning a
  page boundary is credited with both pages.
