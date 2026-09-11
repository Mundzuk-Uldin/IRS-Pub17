"""Re-grade saved answers with the current grader. No model calls, no cost.

Every generation run writes each answer to results/<config>-<timestamp>.jsonl.
When the grader changes -- a key fixed, a rule tightened -- this recomputes
answer accuracy from those saved answers instead of paying to generate them
again. results/runs.csv is left as recorded; the rescored numbers are printed
next to the recorded ones.

Usage:  python3 eval/rescore.py results/section-dense-rerank-*.jsonl ...
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import RUNS_CSV, answer_is_correct, load_questions


def main(paths):
    questions = {q["id"]: q for q in load_questions()}
    recorded = {}
    if RUNS_CSV.exists():
        for row in csv.DictReader(RUNS_CSV.open()):
            stamp = row["run_at"].replace(":", "")
            recorded[f"{row['config']}-{stamp}"] = row["answer_accuracy"]

    for path in paths:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        graded = [r for r in rows if "answer" in r]
        if not graded:
            print(f"{path}: retrieval-only run, nothing to grade")
            continue
        now = [answer_is_correct(r["answer"], questions[r["id"]]) for r in graded]
        flipped = [r["id"] for r, ok in zip(graded, now) if ok != r["correct"]]
        was = recorded.get(Path(path).stem, "?")
        print(f"{Path(path).stem}: recorded {was}%  ->  rescored "
              f"{100 * sum(now) / len(now):.1f}%   changed: {flipped or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
