"""
CLI wrapper for `eval/scoring.py` — computes pass@1 on a generations.jsonl.

Loads the eval set to look up prompt/test/entry_point for each generated
completion, then delegates to `score_completions`. Reports pass@1 plus the
95% CI half-width so the reader immediately sees the uncertainty range.

Usage:
    python eval/score.py \\
        --generations data/processed/baseline_generations.jsonl \\
        --eval-set    data/processed/eval_114.jsonl

Optional `--per-problem-output` writes per-problem pass/fail + error jsonl.
"""

import argparse
import json
import math
from pathlib import Path

from eval.scoring import score_completions


def load_eval_set(eval_set_path: Path) -> dict:
    """Return {task_id: full_eval_record}."""
    return {
        json.loads(line)["task_id"]: json.loads(line)
        for line in eval_set_path.read_text().splitlines()
    }


def join_generations_with_eval(gen_path: Path, eval_records: dict) -> list[dict]:
    """Combine generations with prompt/test/entry_point into scoring records."""
    records = []
    for line in gen_path.read_text().splitlines():
        g = json.loads(line)
        e = eval_records[g["task_id"]]
        records.append({
            "task_id": g["task_id"],
            "prompt": e["prompt"],
            "completion": g["completion"],
            "test": e["test"],
            "entry_point": e["entry_point"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path,
                        default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--per-problem-output", type=Path, default=None,
                        help="If set, write per-problem pass/fail + error to this jsonl.")
    args = parser.parse_args()

    eval_records = load_eval_set(args.eval_set)
    records = join_generations_with_eval(args.generations, eval_records)
    result = score_completions(records)

    # 95% CI half-width for a binomial proportion
    p = result["pass_at_1"]
    n = result["num_problems"]
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    ci_half = 1.96 * se

    print(f"pass@1 = {p:.4f} ({result['num_passed']}/{n})")
    print(f"95% CI: ±{ci_half*100:.2f}pp  (SE {se*100:.2f}pp, N={n})")

    if args.per_problem_output:
        args.per_problem_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_problem_output.open("w") as f:
            for r in result["results"]:
                f.write(json.dumps(r) + "\n")
        print(f"per-problem results → {args.per_problem_output}")


if __name__ == "__main__":
    main()
