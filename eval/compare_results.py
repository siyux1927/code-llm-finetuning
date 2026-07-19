"""
Compare baseline vs fine-tuned generations — pass@1 delta, significance, flipped
problems, and side-by-side samples for blog material.

Statistical significance uses SE_diff = √(SE_baseline² + SE_finetuned²) with a
1.96·SE_diff (95%) half-width — see docs/grilling_m1_m2.md Q1 for the derivation.

Usage:
    python eval/compare_results.py \\
        --baseline  data/processed/baseline_generations.jsonl \\
        --finetuned data/processed/finetuned_generations.jsonl \\
        --eval-set  data/processed/eval_114.jsonl
"""

import argparse
import json
import math
from pathlib import Path

from scoring import score_completions
from score import load_eval_set, join_generations_with_eval


def score_one(gen_path: Path, eval_records: dict):
    records = join_generations_with_eval(gen_path, eval_records)
    result = score_completions(records)
    completions = {r["task_id"]: r["completion"] for r in records}
    return result, completions


def print_pass_at_1_comparison(base_result: dict, tuned_result: dict) -> None:
    p_b = base_result["pass_at_1"]
    p_t = tuned_result["pass_at_1"]
    n = base_result["num_problems"]

    se_b = math.sqrt(p_b * (1 - p_b) / n) if n else 0.0
    se_t = math.sqrt(p_t * (1 - p_t) / n) if n else 0.0
    delta = p_t - p_b
    se_diff = math.sqrt(se_b**2 + se_t**2)
    delta_ci_half = 1.96 * se_diff
    significant = abs(delta) > delta_ci_half

    print("=" * 60)
    print("Pass@1 comparison")
    print("=" * 60)
    print(f"Baseline:    {p_b*100:6.2f}%   (SE ±{se_b*100:.2f}pp, N={n})")
    print(f"Fine-tuned:  {p_t*100:6.2f}%   (SE ±{se_t*100:.2f}pp, N={n})")
    print(f"Delta:      {delta*100:+6.2f}pp  (95% CI ±{delta_ci_half*100:.2f}pp)")
    print(f"Significant at 95%? {'YES' if significant else 'NO — within noise'}")
    print()


def print_flipped_samples(
    label: str,
    task_ids: list[str],
    base_completions: dict,
    tuned_completions: dict,
    n_samples: int,
) -> None:
    print(f"--- Sample of {min(n_samples, len(task_ids))} {label!r} ---")
    for tid in task_ids[:n_samples]:
        print(f"\n[{tid}]")
        print("  Baseline completion (first 200 chars):")
        for line in base_completions[tid][:200].split("\n"):
            print(f"    {line}")
        print("  Fine-tuned completion (first 200 chars):")
        for line in tuned_completions[tid][:200].split("\n"):
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path,
                        default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of flipped-problem side-by-sides to print per direction.")
    args = parser.parse_args()

    eval_records = load_eval_set(args.eval_set)
    base_result, base_completions = score_one(args.baseline, eval_records)
    tuned_result, tuned_completions = score_one(args.finetuned, eval_records)

    print_pass_at_1_comparison(base_result, tuned_result)

    base_passed = {r["task_id"]: r["passed"] for r in base_result["results"]}
    tuned_passed = {r["task_id"]: r["passed"] for r in tuned_result["results"]}

    flipped_up = [tid for tid in base_passed
                  if not base_passed[tid] and tuned_passed[tid]]
    flipped_down = [tid for tid in base_passed
                    if base_passed[tid] and not tuned_passed[tid]]

    print("=" * 60)
    print("Flipped problems")
    print("=" * 60)
    print(f"Base FAIL → Tuned PASS: {len(flipped_up)} problems")
    print(f"Base PASS → Tuned FAIL: {len(flipped_down)} problems (regressions)")
    print()

    if flipped_up:
        print_flipped_samples(
            "fail → pass",
            flipped_up,
            base_completions,
            tuned_completions,
            args.n_samples,
        )
    if flipped_down:
        print()
        print_flipped_samples(
            "pass → fail (regression)",
            flipped_down,
            base_completions,
            tuned_completions,
            args.n_samples,
        )


if __name__ == "__main__":
    main()
