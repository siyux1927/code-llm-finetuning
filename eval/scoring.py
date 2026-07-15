"""
Pass@1 scoring harness for HumanEval-style completions.

This is the seam the whole eval pipeline is built around: given a set of
(prompt, generated completion, test) records, it executes each candidate
program against its test suite and reports pass/fail. It has no dependency
on any model or GPU, so it can be developed and unit-tested locally, then
reused unchanged by both M2 (baseline) and M4 (fine-tuned) scoring runs.

Usage:
    from eval.scoring import score_completions
    result = score_completions(records)  # records: task_id, prompt, completion, test, entry_point
    print(result["pass_at_1"])
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict


class Record(TypedDict):
    task_id: str
    prompt: str
    completion: str
    test: str
    entry_point: str


TIMEOUT_SECONDS = 5.0


def _check_correctness(record: Record) -> tuple[bool, str]:
    program = (
        record["prompt"]
        + record["completion"]
        + "\n"
        + record["test"]
        + f"\ncheck({record['entry_point']})\n"
    )

    # Run in a subprocess (not exec() in-process) so a hung or crashing
    # candidate program can't take down the scoring run itself, and so a
    # timeout can be enforced from the outside.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        program_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, program_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip().splitlines()[-1] if result.stderr else "non-zero exit"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TIMEOUT_SECONDS}s"
    finally:
        Path(program_path).unlink(missing_ok=True)


def score_completions(records: list[Record]) -> dict:
    per_problem = []
    for record in records:
        passed, error = _check_correctness(record)
        per_problem.append({"task_id": record["task_id"], "passed": passed, "error": error})

    pass_count = sum(r["passed"] for r in per_problem)
    return {
        "pass_at_1": pass_count / len(records) if records else 0.0,
        "num_problems": len(records),
        "num_passed": pass_count,
        "results": per_problem,
    }
