import json
from pathlib import Path

from eval import scoring

EVAL_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "eval_50.jsonl"


def _load_eval_records_with_canonical_completions():
    records = []
    with EVAL_SET_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "completion": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
            )
    return records


def test_canonical_solutions_all_pass():
    # The official HumanEval answers should score 100% against their own
    # tests. If this ever fails, the harness itself is broken, not a model.
    records = _load_eval_records_with_canonical_completions()
    result = scoring.score_completions(records)
    assert result["pass_at_1"] == 1.0
    assert result["num_passed"] == result["num_problems"]


def test_wrong_completion_fails():
    record = {
        "task_id": "fake/0",
        "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "completion": "    return a - b\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "entry_point": "add",
    }
    result = scoring.score_completions([record])
    assert result["pass_at_1"] == 0.0
    assert result["results"][0]["passed"] is False


def test_infinite_loop_times_out(monkeypatch):
    monkeypatch.setattr(scoring, "TIMEOUT_SECONDS", 1.0)
    record = {
        "task_id": "fake/1",
        "prompt": "def loop_forever():\n",
        "completion": "    while True:\n        pass\n",
        "test": "def check(candidate):\n    candidate()\n",
        "entry_point": "loop_forever",
    }
    result = scoring.score_completions([record])
    assert result["pass_at_1"] == 0.0
    assert "timeout" in result["results"][0]["error"]
