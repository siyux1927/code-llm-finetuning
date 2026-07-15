"""
M1: Data Preparation

Loads the 164-problem HumanEval benchmark, splits it into two disjoint
50-problem sets (train / eval), and formats the train set for completion-style
fine-tuning (README Decision 4).

Usage:
    micromamba run -n myenv python data/scripts/prepare_data.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
TRAIN_SIZE = 50
EVAL_SIZE = 50

OUT_DIR = Path(__file__).resolve().parent.parent / "processed"


def main() -> None:
    dataset = load_dataset("openai/openai_humaneval", split="test")
    examples = list(dataset)

    rng = random.Random(SEED)
    rng.shuffle(examples)

    train_examples = examples[:TRAIN_SIZE]
    eval_examples = examples[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE]

    assert len(train_examples) == TRAIN_SIZE
    assert len(eval_examples) == EVAL_SIZE
    train_ids = {ex["task_id"] for ex in train_examples}
    eval_ids = {ex["task_id"] for ex in eval_examples}
    assert train_ids.isdisjoint(eval_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train_50.jsonl"
    with train_path.open("w") as f:
        for ex in train_examples:
            record = {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "completion": ex["canonical_solution"],
            }
            f.write(json.dumps(record) + "\n")

    eval_path = OUT_DIR / "eval_50.jsonl"
    with eval_path.open("w") as f:
        for ex in eval_examples:
            record = {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "canonical_solution": ex["canonical_solution"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(train_examples)} train examples to {train_path}")
    print(f"Wrote {len(eval_examples)} eval examples to {eval_path}")


if __name__ == "__main__":
    main()
