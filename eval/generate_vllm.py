"""
M6: Generate HumanEval completions for the M5 GPTQ-4bit checkpoint via vLLM,
timing both a single-request call and a full-114 batch call.

This is the vLLM side of the M6 comparison — the HF-native side is just
`eval/generate_quantized.py` (already written for M5) with a `%%time` cell
wrapped around it in the notebook, no code change needed there. See
docs/grilling_m6_pre.md Q1/Q4.

Decoding params (`STOP_SEQUENCES`, `max_new_tokens=512`, greedy) are
imported from `generate_completions.py` so the vLLM numbers attribute
cleanly to the serving framework, not to a decoding-config drift. vLLM's
native `stop` truncates during generation (not after, like the HF side's
`truncate_at_stop_sequence`) — equivalent under greedy decoding, see
grilling_m6_pre.md Q4.

Runs on Colab T4 (CUDA required).

Usage:
    python eval/generate_vllm.py --model-path models/gptq_4bit \\
        --output data/processed/quant_gptq4bit_vllm_generations.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from generate_completions import MAX_NEW_TOKENS, STOP_SEQUENCES


def load_prompts(eval_set_path: Path) -> list[dict]:
    return [json.loads(line) for line in eval_set_path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local GPTQ checkpoint dir (e.g. models/gptq_4bit).")
    parser.add_argument("--eval-set", type=Path,
                        default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.model_path / "config.json").exists():
        raise SystemExit(
            f"[m6] No model found at {args.model_path} (no config.json inside). "
            f"Did train/quantize_gptq.py run and produce this directory?"
        )

    print(f"[m6-vllm] model:    {args.model_path}")
    print(f"[m6-vllm] eval set: {args.eval_set}")

    llm = LLM(model=str(args.model_path), quantization="gptq")
    sampling_params = SamplingParams(
        temperature=0,  # greedy: deterministic, matches M2/M4/M5
        max_tokens=MAX_NEW_TOKENS,
        stop=STOP_SEQUENCES,
    )

    rows = load_prompts(args.eval_set)

    # Single-request timing — batch size 1, comparable to "one API call alone".
    print("[m6-vllm] timing single request...")
    t0 = time.time()
    _ = llm.generate([rows[0]["prompt"]], sampling_params)
    single_request_seconds = time.time() - t0
    print(f"[m6-vllm] single-request latency: {single_request_seconds:.2f}s")

    # Batch-114 timing — all prompts submitted at once, exercises vLLM's
    # continuous batching. This also doubles as the compatibility check for
    # our GPTQ checkpoint format (grilling_m6_pre.md Q6) — if it got this
    # far, the format loaded fine.
    print(f"[m6-vllm] timing batch of {len(rows)} prompts...")
    t0 = time.time()
    outputs = llm.generate([r["prompt"] for r in rows], sampling_params)
    batch_seconds = time.time() - t0
    print(f"[m6-vllm] batch-{len(rows)} total: {batch_seconds:.2f}s "
          f"({batch_seconds / len(rows):.2f}s/problem average)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f_out:
        for row, output in zip(rows, outputs):
            completion = output.outputs[0].text
            f_out.write(json.dumps({"task_id": row["task_id"], "completion": completion}) + "\n")

    print(f"[m6-vllm] wrote {args.output}")
    print(f"[m6-vllm] summary: single-request={single_request_seconds:.2f}s, "
          f"batch-{len(rows)}={batch_seconds:.2f}s "
          f"({len(rows) / batch_seconds:.2f} problems/sec)")


if __name__ == "__main__":
    main()
