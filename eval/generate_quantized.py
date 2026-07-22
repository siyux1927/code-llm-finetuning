"""
M5: Generate HumanEval completions from a quantized (or unquantized-merged)
model checkpoint.

Loads directly from a local model directory — works for both the fp16
merged reference (`models/merged_fp16`, produced by
`train/quantize_gptq.py`) and the GPTQ-4bit checkpoint
(`models/gptq_4bit`), since transformers reads the quantization config from
the checkpoint's own `config.json`; no explicit BitsAndBytesConfig/
GPTQConfig needed at load time here.

Reuses `generate_completion` (and its locked-in greedy /
max_new_tokens=512 / STOP_SEQUENCES) from `generate_completions.py` so M5
numbers attribute cleanly to the quantization method, not to a
decoding-config drift. See docs/grilling_m5_pre.md Q6.

Runs on Colab T4 (CUDA required).

Usage:
    python eval/generate_quantized.py --model-path models/merged_fp16 \\
        --output data/processed/quant_fp16_generations.jsonl

    python eval/generate_quantized.py --model-path models/gptq_4bit \\
        --output data/processed/quant_gptq4bit_generations.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_completions import generate_completion


def load_model_and_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    # dtype=fp16 must be explicit: without it, from_pretrained upcasts the
    # fp16 merged checkpoint to fp32 (Llama-2-7B: 13.5GB → 27GB), which OOMs
    # the T4. The GPTQ checkpoint is unaffected (4-bit is config-locked), but
    # this same call loads the fp16 merge too. device_map={"": 0} pins to the
    # single GPU rather than letting accelerate's "auto" silently offload
    # layers to CPU/disk — same failure class as issue #8's quantize step.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float16, device_map={"": 0},
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local model dir — fp16 merged or GPTQ checkpoint.")
    parser.add_argument("--eval-set", type=Path,
                        default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.model_path / "config.json").exists():
        raise SystemExit(
            f"[m5] No model found at {args.model_path} (no config.json inside). "
            f"Did train/quantize_gptq.py run and produce this directory?"
        )

    print(f"[m5-gen] model:    {args.model_path}")
    print(f"[m5-gen] eval set: {args.eval_set}")
    print(f"[m5-gen] output:   {args.output}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_set.open() as f_in, args.output.open("w") as f_out:
        for line in f_in:
            row = json.loads(line)
            completion = generate_completion(model, tokenizer, row["prompt"])
            f_out.write(json.dumps({"task_id": row["task_id"], "completion": completion}) + "\n")
            print(f"{row['task_id']}: generated {len(completion)} chars")

    print(f"[m5-gen] wrote {args.output}")


if __name__ == "__main__":
    main()
