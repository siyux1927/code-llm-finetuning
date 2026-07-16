"""
M2: Baseline generation — run the un-fine-tuned, 4-bit-quantized Llama-2-7B
over the eval set and save its completions.

This needs a CUDA GPU (bitsandbytes 4-bit quantization is not supported on
Mac/MPS) and gated access to meta-llama/Llama-2-7b-hf, so it's meant to run
on Colab, not locally. It only produces `baseline_generations.jsonl` — the
actual pass@1 scoring is done separately by `eval/scoring.py`, so the same
scoring code can be reused unchanged for M4's fine-tuned generations.

The base model is loaded in 4-bit (nf4) to match the quantization M3's
QLoRA fine-tuning will use — see the "M2 baseline 应该用哪种精度" decision
in CLAUDE.md / the linked GitHub issue. This isolates "does fine-tuning
help" from "does quantization hurt", which would otherwise be conflated if
baseline ran at fp16 and the fine-tuned model ran at 4-bit.

Usage (on Colab, after `huggingface-cli login` with a token that has
accepted the Llama 2 license):
    pip install -r requirements-colab.txt
    python eval/generate_baseline.py \
        --eval-set data/processed/eval_114.jsonl \
        --output data/processed/baseline_generations.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MAX_NEW_TOKENS = 512

# HumanEval completions should stop once the function body ends. Without a
# stop rule the model keeps generating past the function (more examples,
# comments, a new function...), which breaks the prompt+completion
# concatenation the scoring harness relies on.
STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@"]


def truncate_at_stop_sequence(text: str) -> str:
    cut_at = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at]


def load_model_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: deterministic, standard for pass@1
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return truncate_at_stop_sequence(raw_completion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/baseline_generations.jsonl"))
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    with args.eval_set.open() as f_in, args.output.open("w") as f_out:
        for line in f_in:
            row = json.loads(line)
            completion = generate_completion(model, tokenizer, row["prompt"])
            f_out.write(json.dumps({"task_id": row["task_id"], "completion": completion}) + "\n")
            print(f"{row['task_id']}: generated {len(completion)} chars")

    print(f"Wrote generations to {args.output}")


if __name__ == "__main__":
    main()
