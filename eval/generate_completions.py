"""
Generate model completions for a HumanEval eval set, with or without a LoRA adapter.

- **Baseline (M2)**: no `--adapter` → un-fine-tuned 4-bit Llama-2-7B
- **Fine-tuned (M4)**: `--adapter models/lora_adapter/` → base + LoRA overlay

Generation params (greedy, `max_new_tokens=512`, STOP_SEQUENCES) are locked
identical across both modes so M4 vs M2 differences attribute cleanly to
fine-tuning, not to a decoding-config drift. See docs/grilling_m4_pre.md Q3.

The base model is loaded in 4-bit (nf4) to match M3's QLoRA quantization —
this isolates "does fine-tuning help" from "does quantization hurt".

Runs on Colab T4 (CUDA required; bitsandbytes doesn't work on Mac/MPS).

Usage (Colab, after `huggingface-cli login` with Llama-2 license accepted):

    # M2 baseline
    python eval/generate_completions.py

    # M4 fine-tuned
    python eval/generate_completions.py --adapter models/lora_adapter
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


def load_model_and_tokenizer(adapter_path: Path | None):
    """Load 4-bit Llama-2-7B + tokenizer. If adapter_path is given, overlay the
    LoRA adapter on top (M4 mode)."""
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

    if adapter_path is not None:
        # Import lazily so baseline runs don't require peft to be installed
        # (though in practice requirements-colab.txt installs it either way).
        from peft import PeftModel
        print(f"[m4] loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
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


def default_output_path(adapter_path: Path | None) -> Path:
    if adapter_path is None:
        return Path("data/processed/baseline_generations.jsonl")
    return Path("data/processed/finetuned_generations.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path,
                        default=Path("data/processed/eval_114.jsonl"))
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Optional LoRA adapter path. If given, model = base + adapter (M4).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output jsonl path. Defaults to baseline_ or finetuned_generations.jsonl.")
    args = parser.parse_args()

    # Defensive check — see grilling_m4_pre.md Q8
    if args.adapter is not None:
        if not (args.adapter / "adapter_config.json").exists():
            raise SystemExit(
                f"[m4] Adapter not found at {args.adapter} "
                f"(no adapter_config.json inside). "
                f"Did M3 training finish and models/lora_adapter/ get pulled?"
            )

    if args.output is None:
        args.output = default_output_path(args.adapter)

    mode = "fine-tuned (M4)" if args.adapter else "baseline (M2)"
    print(f"[gen] mode:     {mode}")
    print(f"[gen] eval set: {args.eval_set}")
    print(f"[gen] output:   {args.output}")

    model, tokenizer = load_model_and_tokenizer(args.adapter)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_set.open() as f_in, args.output.open("w") as f_out:
        for line in f_in:
            row = json.loads(line)
            completion = generate_completion(model, tokenizer, row["prompt"])
            f_out.write(json.dumps({"task_id": row["task_id"], "completion": completion}) + "\n")
            print(f"{row['task_id']}: generated {len(completion)} chars")

    print(f"[gen] wrote {args.output}")


if __name__ == "__main__":
    main()
