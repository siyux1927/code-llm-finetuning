"""
M5: GPTQ 4-bit quantization of the M3 fine-tuned model.

Merges the M3 LoRA adapter into the base Llama-2-7B (loaded in full fp16,
not bitsandbytes 4-bit — merging a LoRA delta into a 4-bit-quantized base
isn't a plain weight add, so this needs full precision; see
docs/grilling_m4_pre.md Q2 for why M4 deferred merging to "M5 deployment
stage"), then runs GPTQ 4-bit quantization using train_50.jsonl as
calibration data.

Produces two local directories, neither committed to git (both are
GB-scale — same reasoning as the base model itself never being committed):

    models/merged_fp16/  — full fp16 base+adapter merge. This IS one of the
                            two things M5 evaluates (the "no quantization"
                            reference point), so it stays on disk rather
                            than being deleted after quantizing.
    models/gptq_4bit/     — final GPTQ 4-bit checkpoint.

Design decisions documented in docs/grilling_m5_pre.md.

Runs on Colab T4 (CUDA required). Usage:
    huggingface-cli login
    python train/quantize_gptq.py
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# GPTQ config — see docs/grilling_m5_pre.md Q2/Q3
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128


def load_calibration_texts(train_path: Path) -> list[str]:
    """GPTQ calibration data — train_50.jsonl (prompt+completion concatenated)
    rather than a general corpus like wikitext2. See grilling_m5_pre.md Q4."""
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    return [r["prompt"] + r["completion"] for r in records]


def _patch_peft_gptqmodel_awq_compat() -> None:
    """peft's LoRA-merge dispatch unconditionally imports
    `gptqmodel.nn_modules.qlinear.gemm_awq.AwqGEMMQuantLinear` while probing
    for an AWQ target module match — even though this is a plain fp16 merge
    that never touches AWQ. Recent gptqmodel releases renamed that class to
    `AwqGEMMLinear`, so the import crashes before peft even checks whether
    AWQ applies. Alias the name so the import succeeds; no-op once a
    peft/gptqmodel pairing agrees on the name again."""
    import gptqmodel.nn_modules.qlinear.gemm_awq as _gemm_awq
    if not hasattr(_gemm_awq, "AwqGEMMQuantLinear") and hasattr(_gemm_awq, "AwqGEMMLinear"):
        _gemm_awq.AwqGEMMQuantLinear = _gemm_awq.AwqGEMMLinear


def merge_adapter(adapter_path: Path, merged_dir: Path) -> None:
    """Load base (fp16, unquantized) + M3 adapter, merge, save to disk."""
    print(f"[m5] loading base model in fp16: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )

    _patch_peft_gptqmodel_awq_compat()
    print(f"[m5] loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("[m5] merging adapter into base")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[m5] merged fp16 model saved to {merged_dir}")


def quantize(merged_dir: Path, calib_texts: list[str], output_dir: Path) -> None:
    """Reload the merged model through GPTQConfig — transformers performs the
    actual calibration + layer-by-layer quantization during this
    from_pretrained call, reading weights from merged_dir on disk."""
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    gptq_config = GPTQConfig(
        bits=GPTQ_BITS,
        group_size=GPTQ_GROUP_SIZE,
        dataset=calib_texts,
        tokenizer=tokenizer,
    )
    print(f"[m5] running GPTQ {GPTQ_BITS}-bit quantization "
          f"(calibration: {len(calib_texts)} examples)")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        quantization_config=gptq_config,
        device_map="auto",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[m5] GPTQ {GPTQ_BITS}-bit checkpoint saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--train-set", type=Path,
                        default=Path("data/processed/train_50.jsonl"))
    parser.add_argument("--merged-dir", type=Path, default=Path("models/merged_fp16"),
                        help="fp16 merge output. Not committed to git (GB-scale).")
    parser.add_argument("--output-dir", type=Path, default=Path("models/gptq_4bit"),
                        help="GPTQ checkpoint output. Not committed to git (GB-scale).")
    args = parser.parse_args()

    # Defensive check — same pattern as grilling_m4_pre.md Q8
    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(
            f"[m5] Adapter not found at {args.adapter} "
            f"(no adapter_config.json inside). "
            f"Did M3 training finish and models/lora_adapter/ get pulled?"
        )

    merge_adapter(args.adapter, args.merged_dir)
    calib_texts = load_calibration_texts(args.train_set)
    quantize(args.merged_dir, calib_texts, args.output_dir)


if __name__ == "__main__":
    main()
