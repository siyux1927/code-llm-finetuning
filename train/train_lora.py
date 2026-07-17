"""
M3: QLoRA fine-tuning of Llama-2-7B on 50 canonical HumanEval solutions.

Loads the same 4-bit-quantized Llama-2-7B used in M2 (so M2→M4 improvement is
attributable to LoRA, not to a quantization change), attaches a LoRA adapter,
and trains for a small number of steps on `train_50.jsonl`. Produces a
portable adapter under `models/lora_adapter/` that M4 will load on top of the
base model.

Design decisions (rank, alpha, LR, epochs, loss masking, etc.) are documented
in `docs/grilling_m3_pre.md`. Read that first if you want to know *why* any
number below is what it is — this script's job is to execute those decisions.

Runs on Colab T4 (CUDA required). Usage:
    huggingface-cli login
    python train/train_lora.py
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SEED = 42

# LoRA config — see docs/grilling_m3_pre.md Q2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",          # MLP
]

# Training hyperparams — see docs/grilling_m3_pre.md Q3
EPOCHS = 5
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
MAX_GRAD_NORM = 0.3


def load_base_model_and_tokenizer():
    """Load 4-bit quantized Llama-2-7B + tokenizer, ready for QLoRA training."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Llama tokenizer has no pad_token by default; use eos as pad, matches M2.
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    # Disable KV cache — required with gradient checkpointing, and irrelevant
    # for training anyway.
    model.config.use_cache = False
    # Sets up gradient checkpointing hooks, upcast LayerNorm to fp32, enables
    # input_require_grads (needed for grad checkpointing through frozen 4-bit
    # weights). Standard QLoRA prep step.
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def attach_lora(model):
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    return get_peft_model(model, config)


def build_dataset(train_path: Path, tokenizer) -> Dataset:
    """Tokenize train_50.jsonl with completion-only loss masking.

    For each record:
      input_ids = [BOS] + tokenize(prompt) + tokenize(completion) + [EOS]
      labels    = [-100] * (1 + len(prompt_tokens)) + completion_tokens + [EOS]

    Tokenizing prompt and completion separately (both with add_special_tokens
    =False) avoids BPE-merge ambiguity at the boundary — see M3 grill Q5.
    """
    records = [json.loads(line) for line in train_path.read_text().splitlines()]

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    examples = []
    for r in records:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]

        input_ids = [bos_id] + prompt_ids + completion_ids + [eos_id]
        # Prompt (including BOS) is masked; completion + EOS is learned.
        labels = [-100] * (1 + len(prompt_ids)) + completion_ids + [eos_id]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(labels) == len(attention_mask)
        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        })

    return Dataset.from_list(examples)


def make_padding_collator(tokenizer):
    """Right-pad a batch to its longest sequence, keeping labels aligned.

    HF's DataCollatorForLanguageModeling won't preserve our -100 mask on the
    prompt side, so we roll our own.
    """
    pad_id = tokenizer.pad_token_id

    def collate(examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for e in examples:
            pad_len = max_len - len(e["input_ids"])
            batch["input_ids"].append(e["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(e["attention_mask"] + [0] * pad_len)
            batch["labels"].append(e["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return collate


def sanity_check(ds: Dataset, tokenizer) -> None:
    """Print the first example's tokenization boundary so we can eyeball whether
    completion-only masking is landing where we think it is."""
    e = ds[0]
    boundary = next(i for i, l in enumerate(e["labels"]) if l != -100)
    prompt_str = tokenizer.decode(e["input_ids"][:boundary])
    completion_str = tokenizer.decode(e["input_ids"][boundary:])
    print("[sanity] --- first training example ---")
    print(f"[sanity] total tokens: {len(e['input_ids'])}, prompt tokens: {boundary}")
    print(f"[sanity] prompt (last 80 chars): ...{prompt_str[-80:]!r}")
    print(f"[sanity] completion (first 80 chars): {completion_str[:80]!r}")
    print("[sanity] --- end ---")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-set", type=Path,
                        default=Path("data/processed/train_50.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("train/output"),
                        help="Trainer output dir (checkpoints, logs). Gitignored.")
    parser.add_argument("--adapter-dir", type=Path,
                        default=Path("models/lora_adapter"),
                        help="Final LoRA adapter save path. Committed to git.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    print(f"[m3] loading base model + tokenizer: {MODEL_NAME}")
    model, tokenizer = load_base_model_and_tokenizer()

    print("[m3] attaching LoRA adapter")
    model = attach_lora(model)
    model.print_trainable_parameters()

    print(f"[m3] building dataset from {args.train_set}")
    ds = build_dataset(args.train_set, tokenizer)
    print(f"[m3] dataset: {len(ds)} training examples")
    sanity_check(ds, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=make_padding_collator(tokenizer),
    )

    print("[m3] training...")
    trainer.train()

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))
    print(f"[m3] final adapter + tokenizer saved to {args.adapter_dir}")


if __name__ == "__main__":
    main()
