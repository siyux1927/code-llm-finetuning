# CLAUDE.md

Working notes for continuing this project across machines with Claude Code. Design rationale (why HumanEval, why Llama 2-7B, why QLoRA, etc.) lives in `README.md` — read that first for context, this file is just "where things stand" and "how to run things".

## Environment

- Python runs **only** inside the `myenv` env. Invocation depends on the machine:
  - **Windows PC** (branch `dev/t`): `conda run -n myenv <cmd>`
  - **Mac** (branch `dev/mbp`): `micromamba run -n myenv <cmd>`
- Deps are tracked in `requirements.txt` (install with the machine's respective `<manager> run -n myenv pip install -r requirements.txt`).
- Neither machine has CUDA — only CPU-bound work (data prep, small scripts, JSON/notebook validation, unit tests) runs locally. GPU work (baseline inference, LoRA/QLoRA training, eval generation) needs Colab, per README's VRAM-budget decisions.

## Status

- **M1 (Data Preparation): done.** `data/scripts/prepare_data.py` pulls the 164-problem `openai/openai_humaneval` dataset, shuffles with a fixed seed (42), and splits into **disjoint** train / eval sets:
  - `data/processed/train_50.jsonl` — 50 problems, completion-style `{task_id, prompt, completion}`, for fine-tuning.
  - `data/processed/eval_114.jsonl` — 114 problems (all non-train), `{task_id, prompt, canonical_solution, test, entry_point}`, for M2/M4 pass@1 scoring. Sized to shrink pass@1 SE from ~4.8pp (N=50) to ~2.9pp (N=114) — see `docs/grilling_m1_m2.md`.
- **M2 (Baseline Testing): done.** Ran on Colab via `notebooks/run_baseline_colab.ipynb` (M2/M3/M4 all consolidated into this one notebook — do not modify it, it's the executed record).
  - `eval/scoring.py` — pass@1 scoring harness. Unit-tested locally (`eval/tests/test_scoring.py`).
  - `eval/generate_completions.py` — generated `data/processed/baseline_generations.jsonl` (114 completions, 4-bit Llama-2-7B, no adapter).
  - **Result: pass@1 = 9.65% (11/114), 95% CI ±5.42pp.**
- **M3 (LoRA Fine-tuning): done.**
  - `train/train_lora.py` — QLoRA (r=16, α=32, dropout=0.05, all-linear, 5 epochs, LR 2e-4 cosine, paged_adamw_8bit) on the 50-example train set. Train loss converged 0.5 → ~0.02-0.17 (healthy, no collapse-to-zero overfit signal).
  - Adapter downloaded from Colab as `lora_adapter.zip`, then **converted from fp32 → fp16 locally** (post-hoc dtype cast, no retraining) because the fp32 `adapter_model.safetensors` was 152.6 MB — over GitHub's 100 MB single-file limit, and bigger than the `docs/grilling_m3_pre.md` Q7 estimate of ~40-80 MB. fp16 cast brought it to 76.3 MB (verified: same 448 tensors, same shapes, max abs diff ~3.8e-6 — negligible, doesn't affect M2/M4 comparability since inference compute dtype is fp16 either way per `BitsAndBytesConfig`). Committed at `models/lora_adapter/` (~80 MB total with tokenizer files).
- **M4 (Fine-tuned Evaluation): done.**
  - `eval/generate_completions.py --adapter models/lora_adapter` → `data/processed/finetuned_generations.jsonl`.
  - **Result: pass@1 = 16.67% (19/114), 95% CI ±6.84pp.**
  - `eval/compare_results.py`: **delta +7.02pp, 95% CI half-width ±8.73pp → not significant at 95%.** Flipped: 10 fail→pass, 2 pass→fail (regressions).
  - Decision (made 2026-07-19, not revisited): don't chase the seed-bias diagnostic listed in `docs/grilling_m1_m2.md`/`grilling_m4_pre.md` — write up the result as-is, consistent with the project's own framing of this as a fine-tuning-mechanism demo rather than a SOTA attempt. See `docs/baseline_behavior.md` for the full disclosure language.
- **Blog Part 2** = M4 outputs; see `docs/baseline_behavior.md` for the pre-drafted narrative (now filled in with final numbers).
- **Next up**: Phase 2 (M5 quantization / M6 inference / M7 cost) — see `README.md`.

## Working conventions

- Interesting problems, ambiguities, or design forks hit during implementation get filed as GitHub issues on this repo (not a formal spec process — just a running log of "things worth remembering"). Issues are written in Chinese.
- Git commit messages are written in Chinese, and do not include a `Co-Authored-By` trailer.
- Prose docs under `docs/` and top-level notes (TODO.md, etc.) are written in Chinese. `README.md` and `CLAUDE.md` themselves stay bilingual — English scaffolding, Chinese when it reads more naturally.
- No formal spec/ticket pipeline in use for this project — it's a learn-by-building project, developed interactively and incrementally.
