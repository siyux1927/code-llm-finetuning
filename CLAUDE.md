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
- **M5 (Model Quantization): code ready, not yet run.**
  - `train/quantize_gptq.py` — loads base (fp16, unquantized) + `models/lora_adapter`, `merge_and_unload()`s into a standalone model, saves to `models/merged_fp16/`, then runs GPTQ 4-bit quantization (`transformers.GPTQConfig`, group_size=128, calibrated on `train_50.jsonl`) into `models/gptq_4bit/`. Neither output dir is committed (GB-scale, gitignored) — only `models/lora_adapter/` (M3's adapter) is.
  - `eval/generate_quantized.py` — same role as `generate_completions.py` but for a plain local model directory (fp16 merge or GPTQ checkpoint); imports `generate_completion` from `generate_completions.py` so decoding params (greedy/512/STOP_SEQUENCES) stay identical across M2/M4/M5.
  - `notebooks/run_quantization_colab.ipynb` — Colab launcher, now covering **both M5 and M6** (see below) in one session: merge → quantize → generate ×2 (fp16 merged, GPTQ-4bit) → score ×2 → size/pass@1 summary table → pushes only the small `quant_*_generations.jsonl` back to git.
  - Only quantizes the **M3 fine-tuned model** (not base), only **4-bit** (8-bit variant dropped — see `README.md` Decision 7), calibration reuses `train_50.jsonl`. Full rationale in `docs/grilling_m5_pre.md`.
  - **Note for later**: the "fp16 merged" pass@1 from this run is a different number from M4's 16.67% — M4 ran 4-bit-quantized base + LoRA adapter overlay, M5's fp16 point is the merged model at full precision. Not a contradiction if both show up in the blog; see `grilling_m5_pre.md` Q5.
  - Still to do: run on Colab (needs `optimum` + `gptqmodel`, added to `requirements-colab.txt`), fill in the pass@1/size numbers.
- **M6 (Inference Optimization / vLLM): code ready, not yet run.**
  - `eval/generate_vllm.py` — loads `models/gptq_4bit` via vLLM, times a single-request call and a batch-114 call, writes `data/processed/quant_gptq4bit_vllm_generations.jsonl`. Imports `STOP_SEQUENCES`/`MAX_NEW_TOKENS` from `generate_completions.py` so decoding stays identical to the HF-side comparison.
  - Compares **vLLM vs plain HF `generate()` only** — both on the same M5 GPTQ-4bit checkpoint. Does not re-test fp16 (that quality/size tradeoff is M5's question, not M6's).
  - Same notebook as M5 (`run_quantization_colab.ipynb`) — merged deliberately because M5's GPTQ checkpoint is GB-scale and not committed, so a separate M6 notebook would start with an empty disk and have to redo the quantization just to get it back. Full rationale in `docs/grilling_m6_pre.md`.
  - Notebook has explicit smoke-test cells (Step 3.5) before the expensive 7B steps — tiny-model GPTQ quantization dry run, tiny-model vLLM dry run, free-VRAM check — so library/driver/VRAM problems surface in seconds instead of after a 20-minute run. `requirements-colab.txt` now includes `vllm`.
  - Still to do: run on Colab, confirm vLLM's pass@1 matches M5's GPTQ-4bit number, fill in the three timing numbers (HF total / vLLM single-request / vLLM batch-114).
- **Next up after M5/M6**: M7 (cost analysis, using M6's throughput numbers) — see `README.md`.

## Working conventions

- Interesting problems, ambiguities, or design forks hit during implementation get filed as GitHub issues on this repo (not a formal spec process — just a running log of "things worth remembering"). Issues are written in Chinese.
- Git commit messages are written in Chinese, and do not include a `Co-Authored-By` trailer.
- Prose docs under `docs/` and top-level notes (TODO.md, etc.) are written in Chinese. `README.md` and `CLAUDE.md` themselves stay bilingual — English scaffolding, Chinese when it reads more naturally.
- No formal spec/ticket pipeline in use for this project — it's a learn-by-building project, developed interactively and incrementally.
