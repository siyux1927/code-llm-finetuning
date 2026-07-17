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
  - `data/processed/eval_114.jsonl` — 114 problems (all non-train), `{task_id, prompt, canonical_solution, test, entry_point}`, for M2/M4 pass@1 scoring. Sized to shrink pass@1 SE from ~4.8pp (N=50) to ~2.9pp (N=114) — see `docs/grilling_m1_m2.md` when it exists, or the GitHub issue.
- **M2 (Baseline Testing): partially done.**
  - `eval/scoring.py` — pass@1 scoring harness (executes generated code against HumanEval tests, subprocess + timeout sandboxed). Reused unchanged by M4. Unit-tested locally (`eval/tests/test_scoring.py`, `pytest eval/tests/`), no GPU needed.
  - `eval/generate_completions.py` — generates completions from Llama-2-7B (4-bit-quantized). No `--adapter` = M2 baseline; `--adapter models/lora_adapter/` = M4 fine-tuned. Same quantization M3 uses, so M2→M4 improvement isn't conflated with a quantization change. **Not yet run** — needs a CUDA GPU (bitsandbytes doesn't work on Mac/MPS) and a Hugging Face token with the Llama 2 license accepted. Run on Colab: `pip install -r requirements.txt -r requirements-colab.txt`, then `huggingface-cli login`, then the script.
  - Still to do: actually run generation on Colab (via `notebooks/run_baseline_colab.ipynb`), which now also computes pass@1 in-place via the new `eval/score.py` CLI.
- **M3 (LoRA Fine-tuning): code ready, not yet run.**
  - `train/train_lora.py` — QLoRA training script (r=16, α=32, dropout=0.05, all-linear target modules; 5 epochs, LR 2e-4, cosine, paged_adamw_8bit). Completion-only loss masking, ~62 update steps total. Saves adapter to `models/lora_adapter/` (committed to git), Trainer intermediate outputs to `train/output/` (gitignored). See `docs/grilling_m3_pre.md` for the full decision matrix.
  - `notebooks/run_train_lora_colab.ipynb` — Colab launcher, same pattern as M2's notebook.
  - Still to do: run on Colab (needs the same HF/Llama-2 access as M2), commit the adapter back, then M4.
- **M4 (Fine-tuned Evaluation): code ready, not yet run.**
  - Reuses `eval/generate_completions.py` with `--adapter models/lora_adapter` to load base + LoRA overlay. Same 4-bit + greedy + 512 tokens + STOP_SEQUENCES as M2 for clean attribution.
  - `eval/score.py` — CLI wrapper for `eval/scoring.py`; reports pass@1 with 95% CI half-width.
  - `eval/compare_results.py` — takes baseline + fine-tuned generations, prints delta, significance test, flipped problems, side-by-side samples for blog material.
  - `notebooks/run_finetuned_colab.ipynb` — Colab launcher; runs generation → pass@1 → compare in one flow.
  - See `docs/grilling_m4_pre.md` for the decision matrix.
- **Blog Part 2** = M4 outputs; see `docs/baseline_behavior.md` for the pre-drafted narrative.

## Working conventions

- Interesting problems, ambiguities, or design forks hit during implementation get filed as GitHub issues on this repo (not a formal spec process — just a running log of "things worth remembering"). Issues are written in Chinese.
- Git commit messages are written in Chinese, and do not include a `Co-Authored-By` trailer.
- Prose docs under `docs/` and top-level notes (TODO.md, etc.) are written in Chinese. `README.md` and `CLAUDE.md` themselves stay bilingual — English scaffolding, Chinese when it reads more naturally.
- No formal spec/ticket pipeline in use for this project — it's a learn-by-building project, developed interactively and incrementally.
